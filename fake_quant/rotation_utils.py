import model_utils
import torch
import typing
import utils
import transformers
import torch.nn as nn
import tqdm
import math
import logging
import quant_utils
from hadamard_utils import random_hadamard_matrix, apply_exact_had_to_linear, is_pow2
from fast_hadamard_transform import hadamard_transform
import os

def fuse_ln_linear(layernorm: torch.nn.Module, linear_layers: typing.Iterable[torch.nn.Linear]) -> None:
    """
    Fuse the linear operations in Layernorm into the adjacent linear blocks.
    """
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype
        
        target_device = linear.weight.device
        
        W_ = linear.weight.data.double()
        ln_weight = layernorm.weight.data.double().to(target_device)
        
        linear.weight.data = (W_ * ln_weight).to(linear_dtype)

        if hasattr(layernorm, 'bias') and layernorm.bias is not None:
            if linear.bias is None:
                linear.bias = torch.nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64, device=target_device))
            
            ln_bias = layernorm.bias.data.double().to(target_device)
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, ln_bias)
            linear.bias.data = linear.bias.data.to(linear_dtype)


def bake_mean_into_linear(linear: torch.nn.Linear) -> None:
    """
    This function takes a linear layer and subtracts the means from the
    weights and biases. This will result in the linear layer performing
    the mean substitution which is usually done inside layernorm.
    """
    linear_dtype = linear.weight.dtype
    W_ = linear.weight.data.double()
    linear.weight.data = W_ - W_.mean(dim=-2, keepdim=True)
    linear.weight.data = linear.weight.data.to(linear_dtype)
    if linear.bias is not None:
        b_ = linear.bias.data.double()
        linear.bias.data = b_ - b_.mean()
        linear.bias.data = linear.bias.data.to(linear_dtype)


def fuse_layer_norms(model):

    model_type = model_utils.get_model_type(model)

    kwargs = {'model': model, 'model_type': model_type}

    # embedding fusion
    for W in model_utils.get_embeddings(**kwargs):
        W_ = W.weight.data.double()
        W.weight.data = (W_ - W_.mean(dim=-1, keepdim=True)).to(W.weight.data.dtype)

    layers = model_utils.get_transformer_layers(**kwargs)

    # fuse the linear operations in Layernorm into the adjacent linear blocks.
    for layer in layers:

        # fuse the input layernorms into the linear layers
        if model_type == model_utils.LLAMA_MODEL:
            fuse_ln_linear(layer.post_attention_layernorm, [layer.mlp.up_proj, layer.mlp.gate_proj])
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj,
                           layer.self_attn.k_proj, layer.self_attn.v_proj])
        elif model_type == model_utils.OPT_MODEL:
            fuse_ln_linear(
                layer.self_attn_layer_norm, [
                    layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
            fuse_ln_linear(layer.final_layer_norm, [layer.fc1])
        elif model_type == model_utils.DEEPSEEK_MODEL:
            # fuse input layernorm into the self-attention q, k, v projections
            fuse_ln_linear(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])

            moe_inputs = []
            # check this specific layer is an MoE layer
            if hasattr(layer.mlp, 'experts'): 
                moe_inputs.append(layer.mlp.gate)
                if hasattr(layer.mlp, 'shared_experts'):
                    moe_inputs.extend([layer.mlp.shared_experts.up_proj, layer.mlp.shared_experts.gate_proj])
                for expert in layer.mlp.experts:
                    moe_inputs.extend([expert.up_proj, expert.gate_proj])
            else: 
                # dense layer
                moe_inputs.extend([layer.mlp.up_proj, layer.mlp.gate_proj])
                
            fuse_ln_linear(layer.post_attention_layernorm, moe_inputs)
        else:
            raise ValueError(f'Unknown model type {model_type}')

        if model_type == model_utils.OPT_MODEL:
            bake_mean_into_linear(layer.self_attn.out_proj)
            bake_mean_into_linear(layer.fc2)

    fuse_ln_linear(model_utils.get_pre_head_layernorm(**kwargs), [model_utils.get_lm_head(**kwargs)])

    # model_utils.replace_modules(
    #     model,
    #     transformers.models.llama.modeling_llama.LlamaRMSNorm if model_type == model_utils.LLAMA_MODEL else torch.nn.LayerNorm,
    #     lambda _: model_utils.RMSN(model.config.hidden_size),
    #     replace_layers=False,
    # )

    # fix: dynamically get the exact RMSNorm class from the loaded model : prevents the double-scaling bug on DeepSeek/Mixtral architectures
    if hasattr(layers[0], 'input_layernorm'):
        norm_class = type(layers[0].input_layernorm)
    elif hasattr(layers[0], 'self_attn_layer_norm'):
        norm_class = type(layers[0].self_attn_layer_norm)
    else:
        norm_class = torch.nn.LayerNorm

    model_utils.replace_modules(
        model,
        norm_class,  # dynamically targets DeepseekRMSNorm or MixtralRMSNorm
        lambda _: model_utils.RMSN(model.config.hidden_size),
        replace_layers=False,
    )


def random_orthogonal_matrix(size, device):
    """
    Generate a random orthogonal matrix of the specified size.
    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

    Args:
    size (int): The size of the matrix (size x size).

    Returns:
    torch.Tensor: An orthogonal matrix of the specified size.
    """
    torch.cuda.empty_cache()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size, mode, device=utils.DEV):
    if mode == 'random':
        return random_orthogonal_matrix(size, device)
    elif mode == 'hadamard':
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f'Unknown mode {mode}')


def rotate_embeddings(model, Q: torch.Tensor) -> None:
    # Rotate the embeddings.
    model_type = model_utils.model_type_extractor(model)
    for W in model_utils.get_embeddings(model, model_type):
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, Q, model_type) -> None:
    # Rotate the WQ, WK and WV matrices of the self-attention layer.
    for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
        dtype = W.weight.dtype
        W_ = W.weight.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, Q, model_type) -> None:
    # Rotate output matrix of the self-attention layer.
    if model_type in [model_utils.LLAMA_MODEL, model_utils.MISTRAL_MODEL, model_utils.DEEPSEEK_MODEL]:
        W = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        W = layer.self_attn.out_proj
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(Q.T, W_).to(device="cpu", dtype=dtype)
    if W.bias is not None:
        b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
        W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, Q, model_type, l_id, smooth):
    mlp_inputs = {}
    
    if model_type == model_utils.OPT_MODEL:
        mlp_inputs["fc1"] = layer.fc1
    elif model_type == model_utils.LLAMA_MODEL:
        mlp_inputs["mlp.up_proj"] = layer.mlp.up_proj
        mlp_inputs["mlp.gate_proj"] = layer.mlp.gate_proj
    elif 'mixtral' in str(model_type).lower():
        mlp_inputs["block_sparse_moe.gate"] = layer.block_sparse_moe.gate
        for i, expert in enumerate(layer.block_sparse_moe.experts):
            mlp_inputs[f"block_sparse_moe.experts.{i}.w1"] = expert.w1
            mlp_inputs[f"block_sparse_moe.experts.{i}.w3"] = expert.w3
    elif 'deepseek' in str(model_type).lower():
        # check : a dense layer or an MoE layer
        if hasattr(layer.mlp, 'experts'):
            mlp_inputs["mlp.gate"] = layer.mlp.gate
            if hasattr(layer.mlp, 'shared_experts'):
                mlp_inputs["mlp.shared_experts.up_proj"] = layer.mlp.shared_experts.up_proj
                mlp_inputs["mlp.shared_experts.gate_proj"] = layer.mlp.shared_experts.gate_proj
            for i, expert in enumerate(layer.mlp.experts):
                mlp_inputs[f"mlp.experts.{i}.up_proj"] = expert.up_proj
                mlp_inputs[f"mlp.experts.{i}.gate_proj"] = expert.gate_proj
        else:
            # a dense layer
            mlp_inputs["mlp.up_proj"] = layer.mlp.up_proj
            mlp_inputs["mlp.gate_proj"] = layer.mlp.gate_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    for name, W in mlp_inputs.items():
        dtype = W.weight.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)
        
        # Apply Smoothing dynamically based on layer name
        if smooth is not None:
            is_up_proj = ('up_proj' in name) or ('w3' in name)
            if is_up_proj:
                smooth_key = f'model.layers.{l_id}.' + name.replace('up_proj', 'down_smooth').replace('w3', 'down_smooth')
                if smooth_key in smooth:
                    down_smooth = smooth[smooth_key].to(device=utils.DEV, dtype=torch.float64)
                    up_W = W.to(device=utils.DEV, dtype=torch.float64)
                    up_W.weight.data = torch.div(up_W.weight.data.t(), down_smooth).t().to(device="cpu", dtype=dtype)

def rotate_mlp_output(layer, Q, model_type, use_r4, l_id, smooth):
    mlp_outputs = {}
    
    if model_type == model_utils.OPT_MODEL:
        mlp_outputs["fc2"] = layer.fc2
    elif model_type == model_utils.LLAMA_MODEL:
        mlp_outputs["mlp.down_proj"] = layer.mlp.down_proj
    elif 'mixtral' in str(model_type).lower():
        for i, expert in enumerate(layer.block_sparse_moe.experts):
            mlp_outputs[f"block_sparse_moe.experts.{i}.w2"] = expert.w2
    elif 'deepseek' in str(model_type).lower():
        # ROBUST CHECK: Is this a dense layer or an MoE layer?
        if hasattr(layer.mlp, 'experts'):
            if hasattr(layer.mlp, 'shared_experts'):
                mlp_outputs["mlp.shared_experts.down_proj"] = layer.mlp.shared_experts.down_proj
            for i, expert in enumerate(layer.mlp.experts):
                mlp_outputs[f"mlp.experts.{i}.down_proj"] = expert.down_proj
        else:
            # It's a dense layer
            mlp_outputs["mlp.down_proj"] = layer.mlp.down_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    for name, W in mlp_outputs.items():
        dtype = W.weight.data.dtype
        W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
        W.weight.data = torch.matmul(Q.T, W_)
        
        # Apply Smoothing
        if smooth is not None:
            smooth_key = f'model.layers.{l_id}.' + name.replace('down_proj', 'down_smooth').replace('w2', 'down_smooth')
            if smooth_key in smooth:
                down_smooth = smooth[smooth_key].to(device=utils.DEV, dtype=torch.float64)
                W.weight.data = torch.mul(W.weight.data, down_smooth)
                
        W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)

        if use_r4:
            # Apply exact (inverse) hadamard on the weights of mlp output
            apply_exact_had_to_linear(W, had_dim=-1, output=False)
            
        if W.bias is not None:
            b = W.bias.data.to(device=utils.DEV, dtype=torch.float64)
            W.bias.data = torch.matmul(Q.T, b).to(device="cpu", dtype=dtype)

def matmul_hadU_cuda_had(X, hadK, transpose=False):
    '''
    Apply hadamard transformation.
    It reshapes X and applies Walsh-Hadamard transform to the last dimension.
    Then, it will multiply the retult by another hadamard matrix.
    '''
    from fast_hadamard_transform import hadamard_transform
    from hadamard_utils import get_had172
    n = X.shape[-1]
    K = hadK.shape[-1]

    if transpose:
        hadK = hadK.T.contiguous()
    input = X.float().cuda().view(-1, K, n // K)
    input = hadamard_transform(input.contiguous(), scale=1 / math.sqrt(n))
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.to(X.device).to(X.dtype).reshape(
        X.shape)


def rotate_faster_down_proj(layer, model_type, hardK):
    from fast_hadamard_transform import hadamard_transform
    if model_type == model_utils.LLAMA_MODEL:
        W = layer.mlp.down_proj
    else:
        raise ValueError(f'Faster MLP is onlu supported for LLaMa models!')

    dtype = W.weight.data.dtype
    W.weight.data = matmul_hadU_cuda_had(W.weight.data.float().cuda(), hardK)
    W.weight.data = W.weight.data.to(device="cpu", dtype=dtype)


def rotate_head(model, Q: torch.Tensor) -> None:
    # Rotate the head.
    W = model_utils.get_lm_head(model, model_type=model_utils.model_type_extractor(model))
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W.weight.data = torch.matmul(W_, Q).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, model_type, head_dim, kv_head,
                   l_id, use_r2, r2_mode, r2_matrices=None, # Changed from r2_path
                   smooth=None):
    v_proj = layer.self_attn.v_proj
    if model_type in [model_utils.LLAMA_MODEL, model_utils.MISTRAL_MODEL, model_utils.DEEPSEEK_MODEL]:
        o_proj = layer.self_attn.o_proj
    elif model_type == model_utils.OPT_MODEL:
        o_proj = layer.self_attn.out_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')

    if use_r2 == 'online':
        apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True)
        # apply_exact_had_to_linear(o_proj, had_dim=-1, output=False)
        apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False)

    if 'offline' in use_r2:  
        if r2_matrices is not None:
            # Grab it from RAM, not the hard drive
            Q = r2_matrices[f"model.layers.{l_id}.self_attn.R2"].to(device=utils.DEV, dtype=torch.float64)
        else:
            Q = get_orthogonal_matrix(head_dim, r2_mode)
            if len(Q.shape) != 3:
                Q = Q.repeat(kv_head, 1, 1)
                
        apply_multi_head_rotate(v_proj, Q, head_dim, l_id, kv_head, output=True, smooth=smooth)
        apply_multi_head_rotate(o_proj, Q, head_dim, l_id, kv_head, output=False, smooth=smooth)


def apply_multi_head_rotate(module, Q, head_dim, l_id,
                            kv_head, output=False,
                            smooth=None):
    assert isinstance(module, torch.nn.Linear)

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    num_head = init_shape[1] // head_dim
    n_rep = num_head // kv_head
    W_ = W_.to(device=utils.DEV, dtype=torch.float64)

    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        W_ = W_.reshape(-1, kv_head, head_dim).transpose(0, 1)
        if smooth is not None:
            o_smooth = smooth['model.layers.{}.self_attn.o_smooth'.format(
                l_id)].to(device=utils.DEV, dtype=torch.float64)
            W_ = W_ / o_smooth.view(kv_head, 1, head_dim)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(transposed_shape).t()
    else:
        W_ = W_.reshape(-1, init_shape[1] // head_dim,
                        head_dim).transpose(0, 1)
        if len(Q.shape) == 3:
            Q = Q[:, None, :, :].expand(kv_head, n_rep, head_dim, head_dim)
            Q = Q.reshape(num_head, head_dim, head_dim)
        if smooth is not None:
            o_smooth = smooth['model.layers.{}.self_attn.o_smooth'.format(l_id)]
            o_smooth = o_smooth[:, None, :].expand(kv_head, n_rep, head_dim)
            o_smooth = o_smooth.reshape(num_head, head_dim).to(device=utils.DEV, dtype=torch.float64)
            W_ = W_ * o_smooth.view(kv_head, 1, head_dim)
        W_ = torch.matmul(W_, Q)
        W_ = W_.transpose(0, 1).reshape(init_shape)

    module.weight.data = W_.to(device=dev, dtype=dtype)


@torch.inference_mode()
def rotate_model(model, args):
    
    Q = get_orthogonal_matrix(model.config.hidden_size,
                                  args.rotate_mode)

    smooth_scale = torch.load(args.smooth) if args.smooth is not None else None
    if smooth_scale is not None:
        logging.info(f'Use smooth scale load from: {args.smooth}')

    r2_matrices = None
    # if args.use_r2 == 'offline' and args.r2_path is not None:
    #     # Check if the path points to a file, not a directory
    #     if os.path.isfile(args.r2_path):
    #         logging.info(f'Loading R2 matrices from: {args.r2_path}')
    #         r2_matrices = torch.load(args.r2_path, map_location='cpu')
    #     else:
    #         raise ValueError(f"args.r2_path must point to a .pt file, but got: {args.r2_path}")

    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    kv_head = config.num_key_value_heads

    model_type = model_utils.model_type_extractor(model)
    rotate_embeddings(model, Q)
    rotate_head(model, Q)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model,
                                                model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
        rotate_attention_inputs(layers[idx], Q, model_type)
        rotate_attention_output(layers[idx], Q, model_type)
        rotate_mlp_input(layers[idx], Q, model_type, idx, smooth_scale)
        rotate_mlp_output(layers[idx], Q, model_type, args.use_r4, idx, smooth_scale)
        if args.use_r2 != 'none':
            rotate_ov_proj(layers[idx], model_type, head_dim, kv_head, idx, args.use_r2,
                           args.rotate_mode, r2_matrices, smooth_scale)


@torch.inference_mode
def online_rotate(module, inp):
    x = torch.nn.functional.linear(inp[0], module.Q)
    return (x,) + inp[1:]


def register_online_rotation(module, Q: torch.Tensor):
    assert not hasattr(module, 'Q')
    module.register_buffer('Q', Q.T.to(module.weight.data))  # Note F.linear(x, A) performs x@A.T

    # We use forward_pre_hook because we capture the input using forward_hook, which could then capture the rotated input.
    # If we implement in the forward() the un-rotated original input will be captured.
    module.rotate_handle = module.register_forward_pre_hook(online_rotate)


class QKRotationWrapper(torch.nn.Module):

    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(head_dim), f'Only power of 2 head_dim is supported for K-cache Quantization!'
        self.func = func
        self.k_quantizer = quant_utils.ActQuantizer()
        self.k_bits = 16
        if kwargs is not None:
            assert kwargs['k_groupsize'] in [-1,
                                             head_dim], f'Only token-wise/{head_dim}g quantization is supported for K-cache'
            self.k_bits = kwargs['k_bits']
            self.k_groupsize = kwargs['k_groupsize']
            self.k_sym = kwargs['k_sym']
            self.k_clip_ratio = kwargs['k_clip_ratio']
            self.use_r3 = kwargs['use_r3']
            self.k_quantizer.configure(bits=self.k_bits, groupsize=-1,  # we put -1 to be toke-wise quantization and handle head-wise quantization by ourself
                                       sym=self.k_sym, clip_ratio=self.k_clip_ratio)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        if self.use_r3:
            q = hadamard_transform(q.float(), scale=1 / math.sqrt(q.shape[-1])).to(dtype)
            k = hadamard_transform(k.float(), scale=1 / math.sqrt(k.shape[-1])).to(dtype)
        (bsz, num_heads, seq_len, head_dim) = k.shape

        if self.k_groupsize == -1:  # token-wise quantization
            token_wise_k = k.transpose(1, 2).reshape(-1, num_heads * head_dim)  # Source code: (-1, self.config.hidden_size) throws an error
            self.k_quantizer.find_params(token_wise_k)
            k = self.k_quantizer(token_wise_k).reshape((bsz, seq_len, num_heads, head_dim)).transpose(1, 2).to(q)
        else:  # head-wise quantization
            per_head_k = k.reshape(-1, head_dim)  # Source code: per_head_k = k.view(-1, head_dim) throws an error
            self.k_quantizer.find_params(per_head_k)
            k = self.k_quantizer(per_head_k).reshape((bsz, num_heads, seq_len, head_dim)).to(q)

        self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(module, function_name, *args, **kwargs):
    '''
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    '''
    import monkeypatch
    import functools
    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(module, "forward",
                                                                    function_name, functools.partial(QKRotationWrapper, *args, **kwargs))
    setattr(module, attr_name, wrapper)