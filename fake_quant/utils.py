import torch
import random
import numpy as np
import os
import atexit
import logging
from tqdm import tqdm


from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

supported_models = [
    'meta-llama/Llama-2-7b-hf',
    'meta-llama/Llama-2-13b-hf',
    'meta-llama/Llama-2-70b-hf',
    'meta-llama/Meta-Llama-3-8B',
    'meta-llama/Llama-3.1-8B-Instruct',
    'meta-llama/Meta-Llama-3-70B',
    'meta-llama/Llama-3.1-70B',
    'facebook/opt-125m',
    'deepseek-ai/deepseek-moe-16b-chat',
]
supported_datasets = ['wikitext2', 'ptb', 'c4']

# These flags disable using TensorFloat-32 tensor cores (to avoid numerical issues)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
DEV = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def llama_down_proj_groupsize(model, groupsize):

    assert groupsize > 1, 'groupsize should be greater than 1!'

    if model.config.intermediate_size % groupsize == 0:
        logging.info(f'(Act.) Groupsiz = Down_proj Groupsize: {groupsize}')
        return groupsize

    group_num = int(model.config.hidden_size / groupsize)
    assert groupsize * group_num == model.config.hidden_size, 'Invalid groupsize for llama!'

    down_proj_groupsize = model.config.intermediate_size // group_num
    assert down_proj_groupsize * group_num == model.config.intermediate_size, 'Invalid groupsize for down_proj!'
    logging.info(f'(Act.) Groupsize: {groupsize}, Down_proj Groupsize: {down_proj_groupsize}')
    return down_proj_groupsize


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)


# Dump the log both to console and a log file.
def config_logging(log_file,
                   levels_to_log={logging.INFO, logging.ERROR},
                   to_console=True):
    # Ensure the path exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Define log format
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Custom filter
    class SpecificLevelFilter(logging.Filter):
        def filter(self, record):
            return record.levelno in levels_to_log

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(formatter)
    file_handler.addFilter(SpecificLevelFilter())  # Add filter

    # Console handler
    handlers = [file_handler]
    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(SpecificLevelFilter())  # Add filter
        handlers.append(console_handler)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG, handlers=handlers)

    # Ensure the buffer is written upon program exit
    atexit.register(logging.shutdown)


def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            logging.info(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


def distribute_model(model) -> None:
    """Distribute the model across available GPUs. NB: only implemented for Llama-2."""
    no_split_module_classes = ['LlamaDecoderLayer']
    max_memory = get_balanced_memory(
        model,
        no_split_module_classes=no_split_module_classes,
    )

    device_map = infer_auto_device_map(
        model, max_memory=max_memory, no_split_module_classes=no_split_module_classes
    )

    dispatch_model(
        model,
        device_map=device_map,
        offload_buffers=True,
        offload_dir="offload",
        state_dict=model.state_dict(),
    )

    cleanup_memory()


import torch
import os
import logging


def save_model_in_parts(model, save_qmodel_path, prefix='model_part', num_digits=5, target_file_size=10 * 1024**3):
    """
    Save the model into multiple parts based on file size (e.g., 10GB).
    :param model: PyTorch model to save
    :param save_qmodel_path: Model save path
    :param target_file_size: Target size of each file (unit: bytes, default 10GB)
    :param prefix: File name prefix (default 'model_part')
    :param num_digits: Number of digits for file index (default 5 digits)
    """
    state_dict = model.state_dict()

    # Calculate the size of each parameter of the model (automatically calculated based on the data type of the parameter)
    total_size = 0
    for param in state_dict.values():
        param_size = param.element_size() * param.numel()  # Get the parameter size (in bytes)
        total_size += param_size

    # Calculate the number of parts
    num_parts = (total_size + target_file_size - 1) // target_file_size  # Ceil

    logging.info(
        f"Total model size: {total_size / (1024**3):.2f} GB, divided into {num_parts} parts, each about {target_file_size / (1024**3):.2f} GB")

    # Save model in parts
    idx = 0
    current_part_size = 0  # The actual byte size of the current part
    part = {}

    for name, param in state_dict.items():
        param_size = param.element_size() * param.numel()  # Calculate the byte size of the current parameter
        current_part_size += param_size  # Accumulate the size of the current part
        part[name] = param  # Add current parameter to the part

        # If the size of the current part exceeds the target size, save and start a new part
        if current_part_size >= target_file_size:
            # Format the file name and ensure the index is 5 digits
            part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
            torch.save(part, os.path.join(save_qmodel_path, part_filename))
            logging.info(f"Saved part {idx + 1} of the model: {part_filename}, {num_parts} parts in total.")
            part = {}  # Clear the current part, start the next part
            current_part_size = 0  # Reset the size of the current part
            idx += 1

    # The last part
    if part:
        part_filename = f"{prefix}_{str(idx).zfill(num_digits)}.pth"
        torch.save(part, os.path.join(save_qmodel_path, part_filename))
        logging.info(f"Saved part {idx + 1} of the model: {part_filename}, {num_parts} parts in total.")

    logging.info("Model saving in parts completed.")


def load_model_in_parts(model, folder_path):
    """
    Load multiple parts of the model and assign them to the model block by block.
    :param model: PyTorch model to load
    :param folder_path: Folder path storing the part model files
    """
    # Get all .pth files in the folder and sort by name (ensure the order is correct)
    model_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pth')])

    # Load parts one by one
    with tqdm(total=len(model_files), desc="Loading Model Parts", unit="part") as pbar:
        for file_name in model_files:
            part = torch.load(os.path.join(folder_path, file_name), map_location='cpu')  # Load part

            model.load_state_dict(part, strict=False)  # Update model parameters (assign in real time)

            del part  # Release memory for the loaded part
            pbar.update(1)  # Update progress bar

    logging.info("Model loading completed.")