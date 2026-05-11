#!/bin/bash
#SBATCH --job-name=eval     # Job name
#SBATCH --qos=h200_qos
#SBATCH --output=/storage/users/css/indranilp/slurm_logs/moe_quant_eval_llama270B_444_%j.log        # Standard output log
#SBATCH --error=/storage/users/css/indranilp/slurm_logs/moe_quant_eval_llama270B_444_%j.err         # Standard error log
#SBATCH --time=24:00:00                # Max run time (hh:mm:ss)
#SBATCH --nodes=1                      # Number of nodes (usually 1)
#SBATCH --cpus-per-task=16               # Number of CPU cores
#SBATCH --mem=200GB                      # Memory allocation
#SBATCH --partition=h200                 # Partition for h200 nodes
#SBATCH --account=css                 # Account name for group, adhering to KIAC policies
#SBATCH --gres=gpu:1              # Request 1 GPU

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate moe_quant

# Go to your project directory
cd /storage/users/css/indranilp/Project/MoE_Quant/fake_quant

# Run your Python script
bash eval_scripts/quarot_gptq_wxaykvz_test.sh 0 deepseek-ai/deepseek-moe-16b-chat 4 4 4