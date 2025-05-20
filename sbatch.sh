#!/bin/bash

#SBATCH --job-name=cl_sam2_ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/scratch/gokuladethya.cse.nitt/fyp/slurm-%j.out
#SBATCH --time=4-00:00:00

echo "Allocated Gokul node: jobid:"
squeue -a | grep gok
echo "------------------------------------"

# Get hostnames of allocated nodes
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
echo "SLURM_NODELIST: $SLURM_JOB_NODELIST"
scontrol show hostnames $SLURM_JOB_NODELIST

nodes_array=($nodes)
head_node=${nodes_array[0]}

# Get IP address of the head node
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"

export LOGLEVEL=INFO

# Setup environment
conda init bash
source /scratch/gokuladethya.cse.nitt/miniconda3/etc/profile.d/conda.sh

conda activate indiaai
export WANDB_API_KEY=WANDB_API_KEY
# Diagnostics
srun python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
srun nvidia-smi

# Launch training
echo "Launching torchrun..."
ls /scratch/gokuladethya.cse.nitt/image-segmentation/

export TORCH_RUN_RDZV_TIMEOUT=360000
export TORCH_DISTRIBUTED_DEBUG=INFO
export WANDB_MODE=online
# export TORCHELASTIC_ENABLE_FILE_TIMER=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=0
# export NCCL_NET_GDR_LEVEL=0

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$head_node_ip:29500 \
  /scratch/gokuladethya.cse.nitt/CL-IndicASR/cl_baseline.py \
  --notes "CL-baseline-naive"


