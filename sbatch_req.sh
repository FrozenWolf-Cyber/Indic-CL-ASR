#!/bin/bash

#SBATCH --job-name=cl_sam2_ddp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=/scratch/gokuladethya.cse.nitt/indiaai/slurm-loop-%j.out
#SBATCH --time=6-00:00:00

WATCH_DIR="/scratch/gokuladethya.cse.nitt/indiaai/jobs"
PROCESSED_DIR="/scratch/gokuladethya.cse.nitt/indiaai/processed_jobs"

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
head_node=${nodes[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

echo "Using head node: $head_node ($head_node_ip)"
source /scratch/gokuladethya.cse.nitt/miniconda3/etc/profile.d/conda.sh
conda activate indiaai

export LOGLEVEL=INFO
export TORCH_RUN_RDZV_TIMEOUT=360000
export TORCH_DISTRIBUTED_DEBUG=INFO
export WANDB_API_KEY=283c41dda88b658ba85c2d8ee7d37230f3341d8c
export WANDB_MODE=online

echo "Infinite job monitor running. Watching for scripts in $WATCH_DIR..."

while true; do
    for script in $(ls "$WATCH_DIR"/*.sh 2>/dev/null | sort -V); do
        [ -e "$script" ] || continue  # No .sh file found

        script_name=$(basename "$script")

        echo "[$(date)] Found new script: $script_name"
        chmod +x "$script"
        echo "Running $script_name with srun..."
    	cmd=$(cat "$script")
        cmd="${cmd//\$head_node_ip/$head_node_ip}"
        cmd="${cmd//\$RANDOM/$RANDOM}"
    	echo "Executing: $cmd"
        ### add sleep
        sleep 1
        srun $cmd

        # Mark as processed
        mv "$script" "$PROCESSED_DIR/$script_name"
        echo "[$(date)] Finished $script_name"
    done
    sleep 1  # Check every 10 seconds
done
