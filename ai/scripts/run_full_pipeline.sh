#!/bin/bash
#SBATCH --job-name=wcag-vlm
#SBATCH --output=logs/wcag_vlm_%j.out
#SBATCH --error=logs/wcag_vlm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=all

# Load environment
module load cuda/12.1 python/3.10
source venv/bin/activate

# Default arguments
NUM_NODES=${1:-2}
GPUS_PER_NODE=${2:-4}
NUM_EPOCHS=${3:-3}

echo "Starting WCAG-VLM Pipeline on $(hostname)"
echo "Nodes: $NUM_NODES, GPUs/node: $GPUS_PER_NODE"

# Run orchestration
python orchestrate_pipeline.py --mode full --num-epochs $NUM_EPOCHS --batch-size 8

echo "Job completed at $(date)"
