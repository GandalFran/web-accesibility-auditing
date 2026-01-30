#!/bin/bash
#SBATCH --job-name=wcag-baseline
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=all

# Load environment
module load python/3.10
source venv/bin/activate
# export HF_HUB_OFFLINE=1  <-- DISABLED to allow streaming
# export HF_TOKEN=your_token_here

echo "Starting Code Baseline Audit..."
python scripts/run_code_baseline.py
echo "Baseline Audit Completed."
