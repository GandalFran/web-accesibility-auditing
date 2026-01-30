#!/bin/bash
#SBATCH --job-name=wcag-vlm-quick
#SBATCH --output=logs/wcag_vlm_quick_%j.out
#SBATCH --error=logs/wcag_vlm_quick_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --partition=all

# Load environment
module load cuda/12.1 python/3.10
source venv/bin/activate

echo "Starting WCAG-VLM Project on $(hostname)"
echo "Dataset: Public HuggingFace WebSight (50 samples)"

# 1. Run Inference (Stage 2) - Skips Scraping
echo ">>> Running Stage 2: Inference (LLaVA-7b)..."
python orchestrate_pipeline.py --mode inference_only --models llava-7b

echo ">>> Running Stage 2: Inference (LLaVA-13b) [Comparison]..."
python orchestrate_pipeline.py --mode inference_only --models llava-13b

# 2. Run Analysis (Stage 6)
echo ">>> Running Stage 6: Comparative Analysis..."
python orchestrate_pipeline.py --mode analysis_only

echo "Job completed at $(date)"
