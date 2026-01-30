#!/bin/bash
source venv/bin/activate
export HF_HOME=models/cache
mkdir -p data/results
python orchestrate_pipeline.py --mode inference_only --models llava-1.5-7b-hf
