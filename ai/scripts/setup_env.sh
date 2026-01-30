#!/bin/bash

echo "Setting up WCAG-VLM Environment on HPC..."

# Load modules
# Load modules (if available)
if type module >/dev/null 2>&1; then
    module load cuda/12.1 python/3.10
else
    echo "Module command not found, using system default python"
fi

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/screenshots data/annotations data/results models logs

echo "Setup complete!"
