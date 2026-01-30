#!/bin/bash
# finish_deployment.sh
# Use this script to complete the deployment if the AI assistant encounters network timeouts.

HPC_USER="your_username"
HPC_HOST="hpc-login.example.com"
HPC_DIR="~/wcag-vlm"

echo "=== WCAG-VLM Manual Completion Script ==="
echo "1. Uploading data payload (data_upload.zip)..."
scp -o ConnectTimeout=60 data_upload.zip ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/

if [ $? -eq 0 ]; then
    echo "✔ Upload successful."
    
    echo "2. Unzipping data and submitting job on HPC..."
    ssh -o ConnectTimeout=60 ${HPC_USER}@${HPC_HOST} "cd ${HPC_DIR} && unzip -o data_upload.zip && sbatch scripts/run_quick_pipeline.sh"
    
    if [ $? -eq 0 ]; then
        echo "✔ Job submitted successfully!"
        echo "Monitor with: ssh ${HPC_USER}@${HPC_HOST} 'squeue --me'"
    else
        echo "✘ Error submitting job."
    fi
else
    echo "✘ Error uploading data. Please check your VPN/Network connection to ${HPC_HOST}."
fi
