FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    wget \
    vim \
    htop \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create virtual environment
RUN python3.10 -m venv /workspace/venv
ENV PATH="/workspace/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt /workspace/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/ /workspace/src/
COPY scripts/ /workspace/scripts/
COPY data/ /workspace/data/ 2>/dev/null || true
COPY config/ /workspace/config/ 2>/dev/null || true

# Create necessary directories
RUN mkdir -p /workspace/data/{screenshots,annotations,results} && \
    mkdir -p /workspace/models && \
    mkdir -p /workspace/output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/workspace/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')" || exit 1

# Entry point
ENTRYPOINT ["python"]
CMD ["orchestrate_pipeline.py", "--mode", "full"]
