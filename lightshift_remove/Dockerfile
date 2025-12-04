FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS base

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Upgrade pip
RUN pip3 install --upgrade pip

# Install Python dependencies (your project)
RUN pip3 install --no-cache-dir -r requirements.txt

# Install CUDA-enabled PyTorch (matching CUDA 12.1)
RUN pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Extra packages your code needs
RUN pip3 install python-multipart transformers accelerate

# Copy application code
COPY . /app

EXPOSE 8000

# Start API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
