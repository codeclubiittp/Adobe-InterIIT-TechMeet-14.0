# -------------------------
# 1. Base image with Conda
# -------------------------
FROM continuumio/miniconda3 AS base

# Ensure no interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment file
COPY environment.yaml .

# -------------------------
# 2. Create Conda environment
# -------------------------
RUN conda env create -f environment.yaml && \
    conda clean -afy

# Make environment active by default
SHELL ["bash", "-lc"]
ENV PATH=/opt/conda/envs/lama/bin:$PATH
ENV CONDA_DEFAULT_ENV=lama

# -------------------------
# 3. Copy project files
# -------------------------
COPY . /app

RUN /opt/conda/envs/lama/bin/pip install fastapi python-multipart

# -------------------------
# 4. Expose port & start FastAPI
# -------------------------
EXPOSE 8002

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
