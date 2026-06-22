FROM python:3.10-slim

# =========================
# Environment settings
# =========================
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV INSIGHTFACE_HOME=/root/.insightface

# =========================
# Working directory
# =========================
WORKDIR /service

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# =========================
# Install Python dependencies
# =========================
COPY app/requirements.txt ./requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =========================
# Copy application code
# =========================
COPY app ./app
COPY preload_model.py ./preload_model.py

# =========================
# Preload InsightFace models
# =========================
RUN python3 preload_model.py

# =========================
# Expose API port
# =========================
EXPOSE 8001

# =========================
# Start FastAPI app on port 8001
# =========================
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8001", "--timeout", "180"]