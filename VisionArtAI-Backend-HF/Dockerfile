FROM python:3.11-slim

# ─────────────────────────────────────────────
# System deps (Compiladores incluídos aqui)
# ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─────────────────────────────────────────────
# Python deps
# ─────────────────────────────────────────────
COPY requirements.txt .

# Forçamos a instalação do llama-cpp-python sem tentar usar GPU
# e garantimos que ele seja compilado corretamente
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────
# App files
# ─────────────────────────────────────────────
COPY main.py .
COPY yolo_deart.onnx .
COPY style_classifier.onnx .

# ─────────────────────────────────────────────
# HF / LLM performance config
# ─────────────────────────────────────────────
# No Space Free, o /tmp é o melhor lugar para cache volátil
ENV HF_HOME=/tmp/huggingface
ENV TRANSFORMERS_CACHE=/tmp/huggingface
ENV TOKENIZERS_PARALLELISM=false

# Importante para o llama-cpp não tentar usar AVX512 se o CPU não suportar
ENV CMAKE_ARGS="-DLLAMA_BLAS=OFF -DLLAMA_CUDA=OFF"

# ─────────────────────────────────────────────
# Port
# ─────────────────────────────────────────────
EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]