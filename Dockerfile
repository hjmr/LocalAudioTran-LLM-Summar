# File: Docker1file
FROM nvidia/cuda:12.9.0-runtime-ubuntu24.04

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install system dependencies
RUN apt update && apt install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
RUN mkdir -p frontend/src backend

# Install backend dependencies
COPY backend/requirements.txt backend/
RUN pip3 install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128
RUN pip3 install --no-cache-dir -r backend/requirements.txt

# Install frontend dependencies
COPY frontend/requirements.txt frontend/
RUN pip3 install --no-cache-dir -r frontend/requirements.txt

# Copy your actual code
COPY backend/ backend/
COPY frontend/ frontend/

# Download NLTK data
RUN python3 -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data'); \
               nltk.download('stopwords', download_dir='/usr/local/share/nltk_data'); \
               nltk.download('averaged_perceptron_tagger', download_dir='/usr/local/share/nltk_data')"

# Expose relevant ports
EXPOSE 8000 8501

# Create a startup script to run both FastAPI and Streamlit
RUN echo '#!/bin/bash\n\
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 & \n\
streamlit run frontend/src/app.py --server.port=8501 --server.address=0.0.0.0\n\
wait' > /start.sh && chmod +x /start.sh

CMD ["/start.sh"]
