# Offline Audio Transcription and Summarization using Large Language Model

A Docker-based pipeline that transcribes audio recordings and generates refined summaries/notes using AI LLM Models. It leverages:
* **NVIDIA GPU** for accelerating **Whisper** (transcription) and **Phi** (summarization) within **Docker** containers
* **FastAPI** + **Uvicorn** for a RESTful backend
* **Streamlit** for a user-friendly frontend UI
* **Ollama** for hosting the LLM model (Phi 3.5 mini-instruct) and performing advanced text summarization. Note the biggest reason for using Ollama is for the fact we are using GGUF models. The quantized Q4_K_M model provides quality and performance.

## Table of Contents
* [Overview](#overview)
* [Architecture](#architecture)
* [Folder Structure](#folder-structure)
* [Installation Requirements](#installation-requirements)
* [Environment Variables](#environment-variables)
* [Usage](#usage)
* [Technical Details](#technical-details)
* [Logging & Monitoring](#logging--monitoring)
* [Additional Notes](#additional-notes)
* [Troubleshooting](#troubleshooting)
* [License](#license)

## Overview

This project aims to provide an **end-to-end** solution for:
1. **Transcribing** long or short audio recordings via **OpenAI Whisper - Medium Model**
2. **Summarizing** those transcripts using a **Phi** (model name: `phi3.5 mini-instruct`) running inside an **Ollama** container

### Key Features
* Simple **Docker Compose** stack with two services:
  1. **app**: Runs both FastAPI and Streamlit in one container
  2. **ollama**: Provides the Summarization Large Language Model
* Automatic GPU offloading if NVIDIA drivers and the **NVIDIA Container Toolkit** are available
* **Streamlit** frontend for easy user interaction: drag-and-drop audio, see transcription & summary

## Architecture

```
+----------------------------+
| Docker Container (ollama)  |
| LLM Summarization         |
| (Phi 3.5 mini-instruct)   |
+--------------^------------+
               |
+----------------------------------+
|     Docker Container (app)        |
|  +----------------------------+   |
|  | FastAPI     |  Streamlit  |   |
|  | (Uvicorn)   |  (web UI)   |   |
|  | :8000       |   :8501     |   |
|  +----------------------------+   |
|          |            |          |
|    (Whisper)    (User Uploads)   |
+----------------------------------+
           |
+----------------------+
| Whisper Transcription |
|   (GPU-accelerated)   |
+----------------------+
```

## Folder Structure

```
LocalAudioTran-LLM-Summar/
├─ .dockerignore
├─ .env
├─ .gitignore
├─ README.md
├─ docker-compose.yml
├─ Dockerfile
├─ backend/
│  ├─ requirements.txt
│  └─ app/
│     ├─ main.py
│     ├─ services/
│     │  ├─ transcription.py
│     │  ├─ summarization.py
│     │  └─ __init__.py
│     ├─ utils/
│     │  └─ logger.py
│     ├─ models/
│     │  ├─ schemas.py
│     │  └─ __init__.py
│     └─ __init__.py
├─ frontend/
│  ├─ requirements.txt
│  └─ src/
│     └─ app.py
└─ logs/
```

### Key Directories

* **`backend/`**: Houses the FastAPI application
  * `main.py` - Primary endpoints
  * `transcription.py` - Whisper-based audio transcription
  * `summarization.py` - Ollama integration and multi-step summary approach
  * `logger.py` - Rotating logs setup

* **`frontend/`**: Contains the Streamlit interface
* **`docker-compose.yml`**: Defines `app` and `ollama` services
* **`Dockerfile`**: System setup and dependencies

## Installation Requirements

### 1. Docker & Docker Compose
* Install [Docker](https://docs.docker.com/get-docker/)
* Install Docker Compose plugin
* Verify installation:
  ```bash
  docker --version
  docker-compose --version
  ```

### 2. NVIDIA GPU Setup (Optional but Recommended)
* Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
* Verify GPU visibility:
  ```bash
  nvidia-smi
  ```

### 3. System Requirements
* Disk Space:
  * Docker images: >1GB
  * Full environment + models: Several GB
* RAM: Minimum 32-64GB recommended
* GPU Memory: 12-16GB recommended (if using GPU)
* Internet connection: Required for downloading models

### 4. Environment Setup (Mandatory Step)
Create a `.env` file at the repository root:

```env
HF_TOKEN=hf_123yourhuggingfacetoken
PYTHONPATH=/app
NVIDIA_VISIBLE_DEVICES=all
```

**Important**: Never commit sensitive tokens to public repositories.

## Usage

### 1. Building & Running

```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up
```

This creates two containers:
* `app`: FastAPI (`:8000`) + Streamlit (`:8501`)
* `ollama`: LLM server (`:11434`)

### 2. Accessing Services
* Frontend (Streamlit): `http://localhost:8501`
* Backend (FastAPI): `http://localhost:8000`
* Ollama: Port `11434` (internal use only)

### 3. Processing Audio
1. Open Streamlit interface
2. Upload audio file (supported: mp3, wav, m4a)
3. Click "Process Audio"
4. View results in "Transcription" and "Summary" tabs
5. Optional use the Clipboard to copy the summary as a text file
## Technical Details

### Transcription Flow
1. FastAPI receives `UploadFile`
2. File saved to temporary storage
3. Whisper processes audio (GPU-accelerated if available)
4. Results returned to client

### Summarization Flow
1. **Direct Processing**: Transcript processed in a single pass using Phi model. The biggest reason to choose a large context window is to ensure the model can process the entire transcript without truncation, chunking, overlapping sections etc as the quality gets detoirated with chunking
2. **Structured Output**: Summary organized into clear sections:
   - Overview
   - Main Points
   - Key Insights
   - Action Items / Decisions
   - Open Questions / Next Steps
   - Conclusions

## Logging & Monitoring

### Log Locations
* Backend:
  * `logs/api.log`
  * `logs/transcription.log`
  * `logs/summarization.log`
* Frontend: `logs/frontend.log`

View combined logs:
```bash
docker-compose logs -f
```

## Troubleshooting

### GPU Issues
```bash
# Verify Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-runtime-ubuntu22.04 nvidia-smi

# Check environment
nvidia-smi
```

### Common Problems & Solutions

#### Slow Transcription
* Check for CPU fallback
* Try smaller Whisper model in `transcription.py`

#### Memory Issues
* Reduce model size
* Lower context window
* Ensure sufficient GPU memory (16-24GB)

#### Port Conflicts
Default ports:
* FastAPI: `:8000`
* Streamlit: `:8501`

Solution: Edit port mappings in `docker-compose.yml`

## License

MIT License

Copyright (c) [2025] [AskAresh.com]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.