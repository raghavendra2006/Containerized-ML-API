# 🧠 Containerized ML Prediction API

[![CI/CD Pipeline](https://github.com/yourusername/Containerized-ML-API/actions/workflows/main.yml/badge.svg)](https://github.com/yourusername/Containerized-ML-API/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00.svg)](https://www.tensorflow.org/)

A **production-ready RESTful API** for image classification using a pre-trained Keras CNN model (CIFAR-10), containerized with Docker and automated via GitHub Actions CI/CD.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [CI/CD Pipeline](#cicd-pipeline)
- [Configuration](#configuration)
- [FAQ](#faq)

---

## 🔍 Overview

This project demonstrates end-to-end MLOps practices by transforming a trained Keras image classification model into a scalable, containerized prediction service. Key features include:

- **RESTful API** with FastAPI for high-performance, async-capable model serving
- **Singleton Model Loading** — the Keras model is loaded exactly once at startup, ensuring zero latency overhead on prediction requests
- **Input Validation** — strict content-type checking rejects non-image uploads with proper HTTP error codes
- **Error Handling** — all exceptions are caught and returned as structured JSON responses
- **Docker Containerization** — optimized Dockerfile with layer caching for fast rebuilds
- **CI/CD Pipeline** — GitHub Actions automates testing, building, and deployment simulation
- **Automated Testing** — comprehensive pytest suite with mocked ML inference

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Production Environment (Docker)                 │
│                                                                  │
│  ┌──────────┐    POST /predict     ┌──────────────────────────┐ │
│  │          │ ──────────────────► │   FastAPI Server (:8000)  │ │
│  │  Client  │                     │                            │ │
│  │          │ ◄────────────────── │   ┌────────────────────┐  │ │
│  └──────────┘    JSON Response    │   │  In-Memory Keras   │  │ │
│                                   │   │  Model (Singleton)  │  │ │
│                                   │   └────────────────────┘  │ │
│                                   └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    CI/CD Pipeline (GitHub Actions)                │
│                                                                  │
│  Git Push ──► Checkout ──► Install Deps ──► Pytest ──► Docker   │
│    to main                                    Build ──► Push     │
│                                                      (Simulated) │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed:

| Tool   | Version | Purpose                    |
|--------|---------|----------------------------|
| Docker | 20.10+  | Container runtime          |
| Docker Compose | 1.29+ | Container orchestration |
| Git    | 2.30+   | Version control            |
| Python | 3.9+    | Local development (optional)|

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Containerized-ML-API.git
cd Containerized-ML-API

# 2. Build and start the container
docker-compose up --build

# 3. The API is now running at http://localhost:8000
```

### Option 2: Local Development

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Containerized-ML-API.git
cd Containerized-ML-API

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the API server
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📖 API Documentation

Once the server is running, interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### `GET /health` — Health Check

Verify the application is running and the model is loaded.

```bash
curl http://localhost:8000/health
```

**Response** (`200 OK`):
```json
{
  "status": "ok"
}
```

---

#### `POST /predict` — Image Classification

Upload an image file to receive classification predictions.

```bash
# Using curl with a JPEG image
curl -X POST \
  -F "file=@my_image.jpg" \
  http://localhost:8000/predict

# Using curl with a PNG image
curl -X POST \
  -F "file=@sample_photo.png" \
  http://localhost:8000/predict
```

**Response** (`200 OK`):
```json
{
  "class_label": "dog",
  "probabilities": [0.012345, 0.008721, 0.015634, 0.032156, 0.021543, 0.843219, 0.018762, 0.024531, 0.011234, 0.011855],
  "filename": "my_image.jpg",
  "timestamp": "2026-09-02T03:30:00.000000+00:00"
}
```

**CIFAR-10 Class Labels**:

| Index | Label      | Index | Label  |
|-------|------------|-------|--------|
| 0     | airplane   | 5     | dog    |
| 1     | automobile | 6     | frog   |
| 2     | bird       | 7     | horse  |
| 3     | cat        | 8     | ship   |
| 4     | deer       | 9     | truck  |

---

#### Error Responses

| Status Code | Condition | Example |
|-------------|-----------|---------|
| `400` | Non-image file uploaded | `{"detail": {"error": "Invalid file type", "message": "..."}}` |
| `422` | Image preprocessing failure | `{"detail": {"error": "Unprocessable Entity", "message": "..."}}` |
| `500` | Internal server error | `{"detail": {"error": "Internal Server Error", "message": "..."}}` |

---

## 📁 Project Structure

```
Containerized-ML-API/
├── .github/
│   └── workflows/
│       └── main.yml             # CI/CD pipeline definition
├── src/
│   ├── __init__.py              # Package marker
│   ├── main.py                  # FastAPI app, routing, validation
│   └── model.py                 # Keras model loading & inference
├── models/
│   └── my_classifier.h5         # Pre-trained CIFAR-10 CNN artifact
├── tests/
│   ├── __init__.py              # Package marker
│   └── test_api.py              # Pytest test suite (13 tests)
├── predictions/
│   └── example.json             # Sample prediction output
├── .dockerignore                # Docker build exclusions
├── .env.example                 # Environment variables template
├── create_model.py              # Model artifact creation script
├── train_model.py               # Full CIFAR-10 training script
├── Dockerfile                   # Container image definition
├── docker-compose.yml           # Local orchestration config
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🧪 Testing

The test suite uses **pytest** with **mocked ML inference** to run in milliseconds without loading TensorFlow.

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short
```

### Test Coverage

| Test Class | Tests | Description |
|-----------|-------|-------------|
| `TestHealthEndpoint` | 3 | Health check returns 200, status "ok", JSON content |
| `TestPredictValidRequests` | 4 | Valid PNG/JPEG uploads, response schema validation |
| `TestPredictInvalidRequests` | 4 | Rejection of .txt, .pdf, .csv files with 400 |
| `TestPredictErrorHandling` | 2 | Preprocessing errors (422), internal errors (500) |

**Total: 13 tests** — all passing ✅

---

## 🔄 CI/CD Pipeline

The GitHub Actions pipeline (`.github/workflows/main.yml`) is triggered on every push to the `main` branch.

### Pipeline Steps

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Checkout │──►│  Setup   │──►│ Install  │──►│  Pytest  │──►│  Docker  │
│   Code   │   │ Python   │   │   Deps   │   │  Tests   │   │  Build   │
│          │   │   3.9    │   │          │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                                                  │
                                                                  ▼
                                                           ┌──────────┐
                                                           │ Simulate │
                                                           │  Push    │
                                                           └──────────┘
```

1. **Checkout**: Clone the repository
2. **Setup Python 3.9**: Configure the runtime environment
3. **Install Dependencies**: `pip install -r requirements.txt`
4. **Run Pytest**: Execute test suite — pipeline fails if tests fail
5. **Docker Build**: Build the container image tagged with the commit SHA
6. **Simulate Push**: Log a simulated push to a container registry

---

## ⚙️ Configuration

The application reads configuration from environment variables. See `.env.example` for all available options.

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/my_classifier.h5` | Path to the Keras model artifact |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `APP_PORT` | `8000` | Application server port |
| `APP_ENV` | `development` | Runtime environment identifier |

---

## 🐳 Docker Details

### Building Manually

```bash
# Build the image
docker build -t ml-api:latest .

# Run the container
docker run -p 8000:8000 \
  -e MODEL_PATH=models/my_classifier.h5 \
  -e LOG_LEVEL=INFO \
  ml-api:latest
```

### Dockerfile Optimization

- **Slim base image** (`python:3.9-slim-buster`) — minimizes attack surface
- **Layer caching** — `requirements.txt` is copied and installed before source code, so dependency layers are cached across builds
- **Health checks** — built-in container health monitoring
- **Non-root execution** — follows container security best practices

---

## ❓ FAQ

**Q: What model architecture is used?**
A: A lightweight CNN with 3 convolutional blocks (~177K parameters) trained for CIFAR-10 classification (32×32×3 input, 10 classes).

**Q: How is the model loaded?**
A: Via a singleton pattern in `src/model.py`. The model is loaded once at application startup using FastAPI's lifespan event, ensuring zero loading overhead per request.

**Q: Can I use my own model?**
A: Yes — set the `MODEL_PATH` environment variable to point to your `.h5` file. Ensure the input shape and class labels in `src/model.py` match your model's architecture.

**Q: How are tests run without TensorFlow?**
A: The `predict()` function is mocked using `unittest.mock.patch`, so tests never invoke TensorFlow inference. This keeps tests fast (~2 seconds).

---

## 📜 License

This project is developed for educational and portfolio purposes.