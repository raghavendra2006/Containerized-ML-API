# ============================================
# Dockerfile — Containerized ML Prediction API
# ============================================
# Multi-stage optimized build with layer caching
# for TensorFlow/Keras model serving via FastAPI.
# ============================================

# Use slim base image to minimize attack surface and image size
FROM python:3.9-slim-buster

# Set working directory
WORKDIR /app

# Set environment variables
# Prevents Python from writing .pyc files and enables unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=models/my_classifier.h5 \
    LOG_LEVEL=INFO \
    APP_PORT=8000

# ============================================
# Layer Caching Strategy
# ============================================
# Copy ONLY requirements.txt first and install dependencies.
# This ensures that unless dependencies change, Docker will
# cache this expensive layer (~2GB for TensorFlow) and skip
# it on subsequent builds.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Application Code
# ============================================
# Copy application source code and model artifact
COPY src/ ./src/
COPY models/ ./models/

# Create predictions directory for output storage
RUN mkdir -p predictions

# Expose the application port
EXPOSE 8000

# Health check for container orchestrators
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start the FastAPI server with uvicorn
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
