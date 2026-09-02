"""
src/main.py — FastAPI Application Setup and Routing

This module handles HTTP protocol, request validation, application
lifecycle events, and error handling. The ML logic is delegated to
src/model.py for clean separation of concerns.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from src.model import get_model, predict

# ============================================
# Logging Configuration
# ============================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================
# Application Lifecycle — Lifespan Context Manager
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Warm up the server by loading the Keras model into memory
    at startup. This prevents the first request from suffering
    a massive delay due to model loading.
    """
    logger.info("Application starting up...")
    try:
        get_model()
        logger.info("Model warm-up complete. Server is ready.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    yield
    logger.info("Application shutting down...")


# ============================================
# FastAPI Application
# ============================================
app = FastAPI(
    title="ML Prediction API",
    description=(
        "A production-ready RESTful API for image classification "
        "using a pre-trained Keras CIFAR-10 model. Containerized with "
        "Docker and automated via GitHub Actions CI/CD."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Allowed MIME types for image uploads
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


# ============================================
# Health Check Endpoint
# ============================================
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify application status and readiness.

    Returns:
        JSON: {"status": "ok"} with HTTP 200
    """
    return {"status": "ok"}


# ============================================
# Prediction Endpoint
# ============================================
@app.post("/predict", tags=["Prediction"])
async def predict_endpoint(file: UploadFile = File(...)):
    """
    Accept an image file upload and return classification probabilities.

    The endpoint validates the file content type, preprocesses the image,
    runs inference through the Keras model, and returns structured results.

    Args:
        file: Multipart file upload (image/jpeg or image/png)

    Returns:
        JSON: {
            "class_label": str,
            "probabilities": list[float],
            "filename": str,
            "timestamp": str
        }

    Raises:
        HTTPException 400: Invalid content type (not an image)
        HTTPException 422: Image preprocessing failure
        HTTPException 500: Unexpected internal error
    """

    # ---- Input Validation ----
    # Check content type to ensure the user is uploading an image
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        logger.warning(
            f"Rejected upload: invalid content type '{file.content_type}' "
            f"for file '{file.filename}'"
        )
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Invalid file type",
                "message": (
                    f"Content type '{file.content_type}' is not supported. "
                    f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}"
                ),
            },
        )

    try:
        # Read the file bytes
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail={"error": "Empty file", "message": "The uploaded file is empty."},
            )

        # ---- Run Prediction ----
        result = predict(image_bytes)

        # ---- Build Response ----
        response = {
            "class_label": result["class_label"],
            "probabilities": result["probabilities"],
            "filename": file.filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # ---- Save prediction to predictions/ directory ----
        try:
            os.makedirs("predictions", exist_ok=True)
            safe_filename = file.filename.rsplit(".", 1)[0] if file.filename else "unknown"
            prediction_path = f"predictions/{safe_filename}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            with open(prediction_path, "w") as f:
                json.dump(response, f, indent=2)
            logger.info(f"Prediction saved to {prediction_path}")
        except Exception as save_err:
            # Don't fail the request if saving fails
            logger.warning(f"Failed to save prediction: {save_err}")

        logger.info(
            f"Prediction complete: {result['class_label']} "
            f"(file: {file.filename})"
        )
        return JSONResponse(content=response, status_code=200)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise

    except ValueError as e:
        # Preprocessing failures
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unprocessable Entity",
                "message": f"Failed to process image: {str(e)}",
            },
        )

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Internal server error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred during prediction.",
            },
        )
