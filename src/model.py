"""
src/model.py — ML Model Loading and Prediction Logic

This module isolates all TensorFlow/Keras logic from the web server.
The model is loaded once into a global singleton to avoid repeated I/O
on every prediction request.
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import logging

logger = logging.getLogger(__name__)

# ============================================
# Global Model Singleton
# ============================================
# This ensures the heavy .h5 file is only loaded into RAM once.
# The model is loaded on application startup, not on each request.
_MODEL = None

# CIFAR-10 native input dimensions
IMAGE_SIZE = (32, 32)

# CIFAR-10 class labels
CLASS_LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_model():
    """
    Load the Keras model into memory using a singleton pattern.

    The model path is read from the MODEL_PATH environment variable,
    falling back to 'models/my_classifier.h5' if not set.

    Returns:
        tf.keras.Model: The loaded Keras model ready for inference.
    """
    global _MODEL
    if _MODEL is None:
        model_path = os.getenv("MODEL_PATH", "models/my_classifier.h5")
        logger.info(f"Loading Keras model from: {model_path}")
        _MODEL = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
    return _MODEL


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes into a tensor suitable for model inference.

    Operations performed:
        1. Decode raw bytes into a PIL Image
        2. Convert to RGB (discards alpha channel if present)
        3. Resize to the model's expected input dimensions (32x32)
        4. Normalize pixel values from [0, 255] to [0.0, 1.0]
        5. Expand dimensions to create a batch of 1: (1, 32, 32, 3)

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        np.ndarray: Preprocessed image tensor with shape (1, 32, 32, 3).

    Raises:
        ValueError: If the image cannot be decoded or processed.
    """
    try:
        # Decode bytes into PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB to ensure 3 channels (removes alpha if present)
        image = image.convert("RGB")

        # Resize to the network's expected input dimensions
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize pixel values to [0, 1]
        image_array = np.array(image, dtype=np.float32) / 255.0

        # Add batch dimension: (32, 32, 3) -> (1, 32, 32, 3)
        image_array = np.expand_dims(image_array, axis=0)

        logger.debug(f"Preprocessed image shape: {image_array.shape}")
        return image_array

    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")


def predict(image_bytes: bytes) -> dict:
    """
    Run end-to-end inference on raw image bytes.

    Steps:
        1. Preprocess the image into a model-compatible tensor
        2. Get the singleton model instance
        3. Run model.predict() to obtain class probabilities
        4. Determine the top predicted class via argmax
        5. Return structured prediction results

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        dict: Prediction results containing:
            - class_label (str): Human-readable name of the predicted class
            - probabilities (list[float]): Probability for each class
    """
    # Step 1: Preprocess the image
    processed_image = preprocess_image(image_bytes)

    # Step 2: Get the model (already loaded at startup)
    model = get_model()

    # Step 3: Run inference
    predictions = model.predict(processed_image, verbose=0)

    # Step 4: Extract probabilities and top class
    probabilities = predictions[0].tolist()
    predicted_index = int(np.argmax(probabilities))
    class_label = CLASS_LABELS[predicted_index]

    logger.info(f"Prediction: {class_label} (confidence: {probabilities[predicted_index]:.4f})")

    # Step 5: Return structured result
    return {
        "class_label": class_label,
        "probabilities": [round(p, 6) for p in probabilities],
    }
