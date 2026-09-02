"""
create_model.py — Create a CIFAR-10 CNN model artifact quickly.

Builds the same CNN architecture and trains on a small synthetic dataset
to produce a valid .h5 file with correct input/output tensor shapes.
This allows rapid testing of the deployment pipeline.
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_and_save_model():
    """Build CNN, train briefly on synthetic data, and save."""
    print("=" * 50)
    print("Creating CIFAR-10 CNN Model Artifact")
    print("=" * 50)

    # Build the CNN architecture (same as train_model.py)
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Create synthetic training data (small batch for quick training)
    print("\nTraining on synthetic data for model validation...")
    np.random.seed(42)
    x_train = np.random.rand(500, 32, 32, 3).astype(np.float32)
    y_train = np.random.randint(0, 10, size=(500, 1))

    model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=1)

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/my_classifier.h5"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Verify
    loaded = keras.models.load_model(model_path)
    test_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
    output = loaded.predict(test_input, verbose=0)
    print(f"Verification: output shape={output.shape}, sum={output[0].sum():.4f}")
    print(f"Predicted class: {np.argmax(output[0])}")
    print("\nDone!")


if __name__ == "__main__":
    create_and_save_model()
