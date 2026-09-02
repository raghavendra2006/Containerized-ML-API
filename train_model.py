"""
train_model.py — Train a CIFAR-10 CNN Classifier

This script trains a lightweight Convolutional Neural Network on the
CIFAR-10 dataset and saves the model as an .h5 artifact for deployment.

This file is NOT deployed in the Docker container — it is a development
utility only.

Usage:
    python train_model.py
"""

import os
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TF warnings

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    """
    Build a lightweight CNN architecture for CIFAR-10 classification.

    Architecture:
        - 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool)
        - Global Average Pooling
        - Dense classifier with dropout

    Returns:
        tf.keras.Model: Compiled model ready for training.
    """
    model = keras.Sequential(
        [
            # Input layer
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
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train():
    """Train the model on CIFAR-10 and save the artifact."""
    print("=" * 50)
    print("CIFAR-10 CNN Classifier Training")
    print("=" * 50)

    # Load CIFAR-10 dataset
    print("\n[1/4] Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    print(f"  Training samples: {len(x_train)}")
    print(f"  Test samples:     {len(x_test)}")
    print(f"  Image shape:      {x_train.shape[1:]}")

    # Build model
    print("\n[2/4] Building CNN architecture...")
    model = build_model()
    model.summary()

    # Train with early stopping
    print("\n[3/4] Training model...")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=25,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/my_classifier.h5"
    model.save(model_path)
    print(f"\n  Model saved to: {model_path}")

    # Verify the saved model
    print("\n  Verifying saved model...")
    loaded_model = keras.models.load_model(model_path)
    verify_loss, verify_acc = loaded_model.evaluate(x_test, y_test, verbose=0)
    print(f"  Verified accuracy: {verify_acc:.4f}")

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    train()
