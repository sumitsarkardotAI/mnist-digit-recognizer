"""Train a CNN on MNIST, evaluate it on the held-out test set and save plots and metrics.

Usage:
    python src/train.py --epochs 15 --batch-size 128
"""

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from data import load_mnist

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model() -> keras.Model:
    """Two conv blocks with batch norm and dropout, light on-the-fly augmentation."""
    augment = keras.Sequential(
        [
            layers.RandomRotation(0.03),  # about +-11 degrees
            layers.RandomTranslation(0.08, 0.08),
            layers.RandomZoom(0.08),
        ],
        name="augmentation",
    )

    inputs = keras.Input(shape=(28, 28, 1))
    x = augment(inputs)

    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="mnist_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_history(history, path: Path) -> None:
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h["accuracy"], label="train")
    axes[0].plot(h["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[1].plot(h["loss"], label="train")
    axes[1].plot(h["val_loss"], label="validation")
    axes[1].set_title("Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix (test set)")
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_samples(x, y_true, y_pred, conf, idx, title, path: Path) -> None:
    n = len(idx)
    cols = min(n, 8)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.9))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, idx):
        ax.imshow(x[i].squeeze(), cmap="gray")
        ok = y_true[i] == y_pred[i]
        ax.set_title(f"T:{y_true[i]} P:{y_pred[i]}\n{conf[i]:.0%}", fontsize=8,
                     color="green" if ok else "red")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    set_seed()
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_mnist()
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")

    model = build_model()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test, batch_size=512, verbose=0)
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4, output_dict=True)

    print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    model.save(MODEL_DIR / "mnist_cnn.keras")

    plot_history(history, RESULTS_DIR / "training_curves.png")
    plot_confusion(cm, RESULTS_DIR / "confusion_matrix.png")
    rng = np.random.default_rng(SEED)
    plot_samples(x_test, y_test, y_pred, conf, rng.choice(len(x_test), 16, replace=False),
                 "Random test predictions", RESULTS_DIR / "sample_predictions.png")
    wrong = np.where(y_pred != y_test)[0]
    if len(wrong):
        plot_samples(x_test, y_test, y_pred, conf, wrong[:16],
                     f"Misclassified examples ({len(wrong)} of {len(y_test)})",
                     RESULTS_DIR / "misclassified.png")

    metrics = {
        "test_accuracy": round(float(test_acc), 4),
        "test_loss": round(float(test_loss), 4),
        "misclassified": int(len(wrong)),
        "epochs_trained": len(history.history["loss"]),
        "parameters": int(model.count_params()),
        "per_class_f1": {str(k): round(report[str(k)]["f1-score"], 4) for k in range(10)},
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
