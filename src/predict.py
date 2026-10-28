"""Predict the digit in your own image (photo or scan of a handwritten digit).

Usage:
    python src/predict.py path/to/digit.png

The image is converted to grayscale, inverted if it is dark-on-light (like ink on paper),
cropped to the digit, scaled to 20x20 and centred in a 28x28 frame, matching MNIST preprocessing.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "mnist_cnn.keras"


def preprocess(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("L"), dtype="float32") / 255.0
    if img.mean() > 0.5:  # light background: invert so the digit is white on black like MNIST
        img = 1.0 - img
    mask = img > 0.2
    if mask.any():
        rows, cols = np.where(mask)
        img = img[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    h, w = img.shape
    scale = 20.0 / max(h, w)
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = np.asarray(Image.fromarray((img * 255).astype("uint8")).resize(new, Image.LANCZOS),
                       dtype="float32") / 255.0
    canvas = np.zeros((28, 28), dtype="float32")
    top, left = (28 - digit.shape[0]) // 2, (28 - digit.shape[1]) // 2
    canvas[top:top + digit.shape[0], left:left + digit.shape[1]] = digit
    return canvas[None, ..., None]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()

    model = keras.models.load_model(MODEL_PATH)
    probs = model.predict(preprocess(args.image), verbose=0)[0]
    top3 = probs.argsort()[::-1][:3]
    print(f"Prediction: {top3[0]}  (confidence {probs[top3[0]]:.1%})")
    print("Top 3: " + ", ".join(f"{d}: {probs[d]:.1%}" for d in top3))


if __name__ == "__main__":
    main()
