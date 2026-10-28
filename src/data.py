"""Download and load the MNIST dataset from the original IDX files."""

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIRROR = "https://raw.githubusercontent.com/fgnt/mnist/master/"
FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if not path.exists():
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(MIRROR + name, path)
    return path


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Bad image file: {path}"
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Bad label file: {path}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist():
    """Return (x_train, y_train), (x_test, y_test) as float32 in [0, 1] with a channel axis."""
    p = {k: _download(v) for k, v in FILES.items()}
    x_train = _read_images(p["x_train"]).astype("float32")[..., None] / 255.0
    x_test = _read_images(p["x_test"]).astype("float32")[..., None] / 255.0
    y_train = _read_labels(p["y_train"]).astype("int64")
    y_test = _read_labels(p["y_test"]).astype("int64")
    return (x_train, y_train), (x_test, y_test)
