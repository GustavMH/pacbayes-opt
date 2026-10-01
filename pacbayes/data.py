"""Binary MNIST as in Sec. 4.1."""
import gzip
import os
import urllib.request
import numpy as np
import torch

MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"  # the source used by TensorFlow
MNIST_FILES = {"train_images": "train-images-idx3-ubyte.gz", "train_labels": "train-labels-idx1-ubyte.gz",
               "test_images": "t10k-images-idx3-ubyte.gz", "test_labels": "t10k-labels-idx1-ubyte.gz"}
VALIDATION_SIZE = 5000


def read_idx(path, expected_magic):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    if magic != expected_magic:
        raise ValueError("Invalid magic number %d in %s" % (magic, path))
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_binary_mnist(data_dir="mnist", random_labels=False, label_seed=0, device="cpu"):
    """55,000 training and 10,000 test images (Sec. 4.1). 
    Pixels are scaled to [0, 1], digits 0-4 get label +1 and 5-9 label -1.
    With random_labels, the training labels are drawn uniformly from {-1, +1}"""
    os.makedirs(data_dir, exist_ok=True)
    for name in MNIST_FILES.values():
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(MNIST_URL + name, path)
            print("Downloaded", name)

    def read(key, magic):
        return read_idx(os.path.join(data_dir, MNIST_FILES[key]), magic)

    train_x = read("train_images", 2051).reshape(-1, 784)[VALIDATION_SIZE:]
    train_y = read("train_labels", 2049)[VALIDATION_SIZE:]
    test_x = read("test_images", 2051).reshape(-1, 784)
    test_y = read("test_labels", 2049)

    train_y = np.where(train_y <= 4, 1.0, -1.0)
    test_y = np.where(test_y <= 4, 1.0, -1.0)
    if random_labels:
        train_y = np.random.default_rng(label_seed).choice([-1.0, 1.0], size=train_y.shape[0])

    def to_tensor(a):
        return torch.tensor(a, dtype=torch.float32, device=device)

    return (to_tensor(train_x.astype(np.float32) / 255.0), to_tensor(train_y),
            to_tensor(test_x.astype(np.float32) / 255.0), to_tensor(test_y))
