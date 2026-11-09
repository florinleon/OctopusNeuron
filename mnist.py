"""MNIST (or Fashion-MNIST) as a sequence of continual-learning tasks.

Two task streams are available:

class_incremental
    The classes are split into consecutive groups (ClassesPerTask each, in ClassOrder). Task t shows only the
    images of group t, while the agent always chooses among all ten classes. Earlier classes are never shown again
    during training, so remembering them is entirely up to the learner.

permuted
    Every task shows all ten classes, each task under its own fixed random permutation of the pixels (the first task
    keeps the original image). The input statistics change completely at every switch while the labels stay the same.

With ProvideContext the observation of task t comes with the one-hot code of t. Pixel positions are the centres of
the 28 x 28 pixel tiles in the unit square, so local wiring follows the image geometry (after a permutation the pixel
at a position carries content from elsewhere, which is part of what makes the permuted stream hard).

The four IDX files are downloaded once into DataDir and cached as a single compressed .npz file.
"""

import gzip
import os
import shutil
import struct
import urllib.request

import numpy as np

import config
from bandit import DatasetBandit
from problem import Problem, ProblemSpec


DatasetMirrors = {
    "mnist": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ],
    "fashion_mnist": [
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
        "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
    ],
}

DatasetFiles = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

ImageSide = 28
NumClasses = 10


def read_idx(path):
    """Parse a gzip-compressed IDX file (the MNIST format) into a uint8 array."""
    with gzip.open(path, "rb") as handle:
        content = handle.read()
    zero, data_type, dims = struct.unpack(">HBB", content[:4])
    if zero != 0 or data_type != 0x08:
        raise ValueError("Not an unsigned-byte IDX file: " + path)
    shape = struct.unpack(">" + "I" * dims, content[4:4 + 4 * dims])
    return np.frombuffer(content, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def download(name, folder):
    os.makedirs(folder, exist_ok=True)
    for file_name in DatasetFiles.values():
        target = os.path.join(folder, file_name)
        if os.path.exists(target):
            continue
        errors = []
        for mirror in DatasetMirrors[name]:
            try:
                print("Downloading " + mirror + file_name)
                with urllib.request.urlopen(mirror + file_name, timeout=60) as response, open(target + ".part", "wb") as out:
                    shutil.copyfileobj(response, out)
                os.replace(target + ".part", target)
                break
            except Exception as error:
                errors.append(mirror + ": " + str(error))
        else:
            raise RuntimeError("Could not download " + file_name + ":\n" + "\n".join(errors))


def load_dataset(name, data_dir):
    """Return train/test images (n, 784) uint8 and labels (n,) uint8, downloading and caching them on first use."""
    if name not in DatasetMirrors:
        raise ValueError("Unknown dataset '" + str(name) + "'; known: " + ", ".join(sorted(DatasetMirrors)))
    cache = os.path.join(data_dir, name + ".npz")
    if not os.path.exists(cache):
        folder = os.path.join(data_dir, name)
        download(name, folder)
        arrays = {key: read_idx(os.path.join(folder, file_name)) for key, file_name in DatasetFiles.items()}
        arrays["train_images"] = arrays["train_images"].reshape(len(arrays["train_images"]), -1)
        arrays["test_images"] = arrays["test_images"].reshape(len(arrays["test_images"]), -1)
        np.savez_compressed(cache, **arrays)
    with np.load(cache) as data:
        return {key: data[key] for key in DatasetFiles}


class MnistIncremental(Problem):
    """Class-incremental or permuted MNIST stream, each task served as a contextual bandit."""

    def __init__(self, dtype, rng):
        self.dtype = dtype
        self.rng = rng
        self.data = load_dataset(config.Dataset, config.DataDir)
        self.mode = config.ProblemMode
        pixels = ImageSide * ImageSide
        if self.mode == "class_incremental":
            order = list(config.ClassOrder)
            size = int(config.ClassesPerTask)
            self.task_classes = [order[start:start + size] for start in range(0, len(order), size)]
            self.permutations = [None] * len(self.task_classes)
        elif self.mode == "permuted":
            count = int(config.NumPermutations)
            self.task_classes = [list(range(NumClasses))] * count
            self.permutations = [None] + [rng.permutation(pixels) for _ in range(count - 1)]
        else:
            raise ValueError("Unknown MNIST mode '" + str(self.mode) + "'")
        self.num_tasks = len(self.task_classes)
        self.provide_context = bool(config.ProvideContext)
        centres = (np.arange(ImageSide) + 0.5) / ImageSide
        positions = np.stack(np.meshgrid(centres, centres, indexing="ij"), axis=-1).reshape(pixels, 2)
        context_size = self.num_tasks if self.provide_context else 0
        self.spec = ProblemSpec(pixels, positions, context_size, NumClasses)


    def task_name(self, task):
        if self.mode == "permuted":
            return "permutation " + str(task)
        return "classes " + "-".join(str(label) for label in self.task_classes[task])


    def _environment(self, task, batch_size, split, shuffle):
        labels = self.data[split + "_labels"]
        selected = np.flatnonzero(np.isin(labels, self.task_classes[task]))
        context = np.eye(self.num_tasks)[task] if self.provide_context else None
        items = self.data[split + "_images"][selected]
        return DatasetBandit(items, labels[selected].astype(np.int64), batch_size, shuffle, self.rng, self.dtype,
                             scale=1.0 / 255.0, permutation=self.permutations[task], context=context)


    def train_environment(self, task, batch_size):
        return self._environment(task, batch_size, "train", True)


    def test_environment(self, task, batch_size):
        return self._environment(task, batch_size, "test", False)
