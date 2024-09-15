import gzip
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomMNISTDataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        self.transform = transform

        if self.train:
            self.images_path = os.path.join(data_dir, "train-images-idx3-ubyte.gz")
            self.labels_path = os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
        else:
            self.images_path = os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
            self.labels_path = os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")

        self.images = self._load_images(self.images_path)
        self.labels = self._load_labels(self.labels_path)

    def _load_images(self, path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
            data = data.reshape(-1, 28, 28)
        return data

    def _load_labels(self, path):
        with gzip.open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # numpy.ndarray를 PIL.Image로 변환
        image = Image.fromarray(image, mode="L")

        if self.transform:
            image = self.transform(image)

        return image, label
