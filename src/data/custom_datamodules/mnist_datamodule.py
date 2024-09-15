import os
from typing import Optional

import torch
from torchvision import transforms

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.mnist_collate_fn import mnist_collate_fn
from src.data.datasets.mnist_dataset import CustomMNISTDataset
from src.utils.data_utils import load_yaml_config


class MNISTDataModule(BaseDataModule):
    def __init__(self, data_config_path: str, augmentation_config_path: str, seed: int):
        self.data_config = load_yaml_config(data_config_path)
        self.augmentation_config = load_yaml_config(augmentation_config_path)
        self.seed = seed  # TODO
        super().__init__(self.data_config)

    def setup(self, stage: Optional[str] = None):
        # 시드 설정
        torch.manual_seed(self.seed)

        if self.augmentation_config["augmentation"]["use_augmentation"]:
            train_transforms = self._get_augmentation_transforms()
        else:
            train_transforms = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

        test_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        # Load datasets
        raw_data_path = self.config["data"]["raw_data_path"]
        self.train_dataset = CustomMNISTDataset(
            data_dir=raw_data_path, train=True, transform=train_transforms
        )

        self.test_dataset = CustomMNISTDataset(
            data_dir=raw_data_path, train=False, transform=test_transforms
        )

        # Split train dataset into train and validation
        train_size = int(
            len(self.train_dataset) * self.config["data"]["train_val_split"]
        )
        val_size = len(self.train_dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.train_dataset, [train_size, val_size]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=True,
            collate_fn=mnist_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=mnist_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=mnist_collate_fn,
            persistent_workers=True,
        )

    def _get_augmentation_transforms(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_class = getattr(transforms, transform_config["name"])
            transform_list.append(transform_class(**transform_config["params"]))
        return transforms.Compose(transform_list)
