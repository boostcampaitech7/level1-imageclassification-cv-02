import os
from typing import Optional
import pandas as pd

import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from src.data.base_datamodule import BaseDataModule
from src.data.collate_fns.sketch_collate_fn import sketch_collate_fn
from src.data.datasets.sketch_dataset import CustomSketchDataset

from src.utils.data_utils import load_yaml_config


class SketchDataModule(BaseDataModule):
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
            train_transforms = A.Compose(
                [
                    A.Resize(height=448, width=448),
                    #A.ShiftScaleRotate(shift_limit=(-25,25), scale_limit=0, rotate_limit=0, border_mode=3, p=0.5),
                    #A.ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(-30,30), border_mode=3, p=0.5),
                    #A.GaussianBlur (blur_limit=(3,7), sigma_limit=30, p=0.5),
                    #A.GridDropout(ratio=0.5, fill_value=255, p=0.5),
                    A.RandomGridShuffle(grid=(3, 3), p=0.5),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ]
            )

        test_transforms = A.Compose(
            [
                A.Resize(height=448, width=448),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]
        )

        # Load datasets
        raw_data_path = self.config["data"]["raw_data_path"]
        test_data_path = self.config["data"]["test_data_path"]
        train_df_path = self.config["data"]["train_df_path"]
        test_df_path = self.config["data"]["test_df_path"]
        self.train_info = pd.read_csv(train_df_path)
        self.test_info = pd.read_csv(test_df_path)

        self.train_dataset = CustomSketchDataset(
            data_dir=raw_data_path, info_df=self.train_info, is_inference=False, transform=train_transforms
        )

        self.test_dataset = CustomSketchDataset(
            data_dir=test_data_path, info_df=self.test_info, is_inference=True, transform=test_transforms
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
            collate_fn=sketch_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=sketch_collate_fn,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config["data"]["batch_size"],
            num_workers=self.config["data"]["num_workers"],
            shuffle=False,
            collate_fn=sketch_collate_fn,
            persistent_workers=True,
        )

    def _get_augmentation_transforms(self):
        transform_list = [
            transforms.Resize((448, 448))
        ]
        for transform_config in self.augmentation_config["augmentation"]["transforms"]:
            transform_class = getattr(transforms, transform_config["name"])
            # transform_list.append(transform_class)
            if transform_config["name"] == "RandAugment":
                transform_list.append(
                    transform_class(
                        num_ops=self.hparams.get("num_ops", transform_config["params"]["num_ops"]),
                        magnitude=self.hparams.get("magnitude", transform_config["params"]["magnitude"])
                    )
                )
            else:
                transform_list.append(transform_class(**transform_config["params"]))
         
        transform_list.extend(
            [
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )
        return transforms.Compose(transform_list)
