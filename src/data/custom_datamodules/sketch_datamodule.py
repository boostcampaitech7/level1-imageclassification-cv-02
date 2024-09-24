import os
from typing import Optional
import pandas as pd

import torch
from torchvision import transforms
import albumentations as A

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
            train_transforms = transforms.Compose(
                [
                    transforms.Resize((448, 448)),
                    transforms.RandAugment(),
                    transforms.ToTensor(), 
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ]
            )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((448, 448)),
                transforms.ToTensor(), 
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
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
            if transform_config["name"] == "ShiftScaleRotate":
                transform_list.extend(
                    transform_class(
                        shift_limit=self.hparams.get("shift_limit",  (-0.0625, 0.0625)), 
                        scale_limit =self.hparams.get("scale_limit", 0 ), 
                        rotate_limit =self.hparams.get("rotate_limit", 0 ), 
                        interpolation =self.hparams.get("interpolation", 1 ), 
                        border_mode =self.hparams.get("border_mode", 3 ), 
                        value =self.hparams.get("value", 0), 
                        mask_value =self.hparams.get("mask_value", 0), 
                        always_apply =self.hparams.get("always_apply", False ), 
                        rotate_method=self.hparams.get("rotate_method", "largest_box"),
                        p =self.hparams.get( "num_ops", 0.5 )
                    )
                )
            elif transform_config["name"] == "Rotate":
                transform_list.extend(
                    transform_class(
                        limit =self.hparams.get("limit"(-90, 90)),
                        interpolation =self.hparams.get("interpolation", 1 ), 
                        border_mode =self.hparams.get("border_mode", 3 ), 
                        value =self.hparams.get("value", None), 
                        mask_value =self.hparams.get("mask_value", None), 
                        rotate_method = self.hparams.get("rotate_method", 'largest_box'), 
                        crop_border= self.hparams.get("rotate_method", False),
                        always_apply=self.hparams.get("always_apply", None),
                        p =self.hparams.get( "num_ops", 0.5 )
                    )
                )
            elif transform_config["name"] == "GaussianBlur ":
                transform_list.extend(
                    transform_class(
                        blur_limit =self.hparams.get("blur_limit",(3, 7)),
                        sigma_limit =self.hparams.get("sigma_limit", 0 ), 
                        always_apply=self.hparams.get("always_apply", None),
                        p =self.hparams.get( "num_ops", 0.5 )
                    )
                )
            elif transform_config["name"] == "GridDropout":
                transform_list.extend(
                    transform_class(
                        ratio =self.hparams.get("ratio", 0.5),
                        unit_size_min =self.hparams.get("unit_size_min", None ), 
                        unit_size_max =self.hparams.get("unit_size_max", None ), 
                        holes_number_x =self.hparams.get("holes_number_x", None), 
                        holes_number_y =self.hparams.get("holes_number_y", None), 
                        shift_x = self.hparams.get("shift_x", None), 
                        shift_y = self.hparams.get("shift_y", None), 
                        random_offset= self.hparams.get("random_offset", False),
                        fill_value=self.hparams.get("fill_value", 0),
                        mask_fill_value=self.hparams.get("mask_fill_value", None),
                        unit_size_range=self.hparams.get("unit_size_range", None),
                        holes_number_xy=self.hparams.get("holes_number_xy", None),
                        shift_xy=self.hparams.get("shift_xy", (0, 0)),
                        always_apply=self.hparams.get("always_apply", None),
                        p =self.hparams.get( "num_ops", 0.5 )
                    )
                )
            elif transform_config["name"] == "RandomGridShuffle":
                transform_list.extend(
                    transform_class(
                        grid =self.hparams.get("grid", (3, 3)),
                        always_apply=self.hparams.get("always_apply", None),
                        p =self.hparams.get( "num_ops", 0.5 )
                    )
                )
            else:
                transform_list.extend(transform_class(**transform_config["params"]))
         
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
