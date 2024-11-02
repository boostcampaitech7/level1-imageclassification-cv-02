import pytest
import torch

from src.data.collate_fns.sketch_collate_fn import sketch_collate_fn
from src.data.datamodules.sketch_datamodule import SketchDataModule


def test_sketch_collate_fn():
    # dummy data for test
    batch = [
        (torch.tensor([1, 2]), torch.tensor([3, 4])),
        (torch.tensor([5, 6]), torch.tensor([7, 8])),
    ]
    collated_batch = sketch_collate_fn(batch)

    assert isinstance(collated_batch, list)  # collate_fn의 결과가 리스트인지 확인
    assert all(
        isinstance(item, torch.Tensor) for item in collated_batch
    )  # 리스트 내부의 요소들이 텐서인지 확인


def test_sketch_datamodule():
    data_config_path = "configs/data_configs/sketch_config.yaml"
    augmentation_config_path = "configs/augmentation_configs/sketch_augmentation.yaml"
    sketch_dm = SketchDataModule(data_config_path, augmentation_config_path)
    sketch_dm.setup()
    train_loader = sketch_dm.train_dataloader()
    for batch in train_loader:
        print(f"Batch: {batch}")
        assert isinstance(batch, list)  # 데이터 로더의 배치가 리스트인지 확인
        assert len(batch) == 2  # 배치가 (이미지, 라벨) 형태인지 확인
        images, labels = batch
        assert isinstance(images, torch.Tensor)  # 이미지가 텐서인지 확인
        assert isinstance(labels, torch.Tensor)  # 라벨이 텐서인지 확인
        break