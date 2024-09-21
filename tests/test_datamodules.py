# import pytest
# import torch

# from src.data.collate_fns.mnist_collate_fn import mnist_collate_fn
# from src.data.custom_datamodules.mnist_datamodule import MNISTDataModule


# def test_mnist_collate_fn():
#     # 테스트용 데이터 생성
#     batch = [
#         (torch.tensor([1, 2]), torch.tensor([3, 4])),
#         (torch.tensor([5, 6]), torch.tensor([7, 8])),
#     ]
#     collated_batch = mnist_collate_fn(batch)
#     print(f"Collated batch: {collated_batch}")
#     assert isinstance(collated_batch, list)  # collate_fn의 결과가 리스트인지 확인
#     assert all(
#         isinstance(item, torch.Tensor) for item in collated_batch
#     )  # 리스트 내부의 요소들이 텐서인지 확인


# def test_mnist_datamodule():
#     data_config_path = "../configs/data_configs/mnist_config.yaml"
#     augmentation_config_path = "../configs/augmentation_configs/mnist_augmentation.yaml"
#     mnist_dm = MNISTDataModule(data_config_path, augmentation_config_path)
#     mnist_dm.setup()
#     train_loader = mnist_dm.train_dataloader()
#     for batch in train_loader:
#         print(f"Batch: {batch}")
#         assert isinstance(batch, list)  # 데이터 로더의 배치가 리스트인지 확인
#         assert len(batch) == 2  # 배치가 (이미지, 라벨) 형태인지 확인
#         images, labels = batch
#         assert isinstance(images, torch.Tensor)  # 이미지가 텐서인지 확인
#         assert isinstance(labels, torch.Tensor)  # 라벨이 텐서인지 확인
#         break
