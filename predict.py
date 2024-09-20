import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data.custom_datamodules.mnist_datamodule import MNISTDataModule
from src.plmodules.mnist_module import MNISTModelModule


def main(config_path, checkpoint_path):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    data_module = MNISTDataModule(data_config_path, augmentation_config_path)
    data_module.setup()

    # 모델 설정
    model = MNISTModelModule.load_from_checkpoint(checkpoint_path, config=config)

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, devices=config.trainer.devices
    )

    # 예측 시작
    predictions = trainer.predict(model, datamodule=data_module)
    print(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict with a trained model using PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint)
