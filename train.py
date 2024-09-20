import argparse
import importlib

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def main(config_path, use_wandb=True):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    # .을 기준으로 오른쪽에서 split하여 모듈 경로와 이름을 분리한다.
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()  # 데이터 모듈에는 setup이라는 메소드가 존재한다.

    # 모델 모듈 동적 임포트
    model_module_path, model_module_class = config.model_module.rsplit(".", 1)
    ModelModuleClass = getattr(
        importlib.import_module(model_module_path), model_module_class
    )

    # 모델 설정
    model = ModelModuleClass(config)

    # Wandb 로거 설정 (use_wandb 옵션에 따라)
    logger = None
    if use_wandb:
        logger = WandbLogger(project="Sketch", name="Sketch_Test")

    # 콜백 설정
    checkpoint_callback = ModelCheckpoint(
        monitor=config.callbacks.model_checkpoint.monitor,
        save_top_k=config.callbacks.model_checkpoint.save_top_k,
        mode=config.callbacks.model_checkpoint.mode,
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.callbacks.early_stopping.monitor,
        patience=config.callbacks.early_stopping.patience,
        mode=config.callbacks.early_stopping.mode,
    )

    # 트레이너 설정
    trainer = pl.Trainer(
        **config.trainer,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
        precision=16,
    )

    # 훈련 시작
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Train a model with PyTorch Lightning")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()
    main(args.config, args.use_wandb)
