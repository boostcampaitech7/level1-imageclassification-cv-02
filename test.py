import os
import glob
import yaml
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

from src.data.custom_datamodules.sketch_datamodule import SketchDataModule
from src.plmodules.sketch_module import SketchModelModule


def main(config_path, checkpoint_path=None):
    # load test config
    config = OmegaConf.load(config_path)
    print(config)

    # name 필드 비어있을 경우, model name으로 설정
    model_name = config.model.model_name
    name_prefix = model_name.split('.')[0]
    
    if not config.get('name'):
        config.name = name_prefix

    # data module instance
    data_config_path = config.data_config_path
    augmentation_config_path = None
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = SketchDataModule(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # checkpoint path
    if checkpoint_path is None:
        checkpoint_path = config.checkpoint_path

    # model instance
    model = SketchModelModule.load_from_checkpoint(checkpoint_path, config=config)

    # trainer
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, 
        devices=config.trainer.devices,
        precision=16,
        default_root_dir=config.trainer.default_root_dir # output 폴더로 저장하게끔 
    )

    # inference
    trainer.test(model, datamodule=data_module)

    # csv 파일에 output 저장하기
    output_path = f"{config.trainer.default_root_dir}/{config.name}.csv"  # output 폴더에 저장
    test_info = data_module.test_info
    predictions = model.test_predictions
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})

    test_info.to_csv(output_path, index=False) # pd -> csv 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model with PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=False, help="Path to the model checkpoint"
    )
    args = parser.parse_args()

    main(args.config, args.checkpoint)
