import os
import glob
import yaml
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl

from src.data.custom_datamodules.sketch_datamodule import SketchDataModule
from src.plmodules.sketch_module import SketchModelModule


def main(config_path, checkpoint_path=None):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # model_name에서 '.' 이전 부분 추출하여 name 필드 설정
    model_name = config.model.model_name
    name_prefix = model_name.split('.')[0]
    
    if not config.get('name'):  # name 필드가 비어있다면 설정
        config.name = name_prefix


    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)  # 시드 값을 설정 파일에서 읽어오거나 기본값 42 사용
    data_module = SketchDataModule(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 체크포인트 경로 설정
    if checkpoint_path is None:
        checkpoint_path = config.checkpoint_path

    # 모델 설정
    model = SketchModelModule.load_from_checkpoint(checkpoint_path, config=config)

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, 
        devices=config.trainer.devices,
        precision=16,
        default_root_dir=config.trainer.default_root_dir # output 폴더로 저장하게끔 
    )

    # 평가 시작
    trainer.test(model, datamodule=data_module)

    # csv 파일에 output 저장하기
    output_path = f"{config.trainer.default_root_dir}/{config.name}.csv"  # output 폴더에 저장
    test_info = data_module.test_info
    predictions = model.test_predictions
    test_info['target'] = predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})

    # test_info.to_csv("./" + config.name + '.csv', index=False)
    test_info.to_csv(output_path, index=False)




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
