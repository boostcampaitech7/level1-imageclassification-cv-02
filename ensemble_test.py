import argparse
import importlib
import pytorch_lightning as pl
from omegaconf import OmegaConf
import pandas as pd

from src.data.custom_datamodules.sketch_datamodule import SketchDataModule
from src.plmodules.sketch_module import SketchModelModule


def load_model(module_name, checkpoint_path):
    module_path, class_name = module_name.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)
    model = ModelClass.load_from_checkpoint(checkpoint_path)
    return model

def ensemble_predict(models, data):
    predictions = [model(data) for model in models]
    avg_prediction = sum(predictions) / len(predictions)
    return avg_prediction

def main(config_path):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(importlib.import_module(data_module_path), data_module_class)

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)

    # 데이터 모듈 인스턴스 생성
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 모델 로드
    models = [load_model(model.module, model.checkpoint) for model in config.models]

    # 데이터 로드
    test_data = data_module.test_dataloader()

    # 앙상블 모델로 예측
    all_predictions = []
    for batch in test_data:
        predictions = ensemble_predict(models, batch)
        all_predictions.extend(predictions)

    # csv 파일에 output 저장하기
    output_path = f"{config.trainer.default_root_dir}/{config.model.name}.csv"  # output 폴더에 저장
    test_info = data_module.test_info
    test_info['target'] = all_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble model with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)