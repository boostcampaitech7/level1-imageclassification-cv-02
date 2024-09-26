import argparse
import importlib
import logging
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.ensemble import VotingClassifier
from src.data.custom_datamodules.sketch_datamodule import SketchDataModule
from src.plmodules.sketch_module import SketchModelModule
from src.plmodules.sketch_CNN_module import SketchModelModule_CNN

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(module_name, checkpoint_path, model_config):
    module_path, class_name = module_name.rsplit(".", 1)
    ModelClass = getattr(importlib.import_module(module_path), class_name)
    model = ModelClass.load_from_checkpoint(checkpoint_path, config=model_config)
    logger.info(f"Loaded {class_name} from {checkpoint_path}")
    return model

def main(config_path):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    logger.info(f"Configuration loaded: {config}")

    # 데이터 모듈 설정
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(importlib.import_module(data_module_path), data_module_class)
    data_module = DataModuleClass(config.data_config_path, config.augmentation_config_path, config.get("seed", 42))
    data_module.setup()

     # 모델 로드
    model1 = load_model(config.model_EVA.module, config.model_EVA.checkpoint, config.model_EVA)
    model2 = load_model(config.model_CNN.module, config.model_CNN.checkpoint, config.model_CNN)

    models = [
        ("SketchModelModule", model1),
        ("SketchModelModule_CNN", model2)
    ]

# ---------------------
#     # VotingClassifier 설정
#     ensemble_clf = VotingClassifier(estimators=models, voting='soft')

#     # 데이터 로드
#     dataloader = data_module.test_dataloader()
#     all_predictions = []

#     for batch in dataloader:
#         data = batch[0].numpy()  # assuming the first element is the input data
#         predictions = ensemble_clf.predict(data)
#         all_predictions.extend(predictions)
# ----------------------


    # 결과 저장
    output_path = f"{config.trainer.default_root_dir}/{config.model_EVA.model_name}_ensemble.csv"
    test_info = data_module.test_info
    test_info['target'] = all_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble model with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)