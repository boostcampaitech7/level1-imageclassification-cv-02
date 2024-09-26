import argparse
import importlib
import logging
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

from omegaconf import OmegaConf
# from sklearn.ensemble import VotingClassifier
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

# ---------------------
    # models = [
    #     ("SketchModelModule", model1),
    #     ("SketchModelModule_CNN", model2)
    # ]

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

    model1.eval()
    model2.eval()

    # dataloader = data_module.test_dataloader()
    # all_predictions = []

    # for batch in dataloader:
    #     x = batch[0]

    #     # 만약 데이터가 3D라면 (batch_size, height, width) 형태로 되어 있는지 확인
    #     if x.ndim == 3:
    #         x = x.unsqueeze(1)

    #     with torch.no_grad():
    #         logits1 = model1(x)
    #         logits2 = model2(x)

    #     probs1 = torch.softmax(logits1, dim=1)
    #     probs2 = torch.softmax(logits2, dim=1)

    #     avg_probs = (probs1 + probs2) / 2.0

    #     preds = torch.argmax(avg_probs, dim=1)
    #     all_predictions.extend(preds.cpu().numpy())

# --------------------------------
    models = [
        model1,
        model2
    ]

    # Trainer 생성
    default_root_dir = config.trainer.get("default_root_dir", "output")  # 기본 경로 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, 
        devices=config.trainer.devices,
        precision=16,
        default_root_dir=default_root_dir  # default_root_dir 설정
    )

    # 예측 시작
    soft_voting_predictions = []
    for model in models:
        trainer.test(model, datamodule=data_module)
        soft_voting_predictions.append(np.array(model.test_predictions))

    # Soft Voting 앙상블: 예측 확률 분포로 가정하고 평균 계산
    soft_voting_predictions = np.array(soft_voting_predictions)

    # 예측값이 클래스 인덱스인지 확인하고 변환
    if soft_voting_predictions.ndim == 2:  # 예측값이 1차원 배열 (클래스 인덱스)인 경우
        # 예측값을 확률 분포로 변환
        num_classes = config.model_EVA.model.num_classes
        one_hot_predictions = np.zeros((len(soft_voting_predictions[0]), num_classes))
        for preds in soft_voting_predictions:
            one_hot_preds = np.zeros((len(preds), num_classes))
            one_hot_preds[np.arange(len(preds)), preds.astype(int)] = 1
            one_hot_predictions += one_hot_preds
        soft_voting_predictions = one_hot_predictions / len(models)
    elif soft_voting_predictions.ndim == 3:  # 올바른 확률 분포 형태 (모델 개수, 데이터 개수, 클래스 개수)
        weighted_predictions = (soft_voting_predictions[0] * 0.7) + (soft_voting_predictions[1] * 0.3)
        soft_voting_predictions = weighted_predictions
    else:
        raise ValueError(f"Unexpected dimension for soft_voting_predictions: {soft_voting_predictions.ndim}")

    # 앙상블 확률을 기반으로 최종 예측 결정
    ensemble_predictions = np.argmax(soft_voting_predictions, axis=1)

    # 결과 저장
    output_path = f"{config.trainer.default_root_dir}/{config.model_EVA.model.model_name}_ensemble.csv"
    test_info = data_module.test_info
    test_info['target'] = ensemble_predictions
    test_info = test_info.reset_index().rename(columns={"index": "ID"})
    test_info.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an ensemble model with PyTorch Lightning")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    main(args.config)