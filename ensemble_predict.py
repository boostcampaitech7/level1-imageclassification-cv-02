import argparse
import importlib
import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def main(config_path, use_wandb=False):
    # YAML 파일 로드
    config = OmegaConf.load(config_path)
    print(config)

    # 데이터 모듈 동적 임포트
    data_module_path, data_module_class = config.data_module.rsplit(".", 1)
    DataModuleClass = getattr(
        importlib.import_module(data_module_path), data_module_class
    )

    # 데이터 모듈 설정
    data_config_path = config.data_config_path
    augmentation_config_path = config.augmentation_config_path
    seed = config.get("seed", 42)
    data_module = DataModuleClass(data_config_path, augmentation_config_path, seed)
    data_module.setup()

    # 모델 모듈 동적 임포트 및 설정
    model_modules = []
    for model_info in config.models:
        model_module_path, model_module_class = model_info.module.rsplit(".", 1)
        ModelModuleClass = getattr(
            importlib.import_module(model_module_path), model_module_class
        )
        model = ModelModuleClass.load_from_checkpoint(
            model_info.checkpoint, config=config
        )
        model_modules.append(model)

    # 트레이너 설정
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator, devices=config.trainer.devices
    )

    # 예측 시작
    all_predictions = []
    all_targets = None
    for model in model_modules:
        trainer.test(model, datamodule=data_module)
        predictions = model.test_results["predictions"]
        all_predictions.append(predictions)
        if all_targets is None:
            all_targets = model.test_results["targets"]

    # 앙상블 예측
    ensemble_predictions = torch.mean(torch.stack(all_predictions), dim=0)

    # 성능 지표 계산
    y_true = data_module.test_dataloader().dataset.labels
    y_pred = ensemble_predictions.argmax(dim=1).cpu().numpy()

    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    print(f"Ensemble F1 Score: {f1}")
    print(f"Ensemble Precision: {precision}")
    print(f"Ensemble Recall: {recall}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict with an ensemble of trained models using PyTorch Lightning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument("--use_wandb", action="store_true", help="Use Wandb logger")
    args = parser.parse_args()
    main(args.config, args.use_wandb)
