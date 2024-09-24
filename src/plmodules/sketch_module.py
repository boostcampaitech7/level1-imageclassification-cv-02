import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from src.models.timm_model import TimmModel
import numpy as np


class SketchModelModule(pl.LightningModule):
    def __init__(self, config):
        super(SketchModelModule, self).__init__()
        self.config = config
        print(config.model)
        self.model = TimmModel(
            model_name=config.model.model_name, 
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained
        )
        print(self.model)
        self.precision = MulticlassPrecision(num_classes=config.model.num_classes, average="macro")
        self.recall = MulticlassRecall(num_classes=config.model.num_classes, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=config.model.num_classes, average="macro")
        self.wandb_logger = WandbLogger(project="Sketch", name="SKETCH_TEST")
        self.test_results = {}
        self.test_predictions = []
        self.test_step_outputs = []  # 추가

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_precision", self.precision(y_hat, y))
        self.log("train_recall", self.recall(y_hat, y))
        self.log("train_f1_score", self.f1_score(y_hat, y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        
        # 예측 결과 저장
        logits = F.softmax(y_hat, dim=1)
        preds = logits.argmax(dim=1)
        acc = torch.tensor(torch.sum(preds == y).item() / len(preds))
        # print("[Validation_acc]:", acc)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))
        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []  # 테스트 에포크 시작 시 초기화

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self.forward(x)
        logits = F.softmax(y_hat, dim=1)
        preds = logits.argmax(dim=1)
        output = {"preds": preds.cpu().detach().numpy()} # 우리의 데이터셋에는 target 없음!
        self.test_step_outputs.append(output)  # 결과를 리스트에 추가
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        # preds = torch.cat([output["preds"] for output in outputs])
        preds = np.concatenate([output["preds"] for output in outputs])
        self.test_results["predictions"] = preds

        # 모든 test data에 대한 예측 한 리스트로 합치기!
        self.test_predictions.extend(preds) # csv 파일에 저장하기 위함!
        self.test_step_outputs.clear()  # 메모리 정리

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        lr = self.hparams.get("learning_rate", 2e-4) # sweep으로 lr 조정. / default: 2e-4
        optimizer_class = torch.optim.Adam

        # 옵티마이저 생성
        optimizer = optimizer_class(self.parameters(), lr=lr)

        if self.config.use_sweep is True: # sweep 사용 시
            # 스케줄러 설정
            if self.hparams.get("lr_scheduler") == 'StepLR':
                step_size = self.hparams.get("step_size", 1)  # Sweep에서 step_size 받기
                gamma = self.hparams.get("gamma", 0.7)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, 
                    step_size=step_size,
                    gamma=gamma
                )
            elif self.hparams.get("lr_scheduler") == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.trainer.max_epochs  # CosineAnnealing 스케줄러의 최대 에폭 설정
                )
            elif scheduler_type == 'ReduceLROnPlateau':
                patience = self.hparams.get("patience", 10)
                factor = self.hparams.get("factor", 0.1)
                scheduler = {
                    'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        patience=patience, 
                        factor=factor
                    ),
                    'monitor': 'val_acc',
                    'interval': 'epoch',
                    'frequency': 1
                }
            else:
                scheduler = None
        else: # sweep 사용 X
            if hasattr(self.config, "scheduler"):
                scheduler_class = getattr(
                    torch.optim.lr_scheduler, self.config.scheduler.name
                )
                if self.config.scheduler.name == "ReduceLROnPlateau":
                    scheduler = {
                        'scheduler': scheduler_class(optimizer, **self.config.scheduler.params),
                        'monitor': 'val_acc'
                    }
                else:
                    scheduler = scheduler_class(optimizer, **self.config.scheduler.params)

        if scheduler:
            if isinstance(scheduler, dict):  # ReduceLROnPlateau는 딕셔너리 형태로 반환
                return [optimizer], [scheduler]
            else:
                return [optimizer], [{'scheduler': scheduler}]
        else:
            return optimizer 
