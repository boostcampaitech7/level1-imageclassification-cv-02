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
        optimizer_class = getattr(torch.optim, self.config.optimizer.name)
        optimizer = optimizer_class(self.parameters(), **self.config.optimizer.params)
        if hasattr(self.config, "scheduler"):
            scheduler_class = getattr(
                torch.optim.lr_scheduler, self.config.scheduler.name
            )
            scheduler = scheduler_class(optimizer, **self.config.scheduler.params)
            return [optimizer], [scheduler]
        else:
            return optimizer
