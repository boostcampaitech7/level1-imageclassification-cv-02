import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from src.models.base_model import BaseModel


class MNISTModelModule(pl.LightningModule):
    def __init__(self, config):
        super(MNISTModelModule, self).__init__()
        self.config = config
        self.model = BaseModel(config.model)
        self.precision = MulticlassPrecision(num_classes=10, average="macro")
        self.recall = MulticlassRecall(num_classes=10, average="macro")
        self.f1_score = MulticlassF1Score(num_classes=10, average="macro")
        self.wandb_logger = WandbLogger(project="MNIST", name="MNIST_TEST")
        self.test_results = {}
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
        self.log("val_loss", loss)
        self.log("val_precision", self.precision(y_hat, y))
        self.log("val_recall", self.recall(y_hat, y))
        self.log("val_f1_score", self.f1_score(y_hat, y))
        return loss

    def on_test_epoch_start(self):
        self.test_step_outputs = []  # 테스트 에포크 시작 시 초기화

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)
        self.log("test_precision", self.precision(y_hat, y))
        self.log("test_recall", self.recall(y_hat, y))
        self.log("test_f1_score", self.f1_score(y_hat, y))
        output = {"loss": loss, "preds": y_hat, "targets": y}
        self.test_step_outputs.append(output)  # 결과를 리스트에 추가
        return output

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs
        preds = torch.cat([output["preds"] for output in outputs])
        targets = torch.cat([output["targets"] for output in outputs])

        self.test_results["predictions"] = preds
        self.test_results["targets"] = targets

        accuracy = (preds.argmax(dim=1) == targets).float().mean()
        self.log("test_accuracy", accuracy)

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
