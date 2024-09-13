import torch
import torch.nn.functional as F
from torch import nn
import timm


class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # head 제외한 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False

        # head 부분만 학습하도록 설정
        for param in self.model.head.parameters():
            param.requires_grad = True

        return self.model(x)