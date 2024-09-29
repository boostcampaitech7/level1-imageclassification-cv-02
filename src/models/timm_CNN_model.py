import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimmModel(nn.Module):
    """
    Timm 라이브러리를 사용하여 다양한 사전 훈련된 모델을 제공하는 클래스.
    특정 모델 구조에 따라 파라미터를 freeze하거나 unfreeze할 수 있습니다.
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        drop_head_prob: float,
        drop_path_prob: float,
        attn_drop_prob: float,
    ):
        super(TimmModel, self).__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=drop_head_prob
            # drop_path_rate=drop_path_prob,
            # attn_drop_rate=attn_drop_prob
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for name, param in self.model.named_parameters():
            # resnet 모델의 경우, stage 3, 4 및 head 계층만 unfreeze
            if 'stages.3' in name or 'stages.4' in name or 'head' in name or 'norm' in name:
                param.requires_grad = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
