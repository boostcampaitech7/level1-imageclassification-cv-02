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
        # head 제외한 파라미터를 freeze
        for param in self.model.parameters():
            param.requires_grad = False
        
        for name, param in self.model.named_parameters():
            if 'blocks.18' in name or 'blocks.19' in name or 'blocks.20' in name or 'blocks.21' in name or 'blocks.22' in name or 'blocks.23' in name or 'head' in name:  # 마지막 2개 블록과 head freeze
                param.requires_grad = True
    
    # def unfreeze_2_layers(self):
    #     for name, param in self.model.named_parameters():
    #         if 'blocks.18' in name or 'blocks.19' in name:
    #             param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)