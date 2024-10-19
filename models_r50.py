# import timm.models.swin_transformer_v2
# from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.models._builder import build_model_with_cfg
# from timm.models.swin_transformer_v2 import checkpoint_filter_fn

from timm.models.resnet import Bottleneck

from functools import partial

import timm
import torch
import torch.nn as nn


class ResNet50WithMetadata(timm.models.resnet.ResNet):
    def __init__(self, head_dropout=0.0, num_risk_factors=5, **kwargs):
        super(ResNet50WithMetadata, self).__init__(**kwargs)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(self.num_features, 32, bias=True)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=head_dropout)
        # Adjust for the number of risk factors
        self.linear2 = nn.Linear(32 + num_risk_factors, 2, bias=True)

    def forward(self, x, risk_factors):
        # Forward pass through the base ResNet layers
        x = self.forward_features(x)
        # print("Shape after forward_features:", x.shape)  # Debugging line

        # x = self.global_pool(x)
        x = x.mean(dim=[2, 3])  # ??????????????????????
        print("Shape after pool:", x.shape)

        # Flatten to (B, C)
        # x = x.view(x.size(0), -1)  # dont need when using x.mean
        # print("Shape after view change x.view(x.size(0), -1):", x.shape)

        # Additional layers for metadata
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Concatenate image embeddings with risk factors
        x = torch.cat((x, risk_factors), dim=1)

        # Final classification
        x = self.linear2(x)
        return x


def resnet50_META(pretrained: bool = False, **kwargs) -> ResNet50WithMetadata:
    """Constructs a ResNet-50 model.
    """
    model_args = dict(block=Bottleneck, layers=(3, 4, 6, 3))
    return _create_resnet_META('resnet50', pretrained, **dict(model_args, **kwargs))


def _create_resnet_META(variant, pretrained: bool = False, **kwargs) -> ResNet50WithMetadata:
    return build_model_with_cfg(ResNet50WithMetadata, variant, pretrained, **kwargs)
