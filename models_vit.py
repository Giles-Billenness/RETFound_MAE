# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, head_dropout=0.0, num_risk_factors=5, distil_neurons=32, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.distil_neurons = distil_neurons
        # Global pooling to reduce spatial dimensions
        self.global_pool_dimreduce = nn.AdaptiveAvgPool2d(1)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        if (self.distil_neurons is not None):  # 32,512
            self.linear1 = nn.Linear(embed_dim, self.distil_neurons, bias=True)
            self.activation = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=head_dropout)
            # +5 metadata features
            self.linear2 = nn.Linear(
                self.distil_neurons + num_risk_factors, 2, bias=True)
        else:
            self.linear2 = nn.Linear(
                embed_dim + num_risk_factors, 2, bias=True)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

    def forward_head(self, x, risk_factors):
        if self.distil_neurons is not None:
            # output_mi = self.head_mi(x)
            x = self.linear1(x)  # distil flattened embeddings
            x = self.activation(x)
            x = self.dropout(x)
            x = torch.cat((x, risk_factors), dim=1)
        else:
            # print("x shape:", x.shape)  # [batch, 1024]
            x = torch.cat((x, risk_factors), dim=1)
            # print(f"Shape after concatenation: {x.shape}")

        x = self.linear2(x)  # final layer
        return x

    def forward(self, x, risk_factors):
        x = self.forward_features(x)
        # x = torch.cat((x, risk_factors), dim=1)
        output_mi = self.forward_head(x, risk_factors)
        return output_mi


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# class Prediction_head(nn.Module):
#     """(convolution => [BN] => ReLU) * 2"""

#     def __init__(self, embed_dim):
#         super().__init__()
#         self.double_linear = nn.Sequential(
#             nn.Linear(embed_dim+5, 32, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(32, 2, bias=True)
#         )

#     def forward(self, x):
#         return self.double_linear(x)
