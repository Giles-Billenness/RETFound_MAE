# import timm.models.swin_transformer_v2
from timm.models.swin_transformer_v2 import SwinTransformerV2
from timm.models._builder import build_model_with_cfg
from timm.models.swin_transformer_v2 import checkpoint_filter_fn


from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer

import torch
import torch.nn as nn


class SwinTransformerV2WithMetadata(SwinTransformerV2):
    def __init__(self, head_dropout=0.0, num_risk_factors=5, **kwargs):
        super(SwinTransformerV2WithMetadata, self).__init__(**kwargs)

        # Additional linear layers to process concatenated features (image embeddings + risk factors)
        # Global pooling to reduce spatial dimensions
        # self.global_pool = nn.AdaptiveAvgPool1d(1)  # ???????????????????

        distil_neurons = 512  # 32

        # define pooling layer for dim reduction
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear1 = nn.Linear(
            self.num_features, distil_neurons, bias=True)  # For image features
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=head_dropout)
        # Add risk factor dimension
        self.linear2 = nn.Linear(
            distil_neurons + num_risk_factors, 2, bias=True)

    def forward(self, x, risk_factors):
        # Extract features from the image using the Swin Transformer layers
        x = self.forward_features(x)  # output (B, T(H/4 X W/4), C)
        # print("Shape after forward_features:", x.shape)  # Debugging line

        # Apply global pooling to reduce spatial dimensions (B, C, H, W) -> (B, C)
        # Global Average Pooling over height and width dimensions
        # x = x.permute(0, 3, 1, 2)  # reshape to [1, 1536, 7, 7]
        # x = self.global_pool(x)  # Shape should now be (B, C, 1, 1)
        # print("Shape after global pooling nn.AdaptiveAvgPool2d(1):",x.shape)  # Debugging line
        # x = x.view(x.size(0), -1)  # Flatten to (B, C)
        # print("Shape after view change x.view(x.size(0), -1):",x.shape)  # Debugging line

        # other way:
        x = x.permute(0, 3, 1, 2)  # reshape to [1, 1536, 7, 7]
        x = x.mean(dim=[2, 3])  # should now be (B, C, 7, 7) Flatten to (B, C)

        # First layer to process the image features
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)

        # Concatenate risk factors with the image features
        x = torch.cat((x, risk_factors), dim=1)

        # Final classification layer
        x = self.linear2(x)
        return x


def swinv2_large_window12to24_192to384_META(pretrained=False, **kwargs):
    model_args = dict(
        window_size=24, embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48),
        pretrained_window_sizes=(12, 12, 12, 6))
    return _create_swin_transformer_v2_META(
        'swinv2_large_window12to24_192to384', pretrained=pretrained, **dict(model_args, **kwargs))


def _create_swin_transformer_v2_META(variant, pretrained=False, **kwargs):
    default_out_indices = tuple(i for i, _ in enumerate(
        kwargs.get('depths', (1, 1, 1, 1))))
    out_indices = kwargs.pop('out_indices', default_out_indices)

    model = build_model_with_cfg(
        SwinTransformerV2WithMetadata, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True, out_indices=out_indices),
        **kwargs)
    return model


# # Import your custom class
# model = SwinTransformerV2WithMetadata(
#     img_size=args.input_size,         # Image size as defined by your args
#     window_size=7,                    # Keep the window size as you had it
#     # Number of classes (for stroke classification)
#     num_classes=args.nb_classes,
#     # Embedding dimension (this may be adjusted according to your model configuration)
#     embed_dim=96,
#     # Depths for each Swin stage (adapt this to the specific model version you're using)
#     depths=(2, 2, 6, 2),
#     num_heads=(3, 6, 12, 24),         # Attention heads per stage
#     # Number of metadata features (risk factors: age, sex, etc.)
#     num_risk_factors=5,
#     pretrained=True                   # Load pretrained weights if applicable
# )


# def Swin_large_MetaData(**kwargs):
#     model = SwinTransformerV2WithMetadata(
#         patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
#     return model


# model = timm.create_model('swinv2_large_window12to24_192to384.ms_in22k_ft_in1k',
#                           pretrained=True, img_size=args.input_size, window_size=7, num_classes=args.nb_classes)
