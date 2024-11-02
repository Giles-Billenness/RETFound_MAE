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
    def __init__(self, head_dropout=0.0, num_risk_factors=5, distil_neurons=32, **kwargs):
        super(SwinTransformerV2WithMetadata, self).__init__(**kwargs)

        self.distil_neurons = distil_neurons
        # Global pooling to reduce spatial dimensions
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        if (self.distil_neurons is not None):  # 32,512
            self.linear1 = nn.Linear(
                self.num_features, self.distil_neurons, bias=True)  # For image features
            self.activation = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=head_dropout)
            # Add risk factor dimension
            self.linear2 = nn.Linear(
                self.distil_neurons + num_risk_factors, 2, bias=True)  # final layer
        else:
            self.linear2 = nn.Linear(
                self.num_features + num_risk_factors, 2, bias=True)  # for each extra channel

    def forward(self, x, risk_factors):
        # Extract features from the image using the Swin Transformer layers
        x = self.forward_features(x)  # output (B, T(H/4 X W/4), C)
        # print("Shape after forward_features:", x.shape)  # Debugging line

        if (self.distil_neurons is not None):
            x = x.permute(0, 3, 1, 2)  # reshape to [1, 1536, 7, 7]
            # should now be (B, C, 7, 7) Flatten to (B, C)
            x = x.mean(dim=[2, 3])

            # First layer to process the image features
            x = self.linear1(x)  # distil flattened embeddings
            x = self.activation(x)
            x = self.dropout(x)
            # Concatenate risk factors with the image features
            x = torch.cat((x, risk_factors), dim=1)
        else:
            # risk_factors.shape = [batch_size, 5]
            # Create extra channel for metadata, metadata in form of (B, 5) -> (B, 1, 7, 7) for concatenation
            # risk_factors_channel = risk_factors.unsqueeze(2).unsqueeze(
            # 3).expand(-1, -1, 7, 7)  # Repeat across H and W
            # risk_factors_channel.shape = [batch_size, 5, 7, 7]

            # Change shape to [batch_size, 1536, 7, 7]
            x = x.permute(0, 3, 1, 2)

            # Concatenate the metadata channel with the image features
            # x = torch.cat((x, risk_factors_channel), dim=1)
            # permute back to [batch_size, 7, 7, 1536+5]
            # x = x.permute(0, 2, 3, 1)
            # print(f"Shape after concatenation: {x.shape}")
            # Apply global pooling to reduce spatial dimensions
            # x = x.mean(dim=[1, 2])  # (B, 1541)
            x = self.global_pool(x) # (B, 1541)
            # print(f"Shape after global pooling: {x.shape}")
            x = x.view(x.size(0), -1)  # Flatten to (B, 1541)
            # print(f"Shape after flattening: {x.shape}")
            x = torch.cat((x, risk_factors), dim=1)
            # print(f"Shape after concatenation: {x.shape}")
            # print(x)

        # Final classification layer
        x = self.linear2(x)
        # print(f"Shape after linear2: {x.shape}")
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
