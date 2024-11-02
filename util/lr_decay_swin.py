# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import json


# def param_groups_lrd_swin(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
#     param_group_names = {}
#     param_groups = {}

#     # Define layer decay scales
#     num_layers = len(model.layers) + 2  # Adjust based on Swin layer structure
#     layer_scales = list(layer_decay ** (num_layers - i)
#                         for i in range(num_layers + 1))

#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue
#         if p.ndim == 1 or n in no_weight_decay_list:
#             g_decay = "no_decay"
#             this_decay = 0.0
#         else:
#             g_decay = "decay"
#             this_decay = weight_decay

#         layer_id = get_layer_id_for_swin(n, num_layers)
#         group_name = "layer_%d_%s" % (layer_id, g_decay)

#         if group_name not in param_group_names:
#             this_scale = layer_scales[layer_id]
#             param_group_names[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": []
#             }
#             param_groups[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": []
#             }

#         param_group_names[group_name]["params"].append(n)
#         param_groups[group_name]["params"].append(p)

#     return list(param_groups.values())


# def get_layer_id_for_swin(name, num_layers):
#     if "patch_embed" in name:
#         return 0
#     elif "layers" in name:
#         layer_idx = int(name.split('.')[1])
#         return layer_idx + 1
#     elif "downsample" in name:
#         return int(name.split('.')[1]) + 1
#     else:
#         return num_layers

# # Example usage:
# # param_groups = param_groups_lrd_swin(model)

def param_groups_lrd_swin(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.75):
    """
    Parameter groups for layer-wise lr decay for Swin Transformer models.
    """
    param_group_names = {}
    param_groups = {}

    # The total number of layers is the sum of blocks in all stages + 1 for final normalization layer
    num_blocks = [2, 2, 18, 2]
    # 24 layers in total (based on 4 stages with (2, 2, 18, 2) blocks)
    num_layers = sum(num_blocks) + 1

    layer_scales = list(layer_decay ** (num_layers - i)
                        # Scale list for all layers
                        for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # No weight decay for 1D parameters or listed in no_weight_decay_list
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.0
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Get the layer id using the Swin-specific layer id function
        layer_id = get_layer_id_for_swin(n, num_blocks)

        # Ensure layer_id does not exceed num_layers
        layer_id = min(layer_id, num_layers)

        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]  # Get the scale for this layer

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    return list(param_groups.values())


# def param_groups_lrd_swin(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.95):
#     """
#     Parameter groups for layer-wise lr decay for Swin Transformers.
#     """
#     param_group_names = {}
#     param_groups = {}

#     # Define number of blocks per stage
#     num_blocks = [2, 2, 18, 2]
#     total_blocks = sum(num_blocks) + 1

#     # Scaling factor per layer
#     layer_scales = [layer_decay ** i for i in range(total_blocks + 1)]

#     for n, p in model.named_parameters():
#         if not p.requires_grad:
#             continue

#         # no decay: all 1D parameters and model-specific ones
#         if p.ndim == 1 or n in no_weight_decay_list:
#             g_decay = "no_decay"
#             this_decay = 0.0
#         else:
#             g_decay = "decay"
#             this_decay = weight_decay

#         # Assign layer id based on the parameter name for Swin
#         layer_id = get_layer_id_for_swin(n, num_blocks)

#         group_name = "layer_%d_%s" % (layer_id, g_decay)

#         if group_name not in param_group_names:
#             this_scale = layer_scales[layer_id]

#             param_group_names[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }
#             param_groups[group_name] = {
#                 "lr_scale": this_scale,
#                 "weight_decay": this_decay,
#                 "params": [],
#             }

#         param_group_names[group_name]["params"].append(n)
#         param_groups[group_name]["params"].append(p)

#     return list(param_groups.values())


def get_layer_id_for_swin(name, num_blocks):
    """
    Assign a parameter with its layer id for Swin Transformers.
    """
    if name.startswith('patch_embed'):
        return 0  # Patch embedding is considered layer 0
    elif name.startswith('layers'):
        # For example, 'layers.0.blocks.0.attn.qkv.weight'
        parts = name.split('.')
        stage_id = int(parts[1])  # Get the stage number (0 to 3)

        if parts[2] == 'blocks':
            block_id = int(parts[3])  # Get the block number
            # Offset from previous stages
            layer_offset = sum(num_blocks[:stage_id])
            return layer_offset + block_id + 1  # +1 to start from layer 1
        else:
            # If it's not a block (e.g., 'downsample'), assign to the same layer as the downsample operation
            # Use the same offset as blocks in the same stage
            layer_offset = sum(num_blocks[:stage_id])
            return layer_offset + 1  # Assign to the layer after patch embedding
    else:
        # Assign other parts (like 'norm' or 'head') to the last layer
        return sum(num_blocks) + 1

# def get_layer_id_for_swin(name, num_blocks):
#     """
#     Assign a parameter with its layer id for Swin Transformers.
#     """
#     if name.startswith('patch_embed'):
#         return 0  # Patch embedding is considered layer 0
#     elif name.startswith('layers'):
#         # For example, 'layers.0.blocks.0.attn.qkv.weight'
#         stage_id = int(name.split('.')[1])  # Get the stage number (0 to 3)
#         block_id = int(name.split('.')[3])  # Get the block number

#         # Map stage and block to a unique layer id
#         # Offset from previous stages
#         layer_offset = sum(num_blocks[:stage_id])
#         return layer_offset + block_id + 1  # +1 to start from layer 1
#     else:
#         # Assign other parts (like 'norm' or 'head') to the last layer
#         return sum(num_blocks) + 1
