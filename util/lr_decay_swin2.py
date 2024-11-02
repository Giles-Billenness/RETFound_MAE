def param_groups_lrd_swin(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=0.95):
    """Parameter groups for layer-wise lr decay specific to Swin Transformer"""
    param_group_names = {}
    param_groups = {}

    # Define the layer structure for Swin Transformer
    layer_structure = [2, 2, 18, 2]  # Corresponding to Swin Transformer Blocks
    num_layers = sum(layer_structure) + 1  # Total layers including embedding

    layer_scales = list(layer_decay ** (num_layers - i)
                        for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # No decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = get_layer_id_for_swin(n, layer_structure)
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]
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


def get_layer_id_for_swin(name, layer_structure):
    """Assign a parameter with its layer id for Swin Transformer"""
    if name in ['cls_token', 'pos_embed', 'patch_embed']:
        return 0
    elif name.startswith('layers'):
        parts = name.split('.')
        if 'reduction' in parts or 'norm' in parts:
            layer_num = int(parts[1])
            # Assign a layer id for reduction and norm layers
            return sum(layer_structure[:layer_num]) + 1
        if len(parts) > 3:
            layer_num = int(parts[1])
            block_num = int(parts[3])
            return sum(layer_structure[:layer_num]) + block_num + 1
        else:
            return sum(layer_structure) + 1
    else:
        return sum(layer_structure) + 1

# Example usage
# optimizer = YourOptimizer(model.parameters(), lr=base_lr)
# param_groups = param_groups_lrd_swin(model)
# optimizer.add_param_group(param_groups)
