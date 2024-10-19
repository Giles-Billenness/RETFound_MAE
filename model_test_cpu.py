import models_vit
import torch
import torch.nn as nn
import torch.optim as optim
from util.datasets import build_dataset
import util.misc as misc
import numpy as np
import os
import math
from typing import Iterable

from util.misc import NativeScalerWithGradNormCount as NativeScaler

from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_


# qrsh -l tmem=14G,h_vmem=14G,h_rt=00:59:59,gpu=true

# source activate retfound
# source activate retfoundNEW

# conda activate retfoundNEW

# source /share/apps/source_files/cuda/cuda-11.0.source

# cd /SAN/ioo/AlzeyeTempProjects/Giles/RETFOUND_meta
# python model_test_cpu.py

# Configuration
batch_size = 1
epochs = 2
model_name = 'vit'
usable_dataset = 4

# Set device to CPU for testing
device = torch.device('cpu')

# Small data subset for testing


def build_small_dataset(args):
    dataset_train = build_dataset(is_train='train', args=args)
    print(f"Full dataset length: {len(dataset_train)}")  # Debugging line
    small_dataset = torch.utils.data.Subset(
        dataset_train, indices=list(range(usable_dataset))  # First X items
    )
    print(f"Small dataset length: {len(small_dataset)}")

    return torch.utils.data.DataLoader(
        small_dataset,
        batch_size=batch_size,
        num_workers=0,  # No need for multiple workers in CPU mode
        pin_memory=False,
        drop_last=True,
        shuffle=True
    )

# Basic argument class to use with the dataset and model building


class Args:
    input_size = 224
    nb_classes = 2
    batch_size = batch_size
    mixup = 0
    cutmix = 0
    smoothing = 0
    num_workers = 0
    pin_mem = False
    clip_grad = None
    drop_path = 0.1
    epochs = epochs
    task = 'test_task'
    dist_eval = False
    global_pool = True
    dataset = "stroke"
    color_jitter = None
    aa = 'rand-m9-mstd0.5-inc1'
    reprob = 0.25
    remode = 'pixel'
    recount = 1
    resplit = False
    data_path = "/home/gbillenn/DissProj/Data/mainCSV/filteredConvertedMain/N70"
    accum_iter = 1
    warmup_epochs = 0
    blr = 5e-3
    min_lr = 1e-6
    finetune = "../Weights/RETFound_cfp_weights.pth"
    eval = False


args = Args()
args.lr = args.blr * args.batch_size / 256

# Dataset and dataloader for a small test set
print("Building small dataset")
data_loader_train = build_small_dataset(args)

print("Dataset length: ", len(data_loader_train.dataset))

# Model and training configuration


def prepare_model(model_name, input_size=224, num_classes=2):
    if model_name == 'vit':
        model = models_vit.__dict__['vit_large_patch16'](
            img_size=input_size,
            num_classes=num_classes,
            drop_path_rate=0.1,
            global_pool=True
        )
        # model.head = nn.Identity()  # Disable classification head
    elif model_name == 'swin':
        from models_swin import swinv2_large_window12to24_192to384_META
        model = swinv2_large_window12to24_192to384_META(
            pretrained=True,
            img_size=input_size,
            window_size=7,
            num_classes=0,
            drop_path_rate=0.1
        )
        model.head = nn.Identity()  # Disable classification head
    elif model_name == 'resnet':
        from models_r50 import resnet50_META
        model = resnet50_META(
            pretrained=True,
            num_classes=0,
            drop_path_rate=0.1
        )
        model.head = nn.Identity()  # Disable classification head
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


print("Building model")
# Adjust model name if needed
model = prepare_model(model_name, args.input_size, args.nb_classes)
print(model)

if model_name == 'vit':
    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            assert set(msg.missing_keys) >= {  # change to is subset/superset of, as the linear layer keys are not required
                'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            assert set(msg.missing_keys) >= {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # hmmmmmmmmmm remove?????????
        trunc_normal_(model.head.weight, std=2e-5)
        model.head = nn.Identity()  # Disable classification head

print(model)

model.to(device)  # Transfer model to CPU

# Basic optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

# Loss function
criterion = nn.CrossEntropyLoss()

loss_scaler = NativeScaler()


def train_one_epoch(model: nn.Module, criterion: nn.Module,
                    data_loader: Iterable, optimizer: optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    max_norm: float = 0, args: Args = None):
    model.train()
    header = f'Epoch: [{epoch}]'

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    for data_iter_step, samples in enumerate(data_loader):
        # Update learning rate if needed
        if data_iter_step % accum_iter == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"{header} - Iteration: {data_iter_step}, Learning Rate: {lr}")

        # Transfer samples to device
        images = samples['image'].to(device, non_blocking=True)
        targets = samples['label'].to(device, non_blocking=True)
        risk_factors = samples['risk_factors'].to(device, non_blocking=True)

        # No mixed precision
        outputs = model(images, risk_factors)
        loss = criterion(outputs, targets)

        loss_value = loss.item()
        print(f"Loss value: {loss_value}")

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            return  # Exiting for cleaner shutdown

        loss /= accum_iter
        loss.backward(create_graph=False)  # No scaling needed

        # Gradient clipping
        if max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad()

        print(f"Loss: {loss_value:.4f}, Step: {data_iter_step + 1}")

    print("Training for one epoch completed.")


# Training loop for one epoch
print("Training for one epoch")
for epoch in range(1, args.epochs + 1):
    train_one_epoch(
        model=model, criterion=criterion,
        data_loader=data_loader_train, optimizer=optimizer,
        device=device, epoch=epoch, loss_scaler=loss_scaler,
        max_norm=args.clip_grad, args=args
    )
    print(f"Finished epoch {epoch}.")

# Print training stats
print("Training completed.")
