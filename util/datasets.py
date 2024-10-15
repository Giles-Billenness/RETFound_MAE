# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from PIL import Image
# from torchvision.io import read_image
import pandas as pd
from torch.utils.data import Dataset

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    
    if args.dataset == 'stroke':
        dataset = StrokeDataset(csv_file=os.path.join(args.data_path, f'{is_train}_split.csv'), transform=transform)
    else:
        dataset = datasets.ImageFolder(root, transform=transform)
    

    return dataset


# Define a custom dataset class
class StrokeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        # Load the CSV file into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the image path and class label (stroke_5y) for the current index
        img_path = self.data.iloc[idx]['img_path']
        stroke_label = int(self.data.iloc[idx]['stroke_5y'])  # Convert label to int (0 or 1)

        # Load the image
        image = Image.open(img_path).convert('RGB')
        # image = read_image(img_path)

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Commented out other columns for later use
        # age = self.data.iloc[idx]['age']
        # sex = self.data.iloc[idx]['sex']
        # ethnicity = self.data.iloc[idx]['ethnicity']
        # dm = self.data.iloc[idx]['dm']
        # htn = self.data.iloc[idx]['htn']
        # hardware_model = self.data.iloc[idx]['hardwareModelName']

        # For now, return only the image and the stroke label
        # sample = {
        #     'image': image,
        #     'label': stroke_label
        # }

        return image, stroke_label#sample #

def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train=='train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
