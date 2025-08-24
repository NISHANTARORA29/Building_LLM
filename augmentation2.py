"""
augmentations.py

This file defines advanced augmentation pipelines and custom transforms 
for PyTorch training.

Features included:
1. Standard strong train/val augmentations.
2. Random color space conversion (RGB → HSV, LAB, YUV).
3. Custom Gaussian Noise transform.
4. Progressive resize option for "start small, grow bigger" training.
"""

import random
import torch
from torchvision import transforms
from PIL import Image


# -------------------------------------------------
# 1. Random color space conversion (Lambda transform)
# -------------------------------------------------
def random_colour_space(img: Image.Image) -> Image.Image:
    """
    Randomly convert image to one of HSV, LAB, YUV color spaces.
    """
    color_spaces = ["HSV", "LAB", "YCbCr"]  # PIL supports LAB as "LAB", YUV is "YCbCr"
    chosen_space = random.choice(color_spaces)
    return img.convert(chosen_space)

# Wrap in torchvision transform
colour_transform = transforms.RandomApply(
    [transforms.Lambda(lambda x: random_colour_space(x))],
    p=0.5  # 50% chance of conversion
)


# -------------------------------------------------
# 2. Custom Gaussian Noise Transform
# -------------------------------------------------
class Noise:
    """Adds Gaussian noise to a tensor.
    Example usage:
        transforms.Compose([
            transforms.ToTensor(),
            Noise(mean=0.1, stddev=0.05),
        ])
    """

    def __init__(self, mean=0.0, stddev=0.1):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        noise = torch.zeros_like(tensor).normal_(self.mean, self.stddev)
        return tensor.add_(noise)

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, stddev={self.stddev})"


# -------------------------------------------------
# 3. Progressive resizing utility
# -------------------------------------------------
def get_resize_transform(size: int):
    """
    Returns a Resize + standard normalization transform for progressive training.
    Args:
        size (int): target resolution (e.g., 64, 128, 224)
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


# -------------------------------------------------
# 4. Full Augmentation Pipelines
# -------------------------------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomRotation(degrees=15),
    colour_transform,       # random RGB → HSV/LAB/YUV
    transforms.ToTensor(),
    Noise(mean=0.0, stddev=0.02),  # small Gaussian noise
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
