"""
augmentations.py

This file defines reusable torchvision data augmentation pipelines 
for training and validation.

Key features:
1. Train pipeline includes strong augmentations (flips, jitter, crop, rotation).
2. Validation pipeline is clean (just resize + normalize).
3. You can easily extend with custom transforms if needed.
"""

from torchvision import transforms


# -----------------------------
# 1. Training augmentations
# -----------------------------
train_transforms = transforms.Compose([
    # Random crop + resize to 224x224
    transforms.RandomResizedCrop(
        size=224, 
        scale=(0.08, 1.0), 
        ratio=(0.75, 1.33)
    ),

    # Random horizontal flip
    transforms.RandomHorizontalFlip(p=0.5),

    # Random color adjustments
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),

    # Random slight rotation
    transforms.RandomRotation(degrees=15),

    # Convert to tensor
    transforms.ToTensor(),

    # Normalize using ImageNet stats
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 2. Validation augmentations
# -----------------------------
val_transforms = transforms.Compose([
    transforms.Resize(256),        # resize smaller edge to 256
    transforms.CenterCrop(224),    # crop center region
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# -----------------------------
# 3. Optional custom example
# -----------------------------
# Example: Reflect padding before resizing
custom_transforms = transforms.Compose([
    transforms.Pad(25, padding_mode='reflect'),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
