"""
model_finetune.py

This file defines a transfer learning model (ResNet-50) with:
1. A new classifier head for our custom dataset.
2. The ability to unfreeze specific layers (layer3 + layer4).
3. Differential learning rates:
   - Classifier head (highest LR)
   - Layer4 (LR/3)
   - Layer3 (LR/9)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def get_model(num_classes, found_lr=1e-3, unfreeze_layers=True):
    # Load pretrained ResNet-50
    transfer_model = models.resnet50(pretrained=True)

    # Replace the classifier (fc layer) with new head
    in_features = transfer_model.fc.in_features
    transfer_model.fc = nn.Linear(in_features, num_classes)

    # Optionally unfreeze layer3 + layer4
    if unfreeze_layers:
        target_layers = [transfer_model.layer3, transfer_model.layer4]
        for layer in target_layers:
            for param in layer.parameters():
                param.requires_grad = True

    # Define optimizer with differential LRs
    optimizer = optim.Adam([
        {'params': transfer_model.fc.parameters(), 'lr': found_lr},        # classifier
        {'params': transfer_model.layer4.parameters(), 'lr': found_lr/3}, # last conv block
        {'params': transfer_model.layer3.parameters(), 'lr': found_lr/9}, # second-last conv block
    ])

    return transfer_model, optimizer
