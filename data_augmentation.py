"""
train.py

This file handles:
1. Data augmentation with torchvision.transforms
2. Dataloaders for train/val datasets
3. Training loop using model + optimizer from model_finetune.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model_finetune import get_model


# -----------------------------
# 1. Data augmentation
# -----------------------------
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),         # random crop + resize
    transforms.RandomHorizontalFlip(),         # flip image
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2,
                           hue=0.1),          # color variations
    transforms.RandomRotation(15),             # slight rotations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                         [0.229, 0.224, 0.225])  # ImageNet std
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 2. Datasets + Dataloaders
# -----------------------------
train_dataset = datasets.ImageFolder("data/train", transform=train_transforms)
val_dataset = datasets.ImageFolder("data/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# -----------------------------
# 3. Load model + optimizer
# -----------------------------
num_classes = len(train_dataset.classes)
model, optimizer = get_model(num_classes=num_classes, found_lr=1e-3, unfreeze_layers=True)

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# 4. Training loop
# -----------------------------
def train_model(epochs=5):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] Training Loss: {avg_loss:.4f}")

        # Validation step
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total
        print(f"Validation Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train_model(epochs=10)
