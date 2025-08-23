import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional: Squeeze-and-Excitation block for channel attention
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch, channels, 1, 1)
        return x * y.expand_as(x)

# Residual block (ResNet-style)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        # Adjust input (for skip connection) if dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return F.relu(out)

# Deep CNN model
class DeepCNN(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = ResidualBlock(64, 128, stride=2, use_se=True)
        self.layer3 = ResidualBlock(128, 256, stride=2, use_se=True)
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SEBlock(512)
        )
        self.layer5 = nn.AdaptiveAvgPool2d((1, 1))  # Global average pool
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

# Instantiate the model (e.g., for CIFAR-10)
model = DeepCNN(input_channels=3, num_classes=10)
print(model)
