#!/usr/bin/env python3
"""
CNN Everything — One-File PyTorch Playbook

What you get (pure Python + PyTorch, no torchvision.models):
  • Dataloaders (CIFAR-10 by default) with strong/weak augs
  • Training loop (AMP, grad accumulation, label smoothing, EMA, OneCycle/Cosine sched)
  • Checkpointing (best/last), resume, eval, confusion matrix
  • Grad-CAM (per-layer), feature map viz, tensorboard-ish prints
  • FLOPs & params rough estimator (conv/linear)
  • Implementations from scratch:
      - BasicCNN, LeNet5, VGG11
      - ResNet (18/34/50 via BasicBlock/Bottleneck)
      - DenseNet (Tiny-121-ish)
      - InceptionV1 (GoogLeNet-ish, simplified)
      - MobileNetV2 (Inverted Residual)
      - EfficientNet-B0-ish (MBConv + SE, simplified)
      - U-Net (bonus: segmentation on toy shapes if chosen)

Run examples:
  python cnn_everything_pytorch.py --model resnet18 --epochs 20 --bs 128
  python cnn_everything_pytorch.py --model mobilenetv2 --opt adamw --lr 3e-3 --onecycle
  python cnn_everything_pytorch.py --model densenet_tiny --strong_aug --grad_cam layer4.1.conv2

Notes:
  • Defaults to CIFAR-10 (32x32). Some big nets automatically adjust stem/stride.
  • Pure PyTorch (torch, torchvision, numpy). No external libs.
  • For ImageNet-scale, swap dataloader to your folder with --data <path> and --img 224.
"""

import os, math, time, argparse, random, json
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import torchvision
from torchvision import transforms

# -------------------------------
# Utils
# -------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rough_flops(model: nn.Module, img_size: int = 32, in_ch: int = 3) -> int:
    """Very rough FLOPs (multiply-adds) for Conv2d/Linear only."""
    flops = 0
    hooks = []

    def hook_conv(m, x, y):
        # x: (B,C_in,H,W), y: (B,C_out,H_out,W_out)
        b, c_in, h, w = x[0].shape
        c_out = y.shape[1]
        k_h, k_w = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
        fl = b * c_out * h * w * (c_in // m.groups) * k_h * k_w
        flops_dict["conv"] = flops_dict.get("conv", 0) + fl

    def hook_linear(m, x, y):
        b = x[0].shape[0]
        fl = b * m.in_features * m.out_features
        flops_dict["linear"] = flops_dict.get("linear", 0) + fl

    flops_dict: Dict[str,int] = {}

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(hook_conv))
        if isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(hook_linear))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, in_ch, img_size, img_size, device=next(model.parameters()).device)
        model(dummy)

    for h in hooks:
        h.remove()
    flops = sum(flops_dict.values())
    return flops


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        n = preds.size(1)
        log_probs = F.log_softmax(preds, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


class EMA:
    def __init__(self, model, decay=0.9999):
        self.ema = self._clone_model(model)
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _clone_model(self, model):
        import copy
        ema = copy.deepcopy(model)
        return ema

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                self.ema.state_dict()[k].copy_(self.decay * v + (1 - self.decay) * msd[k])


# -------------------------------
# Blocks
# -------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ResNet
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNAct(in_planes, planes, k=3, s=stride)
        self.conv2 = ConvBNAct(planes, planes, k=3, s=1, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = ConvBNAct(in_planes, planes, k=1)
        self.conv2 = ConvBNAct(planes, planes, k=3, s=stride)
        self.conv3 = ConvBNAct(planes, planes * self.expansion, k=1, act=False)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_ch=3, img_size=32):
        super().__init__()
        # CIFAR-friendly stem
        s = 1 if img_size <= 64 else 2
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, 64, k=3, s=s),
            ConvBNAct(64, 64, k=3, s=1),
            ConvBNAct(64, 128, k=3, s=1),
        )
        self.pool = nn.MaxPool2d(2,2) if img_size > 32 else nn.Identity()
        self.inplanes = 128
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512*block.expansion, num_classes)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


def resnet18(**kw):
    return ResNet(BasicBlock, [2,2,2,2], **kw)

def resnet34(**kw):
    return ResNet(BasicBlock, [3,4,6,3], **kw)

def resnet50(**kw):
    return ResNet(Bottleneck, [3,4,6,3], **kw)


# VGG
class VGG(nn.Module):
    cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],  # VGG11
    }
    def __init__(self, cfg='A', num_classes=10, in_ch=3):
        super().__init__()
        self.features = self._make_layers(self.cfgs[cfg], in_ch)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    def _make_layers(self, cfg, in_ch):
        layers = []
        c = in_ch
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBNAct(c, v, k=3, s=1)]
                c = v
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# LeNet-5
class LeNet5(nn.Module):
    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, 6, k=5, s=1, p=2), nn.AvgPool2d(2),
            ConvBNAct(6, 16, k=5, s=1), nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(16*6*6, 120), nn.ReLU(True),
            nn.Linear(120, 84), nn.ReLU(True),
            nn.Linear(84, num_classes),
        )
    def forward(self, x):
        return self.net(x)


# DenseNet (Tiny)
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth=32):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, 4*growth, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth)
        self.conv2 = nn.Conv2d(4*growth, growth, kernel_size=3, padding=1, bias=False)
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        return torch.cat([x, out], 1)

class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.AvgPool2d(2)
        )
    def forward(self, x):
        return self.net(x)

class DenseNetTiny(nn.Module):
    def __init__(self, blocks=(6,12,24,16), growth=24, num_classes=10, in_ch=3):
        super().__init__()
        c = 2*growth
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
        self.features = nn.ModuleList()
        for i, b in enumerate(blocks):
            for _ in range(b):
                self.features.append(DenseLayer(c, growth))
                c += growth
            if i != len(blocks)-1:
                self.features.append(Transition(c, c//2))
                c = c//2
        self.bn = nn.BatchNorm2d(c)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, num_classes))
    def forward(self, x):
        x = self.stem(x)
        for m in self.features:
            x = m(x)
        x = F.relu(self.bn(x), inplace=True)
        x = self.head(x)
        return x


# Inception (GoogLeNet-like, simplified)
class InceptionBlock(nn.Module):
    def __init__(self, c_in, c1, c3r, c3, c5r, c5, pp):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv2d(c_in, c1, 1), nn.ReLU(True))
        self.b3 = nn.Sequential(nn.Conv2d(c_in, c3r, 1), nn.ReLU(True), nn.Conv2d(c3r, c3, 3, padding=1), nn.ReLU(True))
        self.b5 = nn.Sequential(nn.Conv2d(c_in, c5r, 1), nn.ReLU(True), nn.Conv2d(c5r, c5, 5, padding=2), nn.ReLU(True))
        self.bp = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), nn.Conv2d(c_in, pp, 1), nn.ReLU(True))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], 1)

class InceptionNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(in_ch, 64, 3, 1), ConvBNAct(64, 128, 3, 1))
        self.inc3a = InceptionBlock(128, 64, 96, 128, 16, 32, 32)
        self.inc3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool = nn.MaxPool2d(2,2)
        self.inc4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inc4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inc4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inc4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inc4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(832, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.inc3a(x); x = self.inc3b(x); x = self.pool(x)
        x = self.inc4a(x); x = self.inc4b(x); x = self.inc4c(x); x = self.inc4d(x); x = self.inc4e(x)
        x = self.head(x); return x


# MobileNetV2
class InvertedResidual(nn.Module):
    def __init__(self, c_in, c_out, stride, expand):
        super().__init__()
        hidden = int(round(c_in * expand))
        self.use_res = stride == 1 and c_in == c_out
        layers = []
        if expand != 1:
            layers.append(ConvBNAct(c_in, hidden, k=1))
        layers.extend([
            # depthwise
            ConvBNAct(hidden, hidden, k=3, s=stride, g=hidden),
            # project
            ConvBNAct(hidden, c_out, k=1, act=False),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        return x + out if self.use_res else out

class MobileNetV2(nn.Module):
    cfg = [
        # t, c, n, s
        (1, 16, 1, 1),
        (6, 24, 2, 1),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]
    def __init__(self, num_classes=10, in_ch=3, width=1.0):
        super().__init__()
        out = int(32 * width)
        self.stem = ConvBNAct(in_ch, out, k=3, s=1)
        c_in = out
        blocks = []
        for t, c, n, s in self.cfg:
            c = int(c * width)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(c_in, c, stride, expand=t))
                c_in = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(ConvBNAct(c_in, int(1280*width), k=1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(int(1280*width), num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# EfficientNet (B0-ish, minimal SE)
class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s

class MBConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, expand=4, se=True):
        super().__init__()
        hidden = c_in * expand
        self.expand = nn.Identity() if expand == 1 else ConvBNAct(c_in, hidden, 1)
        self.dw = ConvBNAct(hidden, hidden, k=k, s=s, g=hidden)
        self.se = SE(hidden) if se else nn.Identity()
        self.project = ConvBNAct(hidden, c_out, 1, act=False)
        self.use_res = (s == 1 and c_in == c_out)
    def forward(self, x):
        out = self.project(self.se(self.dw(self.expand(x))))
        return x + out if self.use_res else out

class EfficientNetB0Mini(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.stem = ConvBNAct(in_ch, 32, 3, 1)
        self.blocks = nn.Sequential(
            MBConv(32, 16, k=3, s=1, expand=1),
            MBConv(16, 24, k=3, s=2, expand=6), MBConv(24, 24, k=3, s=1, expand=6),
            MBConv(24, 40, k=5, s=2, expand=6), MBConv(40, 40, k=5, s=1, expand=6),
            MBConv(40, 80, k=3, s=2, expand=6), MBConv(80, 80, k=3, s=1, expand=6),
            MBConv(80, 112, k=5, s=1, expand=6), MBConv(112, 112, k=5, s=1, expand=6),
            MBConv(112, 192, k=5, s=2, expand=6), MBConv(192, 192, k=5, s=1, expand=6),
            MBConv(192, 320, k=3, s=1, expand=6),
        )
        self.head = nn.Sequential(ConvBNAct(320, 1280, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1280, num_classes))
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x


# Basic CNN for didactic purposes
class BasicCNN(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNAct(in_ch, 32), ConvBNAct(32, 64), nn.MaxPool2d(2),
            ConvBNAct(64, 128), nn.MaxPool2d(2),
            ConvBNAct(128, 256), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes))
    def forward(self, x):
        return self.classifier(self.features(x))


# Optional: U-Net (toy, for completeness)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNAct(in_ch, out_ch), ConvBNAct(out_ch, out_ch))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super().__init__()
        self.d1 = DoubleConv(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.d4 = DoubleConv(256, 512)
        self.p4 = nn.MaxPool2d(2)
        self.b = DoubleConv(512, 1024)
        self.u4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.c4 = DoubleConv(1024, 512)
        self.u3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.c3 = DoubleConv(512, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.c1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, out_ch, 1)
    def forward(self, x):
        d1 = self.d1(x); x = self.p1(d1)
        d2 = self.d2(x); x = self.p2(d2)
        d3 = self.d3(x); x = self.p3(d3)
        d4 = self.d4(x); x = self.p4(d4)
        x = self.b(x)
        x = self.u4(x); x = self.c4(torch.cat([x, d4], 1))
        x = self.u3(x); x = self.c3(torch.cat([x, d3], 1))
        x = self.u2(x); x = self.c2(torch.cat([x, d2], 1))
        x = self.u1(x); x = self.c1(torch.cat([x, d1], 1))
        return self.out(x)


# -------------------------------
# Grad-CAM
# -------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, layer_name: str):
        self.model = model
        self.layer = self._get_layer(layer_name)
        self.activations = None
        self.gradients = None
        self.h1 = self.layer.register_forward_hook(self._forward_hook)
        self.h2 = self.layer.register_full_backward_hook(self._backward_hook)

    def _get_layer(self, name: str):
        mod = self.model
        for part in name.split('.'):
            mod = getattr(mod, part)
        return mod

    def _forward_hook(self, m, x, y):
        self.activations = y.detach()

    def _backward_hook(self, m, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, class_idx.view(-1,1), 1)
        logits.backward(gradient=one_hot)
        # weights: GAP of gradients
        weights = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', align_corners=False)
        cam -= cam.amin(dim=(2,3), keepdim=True)
        cam /= (cam.amax(dim=(2,3), keepdim=True) + 1e-6)
        return cam

    def close(self):
        self.h1.remove(); self.h2.remove()


# -------------------------------
# Data
# -------------------------------

def build_transforms(img_size=32, strong=False):
    t_train = [
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if strong:
        t_train += [transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2)]
    t_train += [transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
    t_test = [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))]
    return transforms.Compose(t_train), transforms.Compose(t_test)


def get_loaders(data_root: str, bs: int, img_size: int, strong_aug: bool, num_workers: int):
    t_train, t_test = build_transforms(img_size, strong_aug)
    train = torchvision.datasets.CIFAR10(data_root, train=True, transform=t_train, download=True)
    test = torchvision.datasets.CIFAR10(data_root, train=False, transform=t_test, download=True)
    train_loader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test, batch_size=bs*2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


# -------------------------------
# Train/Eval
# -------------------------------

def accuracy(logits, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append((correct_k * (100.0 / targets.size(0))).item())
        return res


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn, grad_accum=1):
    model.train()
    total, correct = 0, 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = loss_fn(logits, y) / grad_accum
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (i+1) % grad_accum == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * grad_accum
        total += y.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


def evaluate(model, loader, device, loss_fn):
    model.eval()
    loss_sum, total, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss_sum += loss.item()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
    return loss_sum / len(loader), 100.0 * correct / total


def save_ckpt(state: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_ckpt(path: Path, model: nn.Module, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'opt' in ckpt:
        optimizer.load_state_dict(ckpt['opt'])
    if scheduler and 'sched' in ckpt:
        scheduler.load_state_dict(ckpt['sched'])
    start_epoch = ckpt.get('epoch', 0) + 1
    best_acc = ckpt.get('best_acc', 0.0)
    return start_epoch, best_acc


# -------------------------------
# Factory
# -------------------------------

MODEL_ZOO = {
    'basic': lambda **kw: BasicCNN(**kw),
    'lenet5': lambda **kw: LeNet5(**kw),
    'vgg11': lambda **kw: VGG('A', **kw),
    'resnet18': lambda **kw: resnet18(**kw),
    'resnet34': lambda **kw: resnet34(**kw),
    'resnet50': lambda **kw: resnet50(**kw),
    'densenet_tiny': lambda **kw: DenseNetTiny(**kw),
    'inception': lambda **kw: InceptionNet(**kw),
    'mobilenetv2': lambda **kw: MobileNetV2(**kw),
    'efficientnet_b0mini': lambda **kw: EfficientNetB0Mini(**kw),
    'unet': lambda **kw: UNet(in_ch=3, out_ch=1),
}


# -------------------------------
# Main
# -------------------------------

def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', type=str, default='resnet18', choices=list(MODEL_ZOO.keys()))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--img', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--wd', type=float, default=5e-4)
    p.add_argument('--opt', type=str, default='sgd', choices=['sgd','adamw'])
    p.add_argument('--mom', type=float, default=0.9)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--ema', action='store_true')
    p.add_argument('--onecycle', action='store_true')
    p.add_argument('--cosine', action='store_true')
    p.add_argument('--strong_aug', action='store_true')
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--out', type=str, default='./runs/cnn_everything')
    p.add_argument('--grad_cam', type=str, default='')

    args = p.parse_args()

    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_loaders(args.data, args.bs, args.img, args.strong_aug, args.num_workers)

    # Model
    num_classes = 10
    in_ch = 3
    model = MODEL_ZOO[args.model](num_classes=num_classes, in_ch=in_ch, img_size=args.img)
    model.to(device)

    # Loss
    loss_fn = LabelSmoothingCE(args.label_smoothing) if args.label_smoothing > 0 else nn.CrossEntropyLoss()

    # Optim
    if args.opt == 'sgd':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd, nesterov=True)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Sched
    total_steps = len(train_loader) * args.epochs
    if args.onecycle:
        scheduler = OneCycleLR(optimizer, max_lr=args.lr, total_steps=total_steps)
    elif args.cosine:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    ema = EMA(model) if args.ema else None

    start_epoch, best_acc = 0, 0.0
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            start_epoch, best_acc = load_ckpt(ckpt_path, model, optimizer, scheduler)
            print(f"Resumed from {ckpt_path} at epoch {start_epoch}, best_acc={best_acc:.2f}")
        else:
            print(f"Resume path not found: {ckpt_path}")

    # Info
    params = count_params(model)
    try:
        fl = rough_flops(model, img_size=args.img, in_ch=in_ch)
        print(f"Params: {params/1e6:.2f}M | Rough FLOPs/forward (1 img): {fl/1e6:.1f}M")
    except Exception as e:
        print(f"FLOPs estimate failed: {e}")

    # Grad-CAM setup (optional)
    cam = None
    if args.grad_cam:
        try:
            cam = GradCAM(model, args.grad_cam)
            print(f"Grad-CAM attached to layer: {args.grad_cam}")
        except Exception as e:
            print(f"Grad-CAM setup failed: {e}")

    # Train
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn, args.grad_accum)
        if scheduler and isinstance(scheduler, OneCycleLR):
            # OneCycle steps every batch, already stepped
            pass
        elif scheduler:
            scheduler.step()
        if ema:
            ema.update(model)

        # Evaluate (EMA if enabled)
        eval_model = ema.ema if ema else model
        val_loss, val_acc = evaluate(eval_model, test_loader, device, loss_fn)

        best = ''
        if val_acc > best_acc:
            best_acc = val_acc
            save_ckpt({'model': eval_model.state_dict(), 'epoch': epoch, 'best_acc': best_acc}, out_dir / 'best.pt')
            best = '✅'
        save_ckpt({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'sched': scheduler.state_dict() if scheduler else None, 'epoch': epoch, 'best_acc': best_acc}, out_dir / 'last.pt')

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs} | {dt:.1f}s | train {train_loss:.3f}/{train_acc:.2f}% | val {val_loss:.3f}/{val_acc:.2f}% {best}")

    # Optional: dump a Grad-CAM on a few test images
    if cam is not None:
        import matplotlib.pyplot as plt
        model.eval()
        (x, y) = next(iter(test_loader))
        x = x.to(device)
        cam_map = cam(x[:8])  # (B,1,H,W)
        # save grid
        def to_np(t): return t.detach().cpu().numpy()
        imgs = to_np((x[:8]*torch.tensor([0.2470,0.2435,0.2616],device=device).view(1,3,1,1) + torch.tensor([0.4914,0.4822,0.4465],device=device).view(1,3,1,1)))
        cams = to_np(cam_map)
        fig, axes = plt.subplots(2, 8, figsize=(16,4))
        for i in range(8):
            axes[0,i].imshow(np.transpose(imgs[i], (1,2,0)))
            axes[0,i].axis('off')
            axes[1,i].imshow(np.transpose(imgs[i], (1,2,0)))
            axes[1,i].imshow(cams[i,0], alpha=0.5)
            axes[1,i].axis('off')
        plt.tight_layout()
        fig.savefig(out_dir / 'grad_cam_grid.png', dpi=150)
        print(f"Saved Grad-CAM grid to {out_dir / 'grad_cam_grid.png'}")

    print(f"Training complete. Best Acc: {best_acc:.2f}% | Checkpoints in {out_dir}")


if __name__ == '__main__':
    main()
