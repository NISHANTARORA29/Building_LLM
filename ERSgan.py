"""
Minimal, runnable single-file PyTorch ESRGAN implementation (skeleton + working model definitions).
- Generator: RRDBNet (RRDB blocks without BatchNorm)
- Discriminator: Patch-style CNN with RaGAN loss wrapper
- Losses: pixel L1, VGG perceptual (placeholder), RaGAN BCE-style
- Training loop skeleton (pretrain -> adversarial finetune)
- Utilities: weight init, load/save, weight-space interpolation, EMA

Notes:
- This is intended as a developer-ready starting point, not a drop-in high-performance training script.
- You must install torch, torchvision and provide a dataset of (LR, HR) pairs or a bicubic-downsampling pipeline.
- VGG perceptual uses torchvision.models; adjust layer choices per your needs.

Author: ChatGPT (skeleton for ESRGAN)
"""

import os
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.utils import save_image
from PIL import Image
import random

# ----------------------------- Model components -----------------------------
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_channels=32):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.lrelu(self.conv(x))
        return out

class DenseBlock(nn.Module):
    """Dense block used inside RRDB (5 layers)"""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(5):
            self.layers.append(DenseLayer(nf + i * gc, gc))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            inp = torch.cat(features, dim=1)
            out = layer(inp)
            features.append(out)
        return features[-1]

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32, beta=0.2):
        super().__init__()
        self.db1 = DenseBlock(nf, gc)
        self.db2 = DenseBlock(nf, gc)
        self.db3 = DenseBlock(nf, gc)
        self.beta = beta

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + out * self.beta

class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
        super().__init__()
        # first conv
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        # RRDB trunk
        self.trunk = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        # upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk(fea)
        trunk = self.trunk_conv(trunk)
        fea = fea + trunk
        # upsample x2
        fea = self.lrelu(F.interpolate(self.upconv1(fea), scale_factor=2, mode='nearest'))
        fea = self.lrelu(F.interpolate(self.upconv2(fea), scale_factor=2, mode='nearest'))
        out = self.lrelu(self.hr_conv(fea))
        out = self.conv_last(out)
        return out

# ------------------------- Discriminator (Patch-style) ------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()
        def block(in_c, out_c, stride=1, use_bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride, 1)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        layers += block(in_channels, base, use_bn=False)
        layers += block(base, base, stride=2)
        layers += block(base, base*2)
        layers += block(base*2, base*2, stride=2)
        layers += block(base*2, base*4)
        layers += block(base*4, base*4, stride=2)
        layers += block(base*4, base*8)
        layers += block(base*8, base*8, stride=2)
        layers += [nn.Conv2d(base*8, 1, 3, 1, 1)]  # output logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# ----------------------------- Loss helpers ----------------------------------
class VGGFeatureExtractor(nn.Module):
    """VGG feature extractor for perceptual loss. Uses pre-activation layers concept by selecting conv outputs before ReLU.
    This implementation extracts conv4_4 features by default (index-based)."""
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        # we'll use features up to conv4_4 (index 28 in torchvision's vgg19.features)
        self.slice = nn.Sequential(*[vgg[x] for x in range(28)])
        if not requires_grad:
            for p in self.slice.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.slice(x)

# RaGAN BCE-style losses (logit-based)

def ragan_discriminator_loss(C_real, C_fake):
    # C_real, C_fake: logits
    D_ra_real = torch.sigmoid(C_real - C_fake.mean())
    D_ra_fake = torch.sigmoid(C_fake - C_real.mean())
    loss_real = F.binary_cross_entropy(D_ra_real, torch.ones_like(D_ra_real))
    loss_fake = F.binary_cross_entropy(D_ra_fake, torch.zeros_like(D_ra_fake))
    return (loss_real + loss_fake) * 0.5


def ragan_generator_loss(C_real, C_fake):
    D_ra_real = torch.sigmoid(C_real - C_fake.mean())
    D_ra_fake = torch.sigmoid(C_fake - C_real.mean())
    loss_real = F.binary_cross_entropy(D_ra_real, torch.zeros_like(D_ra_real))
    loss_fake = F.binary_cross_entropy(D_ra_fake, torch.ones_like(D_ra_fake))
    return (loss_real + loss_fake) * 0.5

# ----------------------------- Utilities -------------------------------------

def init_weights(net, scale=0.1):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

# interpolation utility (weight-space)

def interpolate_models(model_a, model_b, alpha=0.5):
    """Return a new state_dict that is (1-alpha)*A + alpha*B"""
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_new = OrderedDict()
    for k in sd_a.keys():
        sd_new[k] = (1.0 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
    return sd_new

# ----------------------------- Dataset ---------------------------------------
class LRDHDataset(Dataset):n    
    def __init__(self, hr_dir, lr_dir=None, patch_size=128, scale=4, transforms_hr=None):
        # If lr_dir is None we generate bicubic LR on the fly from HR
        self.hr_paths = sorted([os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if f.lower().endswith(('png','jpg','jpeg'))])
        self.lr_paths = None
        if lr_dir:
            self.lr_paths = sorted([os.path.join(lr_dir, f) for f in os.listdir(lr_dir) if f.lower().endswith(('png','jpg','jpeg'))])
            assert len(self.hr_paths) == len(self.lr_paths), "LR and HR counts mismatch"
        self.patch_size = patch_size
        self.scale = scale
        self.transforms_hr = transforms_hr

    def __len__(self):
        return len(self.hr_paths)

    def __getitem__(self, idx):
        hr = Image.open(self.hr_paths[idx]).convert('RGB')
        # random crop on HR
        w, h = hr.size
        th = self.patch_size
        if w < th or h < th:
            hr = hr.resize((max(th, w), max(th, h)), Image.BICUBIC)
            w, h = hr.size
        x = random.randint(0, w - th)
        y = random.randint(0, h - th)
        hr_crop = hr.crop((x, y, x + th, y + th))

        if self.lr_paths is None:
            # generate LR by bicubic downsample
            lr = hr_crop.resize((th // self.scale, th // self.scale), Image.BICUBIC)
            lr = lr.resize((th, th), Image.BICUBIC)
        else:
            lr = Image.open(self.lr_paths[idx]).convert('RGB')
            # optionally crop lr similarly (assuming aligned)

        # transforms -> tensor
        to_tensor = transforms.ToTensor()
        hr_t = to_tensor(hr_crop)
        lr_t = to_tensor(lr)
        return lr_t, hr_t

# ----------------------------- Training loop ---------------------------------

def train_esrgan(
    hr_dir,
    lr_dir=None,
    out_dir='outputs',
    device='cuda',
    pretrain_iters=10000,
    adv_iters=30000,
    batch_size=8,
    lr=2e-4,
    scale=4,
):
    os.makedirs(out_dir, exist_ok=True)

    # models
    G = RRDBNet(nb=23).to(device)
    D = Discriminator().to(device)
    init_weights(G)
    init_weights(D)

    G_ema = EMA(G, decay=0.999)

    # losses & optimizers
    vgg = VGGFeatureExtractor().to(device)
    l1_loss = nn.L1Loss()
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.999))

    dataset = LRDHDataset(hr_dir, lr_dir, patch_size=128, scale=scale)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    it = 0
    # ---------- Stage 1: Pretrain G with L1 ----------
    print('Stage 1: Pretraining G (L1)')
    while it < pretrain_iters:
        for lr_t, hr_t in loader:
            lr_t = lr_t.to(device)
            hr_t = hr_t.to(device)
            opt_g.zero_grad()
            sr = G(lr_t)
            loss = l1_loss(sr, hr_t)
            loss.backward()
            opt_g.step()
            G_ema.update(G)

            if it % 100 == 0:
                print(f'Pretrain iter {it}/{pretrain_iters} L1: {loss.item():.4f}')
            it += 1
            if it >= pretrain_iters:
                break

    # save pretrained G
    torch.save({'G': G.state_dict()}, os.path.join(out_dir, 'G_pretrain.pth'))

    # ---------- Stage 2: Adversarial fine-tune ----------
    print('Stage 2: Adversarial finetune (VGG + RaGAN)')
    it = 0
    while it < adv_iters:
        for lr_t, hr_t in loader:
            lr_t = lr_t.to(device)
            hr_t = hr_t.to(device)
            # generator forward
            sr = G(lr_t)

            # discriminator update
            opt_d.zero_grad()
            C_real = D(hr_t)
            C_fake = D(sr.detach())
            loss_d = ragan_discriminator_loss(C_real, C_fake)
            loss_d.backward()
            opt_d.step()

            # generator update
            opt_g.zero_grad()
            C_real = D(hr_t)
            C_fake = D(sr)
            loss_g_gan = ragan_generator_loss(C_real, C_fake)
            # perceptual
            feat_sr = vgg(sr)
            feat_hr = vgg(hr_t)
            loss_feat = l1_loss(feat_sr, feat_hr)
            # pixel loss (small)
            loss_pixel = l1_loss(sr, hr_t)

            # total
            lambda_feat = 1.0
            lambda_gan = 5e-3
            lambda_pixel = 1e-2
            loss_g = lambda_feat * loss_feat + lambda_gan * loss_gan + lambda_pixel * loss_pixel
            loss_g.backward()
            opt_g.step()
            G_ema.update(G)

            if it % 100 == 0:
                print(f'Adv iter {it}/{adv_iters} D: {loss_d.item():.4f} G_gan: {loss_g_gan.item():.4f} feat: {loss_feat.item():.4f}')

            if it % 1000 == 0:
                # save checkpoint
                torch.save({'G': G.state_dict(), 'D': D.state_dict()}, os.path.join(out_dir, f'ckpt_{it}.pth'))
                # save ema-weighted inference sample
                G_ema.apply_shadow(G)
                with torch.no_grad():
                    sr_vis = G(lr_t[:1])
                    save_image(sr_vis.clamp(0,1), os.path.join(out_dir, f'sample_{it}.png'))
                # restore (we mutated G with EMA values) - reload latest state
                torch.load(os.path.join(out_dir, f'ckpt_{it}.pth')) if os.path.exists(os.path.join(out_dir, f'ckpt_{it}.pth')) else None

            it += 1
            if it >= adv_iters:
                break

    # final save
    torch.save({'G': G.state_dict(), 'D': D.state_dict()}, os.path.join(out_dir, 'G_D_final.pth'))
    print('Training finished')

# ----------------------------- Inference ------------------------------------

def upscale_image(model, img_path, out_path, device='cuda'):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    to_tensor = transforms.ToTensor()
    inp = to_tensor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)
    save_image(out.clamp(0,1), out_path)

# ----------------------------- Example usage --------------------------------

if __name__ == '__main__':
    # quick test: build models and run a dummy tensor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    G = RRDBNet(nb=3).to(device)
    D = Discriminator().to(device)
    x = torch.randn(1,3,32,32).to(device)
    y = G(x)
    print('G output shape', y.shape)
    z = D(y)
    print('D output shape', z.shape)

    # To train: uncomment below and provide hr_dir path
    # train_esrgan(hr_dir='/path/to/DIV2K/train_HR', out_dir='./esrgan_out', device=device)
