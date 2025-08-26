# minimal, but correct: cWGAN-GP with projection D (PyTorch)
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

class G(nn.Module):
    def __init__(self, z_dim=128, n_classes=0, ch=64, img_ch=3):
        super().__init__()
        self.embed = nn.Embedding(n_classes, z_dim) if n_classes>0 else None
        in_dim = z_dim
        self.fc = nn.Linear(in_dim, 4*4*ch*8)
        self.block1 = nn.Sequential(nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1), nn.BatchNorm2d(ch*4), nn.ReLU(True))
        self.block2 = nn.Sequential(nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1), nn.BatchNorm2d(ch*2), nn.ReLU(True))
        self.block3 = nn.Sequential(nn.ConvTranspose2d(ch*2, ch,   4, 2, 1), nn.BatchNorm2d(ch),   nn.ReLU(True))
        self.to_rgb = nn.Conv2d(ch, img_ch, 3, 1, 1)

    def forward(self, z, y=None):
        if self.embed is not None and y is not None:
            z = z + self.embed(y)  # simple conditioning; for BigGAN use conditional BN
        x = self.fc(z).view(z.size(0), -1, 4, 4)
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = torch.tanh(self.to_rgb(x))
        return x

class ProjectionD(nn.Module):
    def __init__(self, n_classes=0, ch=64, img_ch=3):
        super().__init__()
        def snc(in_c,out_c,ks,st,pd): return spectral_norm(nn.Conv2d(in_c,out_c,ks,st,pd))
        self.c1 = snc(img_ch, ch,   3,1,1)
        self.c2 = snc(ch,     ch,   4,2,1)
        self.c3 = snc(ch,     ch*2, 3,1,1)
        self.c4 = snc(ch*2,   ch*2, 4,2,1)
        self.c5 = snc(ch*2,   ch*4, 3,1,1)
        self.c6 = snc(ch*4,   ch*4, 4,2,1)
        self.lin = spectral_norm(nn.Linear(ch*4*8*8, 1))  # for 64x64 inputs
        self.embed = spectral_norm(nn.Embedding(n_classes, ch*4*8*8)) if n_classes>0 else None

    def forward(self, x, y=None):
        h = F.leaky_relu(self.c1(x), 0.2); h = F.leaky_relu(self.c2(h), 0.2)
        h = F.leaky_relu(self.c3(h), 0.2); h = F.leaky_relu(self.c4(h), 0.2)
        h = F.leaky_relu(self.c5(h), 0.2); h = F.leaky_relu(self.c6(h), 0.2)
        h = h.view(h.size(0), -1)
        out = self.lin(h).squeeze(1)  # critic score
        if self.embed is not None and y is not None:
            out = out + torch.sum(self.embed(y) * h, dim=1)  # projection term
        return out

def gradient_penalty(D, real, fake, y=None, gp_lambda=10.0):
    bsz = real.size(0)
    eps = torch.rand(bsz, 1, 1, 1, device=real.device)
    hat = eps*real + (1-eps)*fake
    hat.requires_grad_(True)
    d_hat = D(hat, y)
    grads = torch.autograd.grad(d_hat.sum(), hat, create_graph=True)[0]
    gp = ((grads.view(bsz, -1).norm(2, dim=1) - 1.0)**2).mean() * gp_lambda
    return gp

# training loop sketch
# for each batch (x_real, y):
#   for t in range(n_critic):
#       z ~ N(0,I); x_fake = G(z,y)
#       d_loss = -(D(x_real,y).mean() - D(x_fake.detach(),y).mean()) + gradient_penalty(...)
#       optD.zero_grad(); d_loss.backward(); optD.step()
#   z ~ N(0,I); x_fake = G(z,y)
#   g_loss = -D(x_fake,y).mean()
#   optG.zero_grad(); g_loss.backward(); optG.step()
#   use EMA(G) for eval; compute FID periodically.
