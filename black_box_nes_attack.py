# black_box_nes_attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Dummy classifier to demo the API (replace with your black-box model wrapper) ----
class TinyCNN(nn.Module):
    def __init__(self, in_ch=1, n_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        h = self.conv(x).flatten(1)
        return self.fc(h)

# Pretend we only have this API (no .backward(), no params):
@torch.no_grad()
def predict_logits(x):
    # in real life: call your remote API / production model
    return MODEL(x)

# ---- NES Attack ----
@torch.no_grad()
def nes_attack(
    x0, y_true=None, y_target=None, steps=60, step_size=0.01, eps=0.3,
    sigma=0.001, samples=60, clip_min=0.0, clip_max=1.0
):
    """
    x0: input [1,C,H,W] in [0,1]
    y_true: untargeted if set (push away from true class)
    y_target: targeted if set (pull towards target class)
    """
    x = x0.clone()
    C = 1 if y_target is not None else 0
    assert (y_true is None) ^ (y_target is not None), "Pick targeted OR untargeted."

    for t in range(steps):
        # Antithetic sampling for lower variance
        u = torch.randn(samples//2, *x.shape[1:], device=x.device)
        u = torch.cat([u, -u], dim=0)

        # Query model for score estimates
        xs = x + sigma * u
        xs = xs.clamp(clip_min, clip_max)

        logits = []
        B = 32  # batch queries to not blow memory / API rate
        for i in range(0, xs.size(0), B):
            logits.append(predict_logits(xs[i:i+B]))
        logits = torch.cat(logits, 0)  # [samples, num_classes]

        if y_target is not None:
            # maximize log prob of target
            scores = F.log_softmax(logits, dim=-1)[:, y_target]
        else:
            # minimize log prob of true class (i.e., push away)
            scores = -F.log_softmax(logits, dim=-1)[:, y_true]

        # NES gradient estimate: E[(f(x+σu) * u)] / σ
        g = (scores.view(-1, *([1]*len(x.shape[1:]))) * u).mean(dim=0) / sigma

        # PGD step in sign of gradient
        x = x + step_size * g.sign()
        # project to epsilon-ball (L_inf)
        x = torch.max(torch.min(x, x0 + eps), x0 - eps)
        x = x.clamp(clip_min, clip_max)

        # optional: early stop if success
        pred = predict_logits(x).argmax(dim=-1).item()
        if (y_target is not None and pred == int(y_target)) or \
           (y_true is not None and pred != int(y_true)):
            break

    return x

# ---- Demo usage ----
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # toy model + random "image"
    MODEL = TinyCNN().to(device).eval()
    x0 = torch.rand(1, 1, 28, 28, device=device)  # pretend MNIST-like
    y = predict_logits(x0).argmax(dim=-1).item()

    # Untargeted attack: move off y
    x_adv = nes_attack(x0, y_true=y, steps=80, step_size=0.01, eps=0.3,
                       sigma=0.001, samples=60)

    print("orig pred:", y, "adv pred:", predict_logits(x_adv).argmax(-1).item())
