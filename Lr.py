import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 1. Dummy dataset (replace with real one)
# -----------------------------
X = torch.randn(2000, 20)   # 2000 samples, 20 features
y = (torch.sum(X, dim=1) > 0).long()  # binary labels

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -----------------------------
# 2. Simple model (replace with your model)
# -----------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden=64, output_dim=2):
        super(SimpleNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )
    def forward(self, x):
        return self.net(x)

model = SimpleNN(20)

# -----------------------------
# 3. Loss + Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-8)  # start super low

# -----------------------------
# 4. LR Finder logic
# -----------------------------
num_iters = len(dataloader) - 1
start_lr = 1e-8
end_lr = 10
beta = 0.98  # smoothing factor for loss

# compute multiplicative factor to increase LR exponentially
lr_mult = (end_lr / start_lr) ** (1/num_iters)

lr = start_lr
optimizer.param_groups[0]['lr'] = lr

avg_loss = 0.0
best_loss = float('inf')
losses = []
lrs = []

for batch_num, (inputs, targets) in enumerate(dataloader):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # smooth loss for stability
    avg_loss = beta * avg_loss + (1 - beta) * loss.item()
    smoothed_loss = avg_loss / (1 - beta**(batch_num+1))

    # record LR & loss
    lrs.append(lr)
    losses.append(smoothed_loss)

    # stop if loss explodes
    if batch_num > 1 and smoothed_loss > 4 * best_loss:
        break
    if smoothed_loss < best_loss or batch_num == 0:
        best_loss = smoothed_loss

    # backward + step
    loss.backward()
    optimizer.step()

    # update LR
    lr *= lr_mult
    optimizer.param_groups[0]['lr'] = lr

# -----------------------------
# 5. Plot LR vs Loss
# -----------------------------
plt.plot(lrs, losses)
plt.xscale("log")
plt.xlabel("Learning Rate (log scale)")
plt.ylabel("Loss")
plt.title("Learning Rate Finder")
plt.show()
