import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np

# Reproducibility
torch.manual_seed(0)

# Dummy data
X = torch.randn(100, 10)
Y = (torch.sum(X, dim=1) > 0).long()

class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 50)
        self.bn1 = nn.BatchNorm1d(50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()

use_adam = True
if use_adam:
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
else:
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

early_stopping_patience = 10
best_loss = np.inf
patience_counter = 0
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X)
    loss = criterion(outputs, Y)

    loss.backward()
    optimizer.step()
    scheduler.step()

    current_lr = scheduler.get_last_lr()[0]
    preds = outputs.argmax(dim=1)
    acc = (preds == Y).float().mean().item()

    print(f'Epoch {epoch+1}/{num_epochs} | Loss: {loss.item():.4f} | '
          f'Acc: {acc:.3f} | LR: {current_lr:.5f}')

    if loss.item() < best_loss:
        best_loss = loss.item()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
