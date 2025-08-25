# rnn_compare.py
# Train & compare GRU vs LSTM vs BiLSTM for next-token prediction (word-level).
# Python 3.10+, PyTorch 2.x

import math
import random
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# 1) Minimal corpus & tokenizer
# ----------------------------
# Replace this with your domain corpus (e.g., medical transcripts).
corpus = """
patient has high blood pressure and diabetes
patient has high blood sugar monitor daily
patient shows elevated heart rate and chest pain
recommend lifestyle changes and medication adherence
schedule follow up visit for blood pressure management
"""

def build_vocab(text: str) -> Tuple[dict, dict, List[int]]:
    tokens = text.strip().lower().split()
    vocab = sorted(set(tokens))
    stoi = {w:i for i,w in enumerate(vocab)}
    itos = {i:w for w,i in stoi.items()}
    ids = [stoi[w] for w in tokens]
    return stoi, itos, ids

stoi, itos, ids = build_vocab(corpus)
vocab_size = len(stoi)

# ----------------------------
# 2) Sequence dataset
# ----------------------------
class SeqDataset(Dataset):
    def __init__(self, ids: List[int], seq_len: int = 8):
        self.ids = ids
        self.seq_len = seq_len
        self.items = []
        for i in range(len(ids) - seq_len):
            x = ids[i:i+seq_len]
            y = ids[i+1:i+seq_len+1]  # next-token targets (shifted by 1)
            self.items.append((x, y))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

seq_len = 8
dataset = SeqDataset(ids, seq_len=seq_len)
loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

# ----------------------------
# 3) Unified RNN model wrapper
# ----------------------------
class RNNLanguageModel(nn.Module):
    """
    Embedding -> {LSTM | GRU | BiLSTM} -> Linear decoder to vocab.
    Matches the math:
      - recurrent update: (h_t, c_t) or h_t via rnn(x_t, h_{t-1})
      - decoder: y_t = softmax(W_y h_t + b_y)
    """
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int = 128,
                 hidden_dim: int = 256,
                 rnn_type: str = "lstm",   # "lstm" | "gru" | "bilstm"
                 num_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        assert rnn_type in {"lstm", "gru", "bilstm"}
        self.rnn_type = rnn_type

        self.embed = nn.Embedding(vocab_size, emb_dim)

        bidirectional = (rnn_type == "bilstm")
        rnn_hidden_dim = hidden_dim

        if rnn_type in {"lstm", "bilstm"}:
            self.rnn = nn.LSTM(
                input_size=emb_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=emb_dim,
                hidden_size=rnn_hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=bidirectional,
                batch_first=True,
            )

        out_dim = rnn_hidden_dim * (2 if bidirectional else 1)
        self.decoder = nn.Linear(out_dim, vocab_size)

        # Tie weights (optional): decoder weight shares with embedding
        # Helps small corpora generalize; comment out if undesired.
        self.decoder.weight = self.embed.weight

    def forward(self, x, hidden=None):
        """
        x: (B, T) token ids
        hidden: optional initial hidden state (and cell for LSTM)
        returns logits: (B, T, V), hidden
        """
        emb = self.embed(x)  # (B, T, E)
        out, hidden = self.rnn(emb, hidden)  # out: (B, T, H or 2H)
        logits = self.decoder(out)  # (B, T, V)
        return logits, hidden

# ----------------------------
# 4) Training utilities
# ----------------------------
def sequence_cross_entropy(logits, targets):
    """
    logits: (B, T, V)
    targets: (B, T)
    """
    B, T, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(B*T, V),
        targets.reshape(B*T),
    )
    return loss

@torch.no_grad()
def evaluate_perplexity(model, loader, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = sequence_cross_entropy(logits, y)
        total_loss += loss.item() * x.numel()
        total_tokens += x.numel()
    mean_loss = total_loss / total_tokens
    ppl = math.exp(mean_loss)
    return mean_loss, ppl

@torch.no_grad()
def generate(model, prefix: List[str], length: int, stoi: dict, itos: dict, device, temperature: float = 1.0):
    """
    Greedy/temperature sampling from the language model.
    Uses the RNN recurrently step-by-step to highlight h_{t} updates.
    """
    model.eval()
    tokens = [stoi[w] for w in prefix]
    hidden = None

    for _ in range(length):
        x = torch.tensor(tokens[-seq_len:], dtype=torch.long).unsqueeze(0).to(device)
        logits, hidden = model(x, hidden=None)  # re-feed context window for simplicity/stability
        next_logits = logits[0, -1, :] / max(1e-5, temperature)
        probs = nn.functional.softmax(next_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1).item()
        tokens.append(idx)

    return [itos[i] for i in tokens]

# ----------------------------
# 5) Train all three models
# ----------------------------
def train_one(model_name: str, device="cuda" if torch.cuda.is_available() else "cpu"):
    print(f"\n=== Training {model_name.upper()} ===")
    torch.manual_seed(42)
    random.seed(42)

    model = RNNLanguageModel(
        vocab_size=vocab_size,
        emb_dim=128,
        hidden_dim=256,
        rnn_type=model_name,   # "lstm", "gru", "bilstm"
        num_layers=1,
        dropout=0.1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)

    # Cosine LR over small epochs for demo
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = sequence_cross_entropy(logits, y)
            loss.backward()

            # Grad clip helps stabilize long sequences (ties to vanishing/exploding gradients theory)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        val_loss, val_ppl = evaluate_perplexity(model, loader, device)
        print(f"Epoch {epoch:02d} | train_loss={total_loss/len(loader):.4f} | val_ppl={val_ppl:.2f}")

    # Sample a short sequence to compare qualitative behavior:
    seed = ["patient", "has", "high", "blood"]
    out = generate(model, seed, length=10, stoi=stoi, itos=itos, device=device, temperature=0.8)
    print(f"Seed:  {' '.join(seed)}")
    print(f"Gen :  {' '.join(out)}")

if __name__ == "__main__":
    # Flip these on/off as you like to compare quickly on the same data/loader.
    for kind in ["lstm", "gru", "bilstm"]:
        train_one(kind)
