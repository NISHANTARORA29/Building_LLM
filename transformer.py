# Implementing a Transformer (from scratch) in PyTorch (no HuggingFace)
# This cell defines a full Transformer (encoder-decoder), with Multi-Head Attention,
# Positional Encoding, Feed-Forward, and an example forward pass + tiny training loop on a toy copy task.
# You can run this cell to test the model. It's self-contained (requires PyTorch).
# If running locally, ensure `torch` is installed (pip install torch).
# Author: ChatGPT (GPT-5 Thinking mini) — concise, well-commented, production-quality code.

import math, copy, time
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------- Utilities ---------------------------
def clones(module, N):
    "Produce N identical layers (deepcopy)"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# ----------------------- Positional Encoding -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :].to(x.device)
        return self.dropout(x)

# ----------------------- Scaled Dot-Product Attention -----------------------
def attention(query, key, value, mask: Optional[torch.Tensor] = None, dropout: Optional[nn.Module] = None):
    "Compute Scaled Dot Product Attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_q, seq_k)
    if mask is not None:
        # mask: broadcastable to scores shape; typically contains True at positions to mask (or 0/1 depending)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

# ----------------------- Multi-Head Attention -----------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, h: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = clones(nn.Linear(d_model, d_model), 4)  # q, k, v, out
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask: Optional[torch.Tensor] = None):
        if mask is not None:
            # Same mask applied to all h heads; mask shape should be broadcastable
            mask = mask.unsqueeze(1)  # (batch, 1, 1/seq_q, seq_k)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linear_layers[:3], (query, key, value))
        ]  # each is (batch, h, seq_len, d_k)

        # 2) Apply attention on all the projected vectors in batch
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # (batch, seq, d_model)
        return self.linear_layers[3](x)

# ----------------------- Position-wise Feed-Forward -----------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

# ----------------------- Encoder & Decoder Layers -----------------------
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, feed_forward: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # Self-attention sublayer
        x2 = self.norm1(x)
        sa = self.self_attn(x2, x2, x2, mask=src_mask)
        x = x + self.dropout1(sa)
        # Feed-forward sublayer
        x2 = self.norm2(x)
        ff = self.feed_forward(x2)
        x = x + self.dropout2(ff)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, self_attn: nn.Module, src_attn: nn.Module, feed_forward: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        # Masked self-attention (target)
        x2 = self.norm1(x)
        sa = self.self_attn(x2, x2, x2, mask=tgt_mask)
        x = x + self.dropout1(sa)
        # Source attention (attend to encoder output)
        x2 = self.norm2(x)
        src_att = self.src_attn(x2, memory, memory, mask=src_mask)
        x = x + self.dropout2(src_att)
        # Feed-forward
        x2 = self.norm3(x)
        ff = self.feed_forward(x2)
        x = x + self.dropout3(ff)
        return x

# ----------------------- Encoder & Decoder Stacks -----------------------
class Encoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.self_attn.linear_layers[0].in_features)  # d_model

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, layer: nn.Module, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.self_attn.linear_layers[0].in_features)  # d_model

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# ----------------------- Full Transformer -----------------------
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, N_enc=6, N_dec=6, h=8, d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()
        self.src_embed = nn.Sequential(nn.Embedding(src_vocab, d_model), PositionalEncoding(d_model, max_len, dropout))
        self.tgt_embed = nn.Sequential(nn.Embedding(tgt_vocab, d_model), PositionalEncoding(d_model, max_len, dropout))

        attn = MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        enc_layer = EncoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(ff), dropout)
        dec_layer = DecoderLayer(d_model, copy.deepcopy(attn), copy.deepcopy(attn), copy.deepcopy(ff), dropout)

        self.encoder = Encoder(enc_layer, N_enc)
        self.decoder = Decoder(dec_layer, N_dec)
        self.out = nn.Linear(d_model, tgt_vocab)

        # weight init (like original paper's suggestion)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src: (batch, src_seq), tgt: (batch, tgt_seq)
        enc = self.encoder(self.src_embed(src), src_mask)
        dec = self.decoder(self.tgt_embed(tgt), enc, src_mask, tgt_mask)
        return self.out(dec)  # (batch, tgt_seq, tgt_vocab)

# ----------------------- Masks -----------------------
def make_src_mask(src, pad_idx=0):
    # src: (batch, seq)
    # mask: (batch, 1, 1, seq) or (batch, 1, seq) broadcastable
    return (src != pad_idx).unsqueeze(-2)  # True where not pad

def make_tgt_mask(tgt, pad_idx=0):
    # create mask to prevent attending to future tokens + mask pads
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(-2)  # (batch,1,seq)
    seq_len = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1).bool()  # True above diag
    # invert subsequent_mask to have True where allowed, False where masked
    subsequent_mask = ~subsequent_mask  # True on diag and below
    tgt_mask = tgt_pad_mask & subsequent_mask  # broadcast to (batch,1,seq,seq) if needed
    return tgt_mask

# ----------------------- Toy training: Copy task -----------------------
def run_toy_copy_task(device='cpu'):
    torch.manual_seed(0)
    # hyperparams for toy model (small d_model for speed)
    SRC_VOCAB = 11  # tokens 0..10 (0 reserved for PAD)
    TGT_VOCAB = 11
    d_model = 64
    model = Transformer(SRC_VOCAB, TGT_VOCAB, d_model=d_model, N_enc=2, N_dec=2, h=4, d_ff=128, dropout=0.1, max_len=50).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Generate random sequences and train to copy them (teacher forcing)
    BATCH = 32
    SEQ_LEN = 10
    steps = 200  # keep small for demo

    for step in range(steps):
        src = torch.randint(1, SRC_VOCAB, (BATCH, SEQ_LEN), device=device)  # avoid PAD
        tgt_input = torch.cat([torch.full((BATCH,1), 2, device=device), src[:, :-1]], dim=1)  # BOS token = 2 (arbitrary)
        tgt_output = src  # we want to copy src to tgt
        src_mask = make_src_mask(src, pad_idx=0)
        tgt_mask = make_tgt_mask(tgt_input, pad_idx=0)

        logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)  # (B, seq, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step==steps-1:
            print(f"Step {step}/{steps} | Loss: {loss.item():.4f}")

    # Quick test
    model.eval()
    with torch.no_grad():
        src = torch.randint(1, SRC_VOCAB, (2, SEQ_LEN), device=device)
        tgt_input = torch.full((2, 1), 2, device=device)  # start with BOS
        outputs = []
        for i in range(SEQ_LEN):
            tgt_mask = make_tgt_mask(tgt_input, pad_idx=0)
            logits = model(src, tgt_input, src_mask=make_src_mask(src), tgt_mask=tgt_mask)  # (B, seq, vocab)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt_input = torch.cat([tgt_input, next_token], dim=1)
        print("Source:\n", src)
        print("Predicted copy:\n", tgt_input[:,1:])

# Run toy task in CPU (change to 'cuda' if GPU available)
run_toy_copy_task(device='cpu')
