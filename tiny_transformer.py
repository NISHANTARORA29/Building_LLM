# tiny_transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.0):
        super().__init__()
        self.d_k = d_k
        self.drop = nn.Dropout(dropout)
    def forward(self, Q, K, V, mask=None):
        # Q,K,V: [B, h, T, d_k]
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # [B,h,Tq,Tk]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.drop(F.softmax(scores, dim=-1))
        return attn @ V, attn  # [B,h,Tq,d_k]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.0):
        super().__init__()
        assert d_model % heads == 0
        self.h = heads
        self.d_k = d_model // heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention(self.d_k, dropout)
        self.drop = nn.Dropout(dropout)

    def split(self, x):
        B, T, D = x.size()
        x = x.view(B, T, self.h, self.d_k).transpose(1, 2)  # [B,h,T,d_k]
        return x

    def combine(self, x):
        B, h, T, d_k = x.size()
        x = x.transpose(1, 2).contiguous().view(B, T, h * d_k)  # [B,T,D]
        return x

    def forward(self, q, k, v, mask=None):
        Q = self.split(self.Wq(q))
        K = self.split(self.Wk(k))
        V = self.split(self.Wv(v))
        if mask is not None:
            # expand mask to [B,1,Tq,Tk]
            mask = mask.unsqueeze(1)
        out, _ = self.attn(Q, K, V, mask)
        out = self.combine(out)
        return self.proj(self.drop(out))

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x): return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.drop(self.attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, tgt_mask=None, mem_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))     # masked self-attn
        x = self.norm2(x + self.drop(self.cross_attn(x, mem, mem, mem_mask))) # encoder-decoder
        x = self.norm3(x + self.drop(self.ff(x)))
        return x

class TinyTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, heads=4, d_ff=512, layers=2, max_len=512):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_model)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff) for _ in range(layers)])
        self.proj = nn.Linear(d_model, tgt_vocab)

    def make_src_mask(self, src_pad_mask):
        # src_pad_mask: [B,Ts] -> [B,1,Ts,Ts] for attention
        return src_pad_mask.unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt_pad_mask):
        B, T = tgt_pad_mask.size()
        # causal mask
        causal = torch.tril(torch.ones(T, T, device=tgt_pad_mask.device)).bool()
        pad = tgt_pad_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        # allow attend only to non-pad & past tokens
        return pad & causal  # broadcast to [B,1,T,T] in attention

    def forward(self, src, tgt_inp, src_pad_mask, tgt_pad_mask):
        # Embeddings + positions
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt_inp))

        # masks
        src_mask = self.make_src_mask(src_pad_mask)    # [B,1,1,Ts]
        tgt_mask = self.make_tgt_mask(tgt_pad_mask)    # [B,1,Tt,Tt]
        mem_mask = src_mask                             # [B,1,1,Ts]

        # encoder
        mem = src
        for l in self.enc_layers:
            mem = l(mem, src_mask)

        # decoder
        x = tgt
        for l in self.dec_layers:
            x = l(x, mem, tgt_mask, mem_mask)

        return self.proj(x)  # [B,Tt,V_tgt]

# ---- Tiny demo ----
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Vsrc, Vtgt = 5000, 6000
    B, Ts, Tt = 2, 7, 6

    model = TinyTransformer(Vsrc, Vtgt).to(device).train()
    src = torch.randint(4, Vsrc, (B, Ts), device=device)
    tgt_inp = torch.randint(4, Vtgt, (B, Tt), device=device)
    # masks: 1 = keep, 0 = pad
    src_len = torch.tensor([7, 5], device=device)
    tgt_len = torch.tensor([6, 4], device=device)
    src_mask = (torch.arange(Ts, device=device).unsqueeze(0) < src_len.unsqueeze(1)).int()
    tgt_mask = (torch.arange(Tt, device=device).unsqueeze(0) < tgt_len.unsqueeze(1)).int()

    logits = model(src, tgt_inp, src_mask, tgt_mask)
    print(logits.shape)

    # cross-entropy with teacher forcing
    tgt_gold = torch.randint(0, Vtgt, (B, Tt), device=device)
    loss = F.cross_entropy(logits.view(-1, Vtgt), tgt_gold.view(-1), ignore_index=0)
    loss.backward()
    print("loss:", float(loss))
