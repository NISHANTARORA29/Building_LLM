#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hackna-GPT — a from-scratch, pure-ML (no external APIs) miniature LLM with optional local RAG
=============================================================================================

This single Python file gives you:
  • A tiny byte-pair encoder (BPE) tokenizer trainer + encoder/decoder
  • A Transformer decoder-only language model (causal LM) in PyTorch
  • Training loop with AdamW, cosine LR schedule, gradient clipping, AMP
  • Text generation (greedy / top-k / nucleus sampling)
  • Optional local Retrieval-Augmented Generation (RAG) using a lightweight
    in-memory vector store built from the model's token embeddings (no FAISS)
  • Simple CLI to train, index docs, and chat — all offline

It will NOT magically “answer everything” (that requires huge data/compute),
but this is a tough, production-style scaffold you can scale up with more data,
bigger models, and better retrieval.

Quickstart
----------
1) Put your raw text (*.txt) into a folder, e.g. `./data/`.
2) Train a tokenizer on your corpus (learns merges):
       python hackna_gpt.py train_tokenizer --data_dir ./data --vocab_size 30000
3) Train the LM:
       python hackna_gpt.py train_lm --data_dir ./data --out_dir ./runs/hackna --vocab_path ./runs/hackna/tokenizer.json \
           --dim 512 --n_layers 8 --n_heads 8 --seq_len 512 --batch_size 16 --steps 20000
4) Build a local RAG index from your reference docs:
       python hackna_gpt.py build_index --corpus_dir ./kb --vocab_path ./runs/hackna/tokenizer.json \
           --ckpt ./runs/hackna/ckpt_latest.pt --index_path ./runs/hackna/index.npz
5) Chat with (optionally) RAG:
       python hackna_gpt.py chat --vocab_path ./runs/hackna/tokenizer.json --ckpt ./runs/hackna/ckpt_latest.pt \
           --index_path ./runs/hackna/index.npz --max_new_tokens 200 --top_p 0.9 --temperature 0.8

Dependencies
------------
• Python 3.9+
• PyTorch 2.x (CUDA if available)

Design notes
------------
• Tokenizer: Minimal BPE (byte-level init) to avoid external deps.
• Model: Decoder-only Transformer with Pre-LN, GELU MLP, rotary pos-embs optional.
• RAG: Mean-pooled token embedding vectors for passages; cosine similarity.
• Files: Checkpoints and tokenizer saved in `out_dir`.

"""
from __future__ import annotations
import os
import io
import re
import gc
import math
import json
import time
import glob
import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def detect_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------
# Tiny Byte-Pair Encoding (BPE)
# ------------------------------
class BPETokenizer:
    """Minimal byte-level BPE tokenizer.
    Saves/loads to JSON; supports training on raw text files.
    """
    def __init__(self):
        self.vocab_size = None
        self.merges: Dict[Tuple[str, str], int] = {}
        self.itos: List[str] = []
        self.stoi: Dict[str, int] = {}
        self.special_tokens = ["<pad>", "<bos>", "<eos>"]

    @staticmethod
    def _read_texts(paths: List[str]) -> str:
        buf = []
        for p in paths:
            with io.open(p, 'r', encoding='utf-8', errors='ignore') as f:
                buf.append(f.read())
        return "\n".join(buf)

    def train(self, data_paths: List[str], vocab_size: int = 30000, max_bytes: int = 50_000_000):
        raw = self._read_texts(data_paths)
        raw = raw.encode('utf-8')[:max_bytes].decode('utf-8', errors='ignore')
        # Start with byte-level tokens
        tokens = list(raw)
        # Build initial vocab as unique bytes
        vocab = set(tokens)
        vocab.update(self.special_tokens)

        def get_stats(tokens: List[str]) -> Dict[Tuple[str, str], int]:
            pairs = {}
            prev = tokens[0] if tokens else None
            for t in tokens[1:]:
                if prev is not None:
                    pairs[(prev, t)] = pairs.get((prev, t), 0) + 1
                prev = t
            return pairs

        def merge_tokens(tokens: List[str], pair: Tuple[str, str]) -> List[str]:
            a, b = pair
            i = 0
            new = []
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                    new.append(a + b)
                    i += 2
                else:
                    new.append(tokens[i])
                    i += 1
            return new

        merges = []
        while True:
            stats = get_stats(tokens)
            if not stats:
                break
            best = max(stats, key=stats.get)
            tokens = merge_tokens(tokens, best)
            merges.append(best)
            # Update vocab
            vocab.add(best[0] + best[1])
            if len(vocab) >= vocab_size:
                break
        # finalize vocab and indices
        self.itos = list(self.special_tokens) + sorted([v for v in vocab if v not in self.special_tokens], key=len)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.vocab_size = len(self.itos)

    def _tokenize(self, text: str) -> List[str]:
        # Greedy longest-match using learned merges (simple heuristic)
        i = 0
        out = []
        while i < len(text):
            j = min(len(text), i + 200)
            matched = None
            while j > i:
                piece = text[i:j]
                if piece in self.stoi:
                    matched = piece
                    break
                j -= 1
            if matched is None:
                # fallback to single char
                matched = text[i]
            out.append(matched)
            i += len(matched)
        return out

    def encode(self, text: str, add_special: bool = True) -> List[int]:
        toks = self._tokenize(text)
        ids = [self.stoi[t] if t in self.stoi else self.stoi.get("<unk>", 0) for t in toks]
        if add_special:
            return [self.stoi["<bos>"]] + ids + [self.stoi["<eos>"]]
        return ids

    def decode(self, ids: List[int]) -> str:
        pieces = [self.itos[i] for i in ids if 0 <= i < len(self.itos)]
        text = "".join(pieces)
        # strip special tokens artifacts
        text = text.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "")
        return text

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'itos': self.itos,
                'special_tokens': self.special_tokens,
            }, f)

    @staticmethod
    def load(path: str) -> 'BPETokenizer':
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        tok = BPETokenizer()
        tok.vocab_size = obj['vocab_size']
        tok.itos = obj['itos']
        tok.stoi = {s: i for i, s in enumerate(tok.itos)}
        tok.special_tokens = obj['special_tokens']
        return tok

# ------------------------------
# Dataset
# ------------------------------
class TextFolderDataset(Dataset):
    def __init__(self, data_dir: str, tokenizer: BPETokenizer, seq_len: int):
        self.files = sorted(glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True))
        assert len(self.files) > 0, f"No .txt files found under {data_dir}"
        self.tok = tokenizer
        self.seq_len = seq_len
        # Pre-encode all files (simple; for large corpora, stream instead)
        ids = []
        for fp in self.files:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            ids.extend(self.tok.encode(text))
        self.ids = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(1, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = self.ids[i:i+self.seq_len]
        y = self.ids[i+1:i+self.seq_len+1]
        if len(x) < self.seq_len:
            pad_id = self.tok.stoi["<pad>"]
            x = F.pad(x, (0, self.seq_len - len(x)), value=pad_id)
            y = F.pad(y, (0, self.seq_len - len(y)), value=pad_id)
        return x, y

# ------------------------------
# Transformer LM (decoder-only)
# ------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm_x

class RoPE:
    def __init__(self, dim, base=10000.0):
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.registered = False
        self.inv_freq = inv_freq
    def _build_cache(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum('i,j->ij', t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)  # interleave
        self.cos = emb.cos()[None, None, :, :]
        self.sin = emb.sin()[None, None, :, :]
        self.registered = True
        self.max_seq = seq_len
        self.device = device
    def rotate(self, q, k):
        # q,k: (B, H, T, D)
        if (not self.registered) or q.size(2) > self.max_seq or q.device != self.device:
            self._build_cache(q.size(2), q.device)
        cos, sin = self.cos[:, :, :q.size(2), :], self.sin[:, :, :q.size(2), :]
        def apply_rot(x):
            x1, x2 = x[..., :self.dim:2], x[..., 1:self.dim:2]
            x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x[..., :self.dim])
            xr = x[..., :self.dim] * cos + x_rot * sin
            if x.size(-1) > self.dim:
                xr = torch.cat([xr, x[..., self.dim:]], dim=-1)
            return xr
        return apply_rot(q), apply_rot(k)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_dim=None, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.o = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.rope = RoPE(self.d_head, base=10_000.0) if rope_dim else None

    def forward(self, x, mask):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, Dh)
        if self.rope is not None:
            q, k = self.rope.rotate(q, k)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att + mask  # causal mask
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.proj_drop(self.o(y))
        return y

class MLP(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expansion * d_model)
        self.fc2 = nn.Linear(expansion * d_model, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, rope_dim=d_model//n_heads, dropout=dropout)
        self.ln2 = RMSNorm(d_model)
        self.mlp = MLP(d_model, expansion=4, dropout=dropout)
    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=8, n_heads=8, seq_len=512, dropout=0.0):
        super().__init__()
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.seq_len
        # causal mask: (1, 1, T, T)
        mask = torch.full((1, 1, T, T), float('-inf'), device=idx.device)
        mask = torch.triu(mask, diagonal=1)
        tok = self.tok_emb(idx)  # (B,T,C)
        x = tok + self.pos_emb[:, :T, :]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=128, temperature=1.0, top_k=None, top_p=None, eos_id=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(1e-6, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cdf = probs.cumsum(dim=-1)
                cutoff = (cdf > top_p).float().argmax(dim=-1)
                for b in range(logits.size(0)):
                    k = cutoff[b].item()
                    sorted_logits[b, k+1:] = -float('inf')
                logits = torch.gather(sorted_logits, -1, torch.argsort(sorted_indices))
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_id), dim=1)
            if eos_id is not None and (next_id == eos_id).all():
                break
        return idx

    @torch.no_grad()
    def embed(self, idx):
        # returns mean pooled hidden states for sentence embedding
        h = self.ln_f(self.tok_emb(idx) + self.pos_emb[:, :idx.size(1), :])
        return h.mean(dim=1)

# ------------------------------
# Training loop
# ------------------------------
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    vocab_path: str
    seq_len: int = 512
    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    dropout: float = 0.0
    batch_size: int = 16
    steps: int = 10_000
    lr: float = 3e-4
    warmup_steps: int = 500
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    amp: bool = True
    seed: int = 1337


def save_ckpt(path, model, opt, step, tok_path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'opt': opt.state_dict(),
        'step': step,
        'vocab_path': tok_path,
        'config': {
            'vocab_size': model.head.out_features,
            'seq_len': model.seq_len,
        }
    }, path)


def train_lm(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = detect_device()
    print(f"Using device: {device}")

    tok = BPETokenizer.load(cfg.vocab_path)
    ds = TextFolderDataset(cfg.data_dir, tok, cfg.seq_len)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    model = TransformerLM(vocab_size=len(tok.itos), d_model=cfg.dim, n_layers=cfg.n_layers,
                          n_heads=cfg.n_heads, seq_len=cfg.seq_len, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == 'cuda')

    def lr_schedule(step):
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    global_step = 0
    model.train()
    t0 = time.time()

    while global_step < cfg.steps:
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == 'cuda'):
                logits = model(xb)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), ignore_index=tok.stoi["<pad>"])
            opt.param_groups[0]['lr'] = cfg.lr * lr_schedule(global_step)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()

            if global_step % 100 == 0:
                tok_s = f"step {global_step:6d}/{cfg.steps} | loss {loss.item():.3f} | lr {opt.param_groups[0]['lr']:.2e}"
                print(tok_s)
            if global_step % 1000 == 0:
                save_ckpt(os.path.join(cfg.out_dir, 'ckpt_latest.pt'), model, opt, global_step, cfg.vocab_path)
            global_step += 1
            if global_step >= cfg.steps:
                break
    save_ckpt(os.path.join(cfg.out_dir, 'ckpt_final.pt'), model, opt, global_step, cfg.vocab_path)
    print(f"Done. Trained {global_step} steps in {time.time()-t0:.1f}s")

# ------------------------------
# Lightweight local RAG index
# ------------------------------
class RAGIndex:
    def __init__(self, tokenizer: BPETokenizer, model: TransformerLM, device):
        self.tok = tokenizer
        self.model = model
        self.device = device
        self.passages: List[str] = []
        self.vecs: Optional[torch.Tensor] = None  # (N, D)

    @torch.no_grad()
    def add_passages(self, texts: List[str], batch_size: int = 8, max_len: int = 256):
        self.model.eval()
        vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            ids = [torch.tensor(self.tok.encode(t)[:max_len], dtype=torch.long) for t in batch]
            pad_id = self.tok.stoi["<pad>"]
            maxL = max(x.size(0) for x in ids)
            ids = [F.pad(x, (0, maxL - x.size(0)), value=pad_id) for x in ids]
            ids = torch.stack(ids, dim=0).to(self.device)
            v = self.model.embed(ids)
            vecs.append(F.normalize(v, dim=-1))
        new_vecs = torch.cat(vecs, dim=0)
        self.passages.extend(texts)
        self.vecs = new_vecs if self.vecs is None else torch.cat([self.vecs, new_vecs], dim=0)

    @torch.no_grad()
    def search(self, query: str, k: int = 5, max_len: int = 128) -> List[Tuple[float, str]]:
        if self.vecs is None or len(self.passages) == 0:
            return []
        q_ids = torch.tensor(self.tok.encode(query)[:max_len], dtype=torch.long, device=self.device).unsqueeze(0)
        q_vec = F.normalize(self.model.embed(q_ids), dim=-1)  # (1, D)
        sims = (self.vecs @ q_vec.t()).squeeze(1)  # (N,)
        vals, idx = torch.topk(sims, k=min(k, sims.numel()))
        return [(vals[i].item(), self.passages[idx[i]]) for i in range(vals.size(0))]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        npz = {
            'passages': self.passages,
            'vecs': self.vecs.detach().cpu().numpy() if self.vecs is not None else None,
        }
        import numpy as np
        np.savez_compressed(path, **npz)

    @staticmethod
    def load(path: str, tokenizer: BPETokenizer, model: TransformerLM, device) -> 'RAGIndex':
        import numpy as np
        data = np.load(path, allow_pickle=True)
        idx = RAGIndex(tokenizer, model, device)
        idx.passages = list(data['passages'])
        if data['vecs'] is not None:
            idx.vecs = torch.tensor(data['vecs'], dtype=torch.float32, device=device)
        else:
            idx.vecs = None
        return idx

# ------------------------------
# CLI & orchestration
# ------------------------------

def cmd_train_tokenizer(args):
    files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.txt'), recursive=True))
    assert files, f"No .txt files under {args.data_dir}"
    tok = BPETokenizer()
    tok.train(files, vocab_size=args.vocab_size, max_bytes=args.max_bytes)
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, 'tokenizer.json')
    tok.save(path)
    print(f"Saved tokenizer to {path} | vocab_size={tok.vocab_size}")


def cmd_train_lm(args):
    cfg = TrainConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        vocab_path=args.vocab_path,
        seq_len=args.seq_len,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        steps=args.steps,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        seed=args.seed,
    )
    os.makedirs(cfg.out_dir, exist_ok=True)
    train_lm(cfg)


def cmd_build_index(args):
    device = detect_device()
    tok = BPETokenizer.load(args.vocab_path)
    ckpt = torch.load(args.ckpt, map_location=device)
    model = TransformerLM(vocab_size=len(tok.itos), d_model=args.dim or 512, n_layers=args.n_layers or 8,
                          n_heads=args.n_heads or 8, seq_len=args.seq_len or 512)
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device)
    model.eval()

    files = sorted(glob.glob(os.path.join(args.corpus_dir, '**', '*.txt'), recursive=True))
    assert files, f"No .txt files under {args.corpus_dir}"
    texts = []
    for fp in files:
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read().strip()
        # split into passages ~512-1k chars
        for i in range(0, len(txt), 800):
            chunk = txt[i:i+800].strip()
            if chunk:
                texts.append(chunk)
    idx = RAGIndex(tok, model, device)
    idx.add_passages(texts, batch_size=16, max_len=args.max_len)
    idx.save(args.index_path)
    print(f"Saved index with {len(idx.passages)} passages to {args.index_path}")


def rag_prompt(query: str, contexts: List[str], k: int = 3) -> str:
    instruct = (
        "You are Hackna-GPT, a concise assistant. Use the CONTEXT to answer the QUESTION.\n"
        "If the answer is not in context, say you don't know.\n\n"
    )
    ctx = "\n---\n".join(contexts[:k])
    return f"{instruct}CONTEXT:\n{ctx}\n\nQUESTION: {query}\nANSWER:"


@torch.no_grad()
def chat_loop(args):
    device = detect_device()
    tok = BPETokenizer.load(args.vocab_path)
    ckpt = torch.load(args.ckpt, map_location=device)
    model_cfg = ckpt.get('config', {})
    model = TransformerLM(vocab_size=len(tok.itos), d_model=args.dim or 512, n_layers=args.n_layers or 8,
                          n_heads=args.n_heads or 8, seq_len=args.seq_len or model_cfg.get('seq_len', 512))
    model.load_state_dict(ckpt['model'], strict=False)
    model.to(device).eval()

    idx = None
    if args.index_path and os.path.exists(args.index_path):
        idx = RAGIndex.load(args.index_path, tok, model, device)
        print(f"Loaded RAG index with {len(idx.passages)} passages")

    bos = tok.stoi.get("<bos>")
    eos = tok.stoi.get("<eos>")

    print("\n[Hackna-GPT] Type your question. Ctrl+C to exit.\n")
    while True:
        try:
            q = input("You: ").strip()
            if not q:
                continue
            # Optional RAG
            prompt = q
            if idx is not None:
                hits = idx.search(q, k=args.rag_topk)
                contexts = [t for _, t in hits]
                if contexts:
                    prompt = rag_prompt(q, contexts, k=args.rag_use)
            ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                                 top_k=args.top_k, top_p=args.top_p, eos_id=eos)
            text = tok.decode(out[0].tolist())
            # strip the prompt
            ans = text[len(prompt):].strip()
            print(f"Hackna-GPT: {ans}\n")
        except KeyboardInterrupt:
            print("\nBye!")
            break


# ------------------------------
# Argparse
# ------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Hackna-GPT: pure-ML mini LLM + RAG")
    sub = p.add_subparsers(dest='cmd', required=True)

    # train_tokenizer
    p_tok = sub.add_parser('train_tokenizer', help='Train BPE tokenizer on a text folder')
    p_tok.add_argument('--data_dir', type=str, required=True)
    p_tok.add_argument('--out_dir', type=str, required=True)
    p_tok.add_argument('--vocab_size', type=int, default=30000)
    p_tok.add_argument('--max_bytes', type=int, default=50_000_000)
    p_tok.set_defaults(func=cmd_train_tokenizer)

    # train_lm
    p_tr = sub.add_parser('train_lm', help='Train the Transformer LM')
    p_tr.add_argument('--data_dir', type=str, required=True)
    p_tr.add_argument('--out_dir', type=str, required=True)
    p_tr.add_argument('--vocab_path', type=str, required=True)
    p_tr.add_argument('--seq_len', type=int, default=512)
    p_tr.add_argument('--dim', type=int, default=512)
    p_tr.add_argument('--n_layers', type=int, default=8)
    p_tr.add_argument('--n_heads', type=int, default=8)
    p_tr.add_argument('--dropout', type=float, default=0.0)
    p_tr.add_argument('--batch_size', type=int, default=16)
    p_tr.add_argument('--steps', type=int, default=10000)
    p_tr.add_argument('--lr', type=float, default=3e-4)
    p_tr.add_argument('--warmup_steps', type=int, default=500)
    p_tr.add_argument('--weight_decay', type=float, default=0.01)
    p_tr.add_argument('--grad_clip', type=float, default=1.0)
    p_tr.add_argument('--no_amp', action='store_true')
    p_tr.add_argument('--seed', type=int, default=1337)
    p_tr.set_defaults(func=cmd_train_lm)

    # build_index
    p_ix = sub.add_parser('build_index', help='Build a local vector index for RAG from text files')
    p_ix.add_argument('--corpus_dir', type=str, required=True)
    p_ix.add_argument('--vocab_path', type=str, required=True)
    p_ix.add_argument('--ckpt', type=str, required=True)
    p_ix.add_argument('--index_path', type=str, required=True)
    p_ix.add_argument('--seq_len', type=int, default=512)
    p_ix.add_argument('--dim', type=int, default=None)
    p_ix.add_argument('--n_layers', type=int, default=None)
    p_ix.add_argument('--n_heads', type=int, default=None)
    p_ix.add_argument('--max_len', type=int, default=256)
    p_ix.set_defaults(func=cmd_build_index)

    # chat
    p_chat = sub.add_parser('chat', help='Interactive chat with optional RAG')
    p_chat.add_argument('--vocab_path', type=str, required=True)
    p_chat.add_argument('--ckpt', type=str, required=True)
    p_chat.add_argument('--index_path', type=str, default=None)
    p_chat.add_argument('--seq_len', type=int, default=512)
    p_chat.add_argument('--dim', type=int, default=None)
    p_chat.add_argument('--n_layers', type=int, default=None)
    p_chat.add_argument('--n_heads', type=int, default=None)
    p_chat.add_argument('--max_new_tokens', type=int, default=128)
    p_chat.add_argument('--temperature', type=float, default=0.8)
    p_chat.add_argument('--top_k', type=int, default=None)
    p_chat.add_argument('--top_p', type=float, default=0.9)
    p_chat.add_argument('--rag_topk', type=int, default=5)
    p_chat.add_argument('--rag_use', type=int, default=3)
    p_chat.set_defaults(func=lambda a: chat_loop(a))

    return p


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
