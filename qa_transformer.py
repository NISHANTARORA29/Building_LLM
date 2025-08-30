# qa_transformer.py
# Train a from-scratch Transformer (encoder-decoder) for custom Q&A pairs in CSV.
# No HuggingFace. Uses GELU, Adam. Greedy decoding for evaluation.
# Usage:
#   python qa_transformer.py --train_csv my_qa.csv --epochs 10 --batch_size 32
#   (Optional) --val_csv val_qa.csv
import argparse, csv, math, os, re, random, time
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------- Utils -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def clean_text(s: str) -> str:
    # lowercase + basic punctuation spacing
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    # optional: separate punctuation to their own tokens
    s = re.sub(r"([.,!?;:()\"'])", r" \1 ", s)
    s = re.sub(r"\s+", " ", s)
    return s

# ----------------------------- Vocab -----------------------------
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)

class Vocab:
    def __init__(self, counter: Counter, min_freq: int = 2, max_size: Optional[int] = None):
        # build word-level vocab
        items = [(w, c) for w, c in counter.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        if max_size:
            items = items[: max(0, max_size - len(SPECIALS))]
        itos = SPECIALS + [w for w, _ in items]
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    def encode(self, text: str, add_eos: bool = False, max_len: Optional[int] = None) -> List[int]:
        toks = clean_text(text).split()
        ids = [self.stoi.get(t, UNK) for t in toks]
        if add_eos:
            ids = ids + [EOS]
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = []
        for i in ids:
            if i == EOS: break
            if i == PAD: continue
            toks.append(self.itos[i] if i < len(self.itos) else "<unk>")
        return " ".join(toks)

    def __len__(self): return len(self.itos)

# ----------------------------- Data -----------------------------
class QADataset(Dataset):
    def __init__(self, rows: List[Tuple[str, str]], vocab: Vocab, max_src_len=128, max_tgt_len=128):
        self.data = rows
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        q, a = self.data[idx]
        src = self.vocab.encode(q, add_eos=True, max_len=self.max_src_len)
        tgt = self.vocab.encode(a, add_eos=True, max_len=self.max_tgt_len)
        # decoder input starts with <sos>, target_out is gold (with <eos> at end)
        tgt_in = [SOS] + tgt[:-0]  # keep eos for supervision; decoder will learn to emit EOS
        # clip tgt_in to max len (ensure at least 1)
        if len(tgt_in) > self.max_tgt_len + 1:
            tgt_in = tgt_in[: self.max_tgt_len + 1]
        return torch.tensor(src), torch.tensor(tgt_in), torch.tensor(tgt)

def pad_batch(batch, pad_idx=PAD):
    srcs, tgts_in, tgts_out = zip(*batch)
    src_max = max(x.size(0) for x in srcs)
    tgt_max = max(x.size(0) for x in tgts_in)
    def pad_seq(seq, maxlen):
        out = torch.full((len(seq), maxlen), pad_idx, dtype=torch.long)
        for i, s in enumerate(seq):
            out[i, : s.size(0)] = s
        return out
    return pad_seq(srcs, src_max), pad_seq(tgts_in, tgt_max), pad_seq(tgts_out, max(x.size(0) for x in tgts_out))

def load_csv(path: str) -> List[Tuple[str, str]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        assert "question" in rdr.fieldnames and "answer" in rdr.fieldnames, "CSV must have headers: question,answer"
        for r in rdr:
            q = (r["question"] or "").strip()
            a = (r["answer"] or "").strip()
            if q and a: rows.append((q, a))
    return rows

def build_vocab(rows: List[Tuple[str, str]], min_freq=2, max_size=None) -> Vocab:
    counter = Counter()
    for q, a in rows:
        counter.update(clean_text(q).split())
        counter.update(clean_text(a).split())
    return Vocab(counter, min_freq=min_freq, max_size=max_size)

# ----------------------------- Transformer Core -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

def attention(q, k, v, mask=None, dropout: Optional[nn.Module] = None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # mask == 0 where we want to block
        scores = scores.masked_fill(mask == 0, float("-inf"))
    p = F.softmax(scores, dim=-1)
    if dropout is not None:
        p = dropout(p)
    return p @ v, p

class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.h = h
        self.d_k = d_model // h
        self.qkv = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Tq, D = q.shape
        def proj(x, lin):
            x = lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # (B,h,T,d_k)
            return x
        qh, kh, vh = [proj(x, l) for x, l in zip((q, k, v), self.qkv)]
        if mask is not None:
            # mask shape should broadcast to (B, h, Tq, Tk)
            mask = mask.unsqueeze(1)
        ctx, _ = attention(qh, kh, vh, mask=mask, dropout=self.drop)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, self.h * self.d_k)
        return self.out(ctx)

class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.drop(F.gelu(self.w1(x))))  # GELU

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(heads, d_model, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        y = self.mha(self.norm1(x), self.norm1(x), self.norm1(x), src_mask)
        x = x + self.drop1(y)
        y = self.ff(self.norm2(x))
        x = x + self.drop2(y)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_mha = MultiHeadAttention(heads, d_model, dropout)
        self.src_mha = MultiHeadAttention(heads, d_model, dropout)
        self.ff = PositionwiseFF(d_model, d_ff, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.d3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        y = self.self_mha(self.n1(x), self.n1(x), self.n1(x), tgt_mask)
        x = x + self.d1(y)
        y = self.src_mha(self.n2(x), memory, memory, src_mask)
        x = x + self.d2(y)
        y = self.ff(self.n3(x))
        x = x + self.d3(y)
        return x

class Encoder(nn.Module):
    def __init__(self, N, d_model, heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask):
        for l in self.layers: x = l(x, src_mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, N, d_model, heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for l in self.layers: x = l(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, N_enc=4, N_dec=4, heads=4, d_ff=1024, dropout=0.1, max_len=1024):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, dropout, max_len)
        self.encoder = Encoder(N_enc, d_model, heads, d_ff, dropout)
        self.decoder = Decoder(N_dec, d_model, heads, d_ff, dropout)
        self.out = nn.Linear(d_model, tgt_vocab)
        self._reset_params()

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def encode(self, src_ids, src_mask):
        x = self.pos(self.src_embed(src_ids))
        return self.encoder(x, src_mask)

    def decode(self, tgt_ids, memory, src_mask, tgt_mask):
        y = self.pos(self.tgt_embed(tgt_ids))
        return self.decoder(y, memory, src_mask, tgt_mask)

    def forward(self, src_ids, tgt_in_ids, src_mask, tgt_mask):
        memory = self.encode(src_ids, src_mask)
        dec = self.decode(tgt_in_ids, memory, src_mask, tgt_mask)
        return self.out(dec)

# ----------------------------- Masks -----------------------------
def make_src_mask(src_ids, pad_idx=PAD):
    # (B, S) -> (B, 1, 1, S) True where valid
    return (src_ids != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt_ids, pad_idx=PAD):
    # padding mask
    pad_mask = (tgt_ids != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
    T = tgt_ids.size(1)
    # subsequent mask: allow attending to <= current position
    subsequent = torch.triu(torch.ones((1, 1, T, T), device=tgt_ids.device, dtype=torch.bool), diagonal=1)
    look_ahead = ~subsequent  # True where allowed
    return pad_mask & look_ahead

# ----------------------------- Training / Inference -----------------------------
def greedy_decode(model: Seq2SeqTransformer, src, vocab: Vocab, max_len=64):
    model.eval()
    with torch.no_grad():
        src_mask = make_src_mask(src)
        memory = model.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), SOS, dtype=torch.long, device=src.device)
        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys)
            out = model.decode(ys, memory, src_mask, tgt_mask)
            logits = model.out(out[:, -1:, :])  # (B,1,V)
            next_token = logits.argmax(dim=-1)  # (B,1)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == EOS).all():
                break
        return ys[:, 1:]  # drop SOS

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)
        logits = model(src, tgt_in, src_mask, tgt_mask)  # (B,T,V)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * src.size(0); count += src.size(0)
    return total / max(count, 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, vocab: Vocab, show_samples=2):
    model.eval()
    total, count = 0.0, 0
    samples_printed = 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_in)
        logits = model(src, tgt_in, src_mask, tgt_mask)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        total += loss.item() * src.size(0); count += src.size(0)

        # sample decode a couple of examples
        if samples_printed < show_samples:
            pred = greedy_decode(model, src[:1], vocab, max_len=tgt_out.size(1))
            print("\n[SAMPLE]")
            print("Q:", vocab.decode(src[0].tolist()))
            print("Gold:", vocab.decode(tgt_out[0].tolist()))
            print("Pred:", vocab.decode(pred[0].tolist()))
            samples_printed += 1

    return total / max(count, 1)

# ----------------------------- Main -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_vocab", type=int, default=None)
    parser.add_argument("--max_src_len", type=int, default=64)
    parser.add_argument("--max_tgt_len", type=int, default=128)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="qa_transformer.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    # Load data
    train_rows = load_csv(args.train_csv)
    if not train_rows:
        raise ValueError("No training rows found in CSV.")
    val_rows = load_csv(args.val_csv) if args.val_csv else None

    # Build vocab from train (and val if provided)
    rows_for_vocab = train_rows + (val_rows or [])
    vocab = build_vocab(rows_for_vocab, min_freq=args.min_freq, max_size=args.max_vocab)
    print(f"Vocab size: {len(vocab)}")

    # Datasets
    train_ds = QADataset(train_rows, vocab, args.max_src_len, args.max_tgt_len)
    val_ds = QADataset(val_rows, vocab, args.max_src_len, args.max_tgt_len) if val_rows else None

    # Loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_batch) if val_ds else None

    # Model
    model = Seq2SeqTransformer(
        src_vocab=len(vocab),
        tgt_vocab=len(vocab),
        d_model=args.d_model,
        N_enc=args.enc_layers,
        N_dec=args.dec_layers,
        heads=args.heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=max(args.max_src_len, args.max_tgt_len) + 10
    ).to(args.device)

    # Optimizer / Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    best_val = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        msg = f"Epoch {epoch}/{args.epochs} | train_loss={tr_loss:.4f}"
        if val_loader:
            val_loss = evaluate(model, val_loader, criterion, args.device, vocab)
            msg += f" | val_loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model": model.state_dict(), "vocab": vocab.itos}, args.save_path)
                msg += " | (saved)"
        else:
            # Save best-so-far on training only
            if tr_loss < best_val:
                best_val = tr_loss
                torch.save({"model": model.state_dict(), "vocab": vocab.itos}, args.save_path)
                msg += " | (saved)"
        print(msg)

    print(f"Done in {time.time()-t0:.1f}s. Best checkpoint at {args.save_path}")

    # Quick interactive demo on a few training samples
    print("\n== Demo on a few training questions ==")
    model.eval()
    for i in range(min(3, len(train_ds))):
        src, tgt_in, tgt_out = train_ds[i]
        src = src.unsqueeze(0).to(args.device)
        pred = greedy_decode(model, src, vocab, max_len=args.max_tgt_len)
        print(f"Q: {vocab.decode(src[0].tolist())}")
        print(f"A_gold: {vocab.decode(tgt_out.tolist())}")
        print(f"A_pred: {vocab.decode(pred[0].tolist())}\n")

if __name__ == "__main__":
    main()
