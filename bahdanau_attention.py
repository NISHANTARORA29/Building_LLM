# bahdanau_attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim, attn_dim):
        super().__init__()
        self.W_h = nn.Linear(enc_dim, attn_dim, bias=False)
        self.W_s = nn.Linear(dec_dim, attn_dim, bias=False)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_state, mask=None):
        """
        enc_outputs: [B, T, enc_dim]
        dec_state:   [B, dec_dim]
        mask:        [B, T] (1 for valid, 0 for pad)
        returns: context [B, enc_dim], attn_weights [B, T]
        """
        # scores = v^T tanh(W_h h_i + W_s s_t)
        Wh = self.W_h(enc_outputs)                 # [B, T, A]
        Ws = self.W_s(dec_state).unsqueeze(1)      # [B, 1, A]
        score = self.v(torch.tanh(Wh + Ws)).squeeze(-1)  # [B, T]

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)

        attn = F.softmax(score, dim=-1)            # [B, T]
        context = torch.bmm(attn.unsqueeze(1), enc_outputs).squeeze(1)  # [B, enc_dim]
        return context, attn

class Encoder(nn.Module):
    def __init__(self, vocab, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.rnn = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.out_dim = 2*hid

    def forward(self, x, lengths):
        emb = self.emb(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out  # [B,T,2H]

class Decoder(nn.Module):
    def __init__(self, vocab, emb, enc_dim, dec_hid, attn_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.attn = BahdanauAttention(enc_dim, dec_hid, attn_dim)
        self.rnn = nn.GRU(emb + enc_dim, dec_hid, batch_first=True)
        self.fc  = nn.Linear(dec_hid, vocab)

    def forward(self, y_inp, enc_outputs, mask):
        """
        Teacher forcing for demo: y_inp is target shifted right
        """
        B, T = y_inp.size()
        dec_state = torch.zeros(1, B, self.rnn.hidden_size, device=y_inp.device)
        logits = []

        emb_toks = self.emb(y_inp)  # [B,T,E]
        h_t = dec_state

        for t in range(T):
            context, _ = self.attn(enc_outputs, h_t[-1], mask)    # [B,enc_dim]
            x_t = torch.cat([emb_toks[:, t, :], context], dim=-1) # [B,E+enc_dim]
            out, h_t = self.rnn(x_t.unsqueeze(1), h_t)            # out: [B,1,H]
            logits.append(self.fc(out.squeeze(1)))                # [B,V]

        return torch.stack(logits, dim=1)  # [B,T,V]

# ---- Tiny demo ----
if __name__ == "__main__":
    B, T_src, T_tgt = 2, 7, 6
    V, E, H = 1000, 64, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = Encoder(V, E, H//2).to(device)  # biGRU -> enc_dim = H
    dec = Decoder(V, E, enc_dim=H, dec_hid=H, attn_dim=64).to(device)

    x = torch.randint(0, V, (B, T_src), device=device)
    lengths = torch.tensor([7, 5], device=device)
    mask = torch.arange(T_src, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # [B,T_src]

    y_inp = torch.randint(0, V, (B, T_tgt), device=device)

    enc_out = enc(x, lengths)             # [B,T_src,H]
    logits  = dec(y_inp, enc_out, mask)   # [B,T_tgt,V]
    print(logits.shape)
