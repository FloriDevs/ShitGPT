"""Winziges GPT: trainieren, weitertrainieren (--resume), Text generieren (--sample).

  python train.py                      # neu starten
  python train.py --resume --max_iters 10000   # vom letzten Checkpoint weiter
  python train.py --sample "Die Geschichte des"
"""
import argparse, math, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data")
p.add_argument("--out", default="out")
p.add_argument("--resume", action="store_true")
p.add_argument("--sample", default=None)
p.add_argument("--n_tokens", type=int, default=200)
p.add_argument("--temp", type=float, default=0.8)
# Modellgroesse (bei --resume/--sample aus dem Checkpoint uebernommen)
p.add_argument("--block", type=int, default=256)
p.add_argument("--d", type=int, default=256)
p.add_argument("--heads", type=int, default=4)
p.add_argument("--layers", type=int, default=4)
p.add_argument("--dropout", type=float, default=0.1)
# Training
p.add_argument("--max_iters", type=int, default=5000)   # absolutes Ziel
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=6e-4)
p.add_argument("--warmup", type=int, default=100)
p.add_argument("--eval_every", type=int, default=250)
args = p.parse_args()

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(1337)


# ---------------------------------------------------------------- Modell
class Block(nn.Module):
    def __init__(s, d, h, drop):
        super().__init__()
        s.h, s.drop_p = h, drop
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d, bias=False)
        s.proj = nn.Linear(d, d, bias=False)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(),
                              nn.Linear(4 * d, d, bias=False))
        s.drop = nn.Dropout(drop)

    def forward(s, x):
        B, T, C = x.shape
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = [t.view(B, T, s.h, C // s.h).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=s.drop_p if s.training else 0.0)
        x = x + s.drop(s.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + s.drop(s.mlp(s.ln2(x)))


class GPT(nn.Module):
    def __init__(s, vocab, block, d, heads, layers, dropout):
        super().__init__()
        s.block = block
        s.tok, s.pos = nn.Embedding(vocab, d), nn.Embedding(block, d)
        s.blocks = nn.ModuleList(Block(d, heads, dropout) for _ in range(layers))
        s.ln = nn.LayerNorm(d)
        s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.tok.weight            # Weight Tying spart Parameter
        s.apply(s._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(s, idx, targets=None):
        B, T = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x)
        logits = s.head(s.ln(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(s, idx, n, temp=0.8, top_k=50):
        for _ in range(n):
            logits, _ = s(idx[:, -s.block:])
            logits = logits[:, -1] / temp
            k = min(top_k, logits.size(-1))
            v, _ = torch.topk(logits, k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


# ---------------------------------------------------------------- Setup
tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
vocab = tok.get_vocab_size()
ckpt_path = os.path.join(args.out, "ckpt.pt")
ckpt = None
if (args.resume or args.sample is not None) and os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]                            # Architektur aus Checkpoint
else:
    cfg = dict(block=args.block, d=args.d, heads=args.heads,
               layers=args.layers, dropout=args.dropout)
if args.sample is not None and ckpt is None:
    raise SystemExit("Kein Checkpoint gefunden - erst trainieren.")

model = GPT(vocab, **cfg).to(device)
if ckpt:
    model.load_state_dict(ckpt["model"])
print(f"Device: {device} | Parameter: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")

# ---------------------------------------------------------------- Sampling
if args.sample is not None:
    eot = tok.token_to_id("<|eot|>")
    ids = tok.encode(args.sample).ids if args.sample else [eot]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    out = model.generate(x, args.n_tokens, temp=args.temp)
    print(tok.decode(out[0].tolist()))
    raise SystemExit


# ---------------------------------------------------------------- Training
def get_batch(split):
    data = np.memmap(os.path.join(args.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = np.random.randint(0, len(data) - cfg["block"] - 1, args.batch)
    x = np.stack([data[i:i + cfg["block"]] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + cfg["block"]] for i in ix]).astype(np.int64)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def estimate_loss():
    model.eval()
    res = {}
    for split in ("train", "val"):
        res[split] = sum(model(*get_batch(split))[1].item() for _ in range(20)) / 20
    model.train()
    return res


def lr_at(it):
    if it < args.warmup:
        return args.lr * (it + 1) / args.warmup
    if it >= args.max_iters:
        return args.lr / 10
    r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
    return args.lr / 10 + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.lr / 10)


def save(it):
    os.makedirs(args.out, exist_ok=True)
    tmp = ckpt_path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "iter": it, "cfg": cfg}, tmp)
    os.replace(tmp, ckpt_path)                   # atomar: kein kaputter Checkpoint


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
it = 0
if ckpt:
    opt.load_state_dict(ckpt["opt"])
    it = ckpt["iter"]
    print(f"Weiter ab Iteration {it}")

model.train()
t0 = time.time()
while it < args.max_iters:
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    if it % args.eval_every == 0:
        l = estimate_loss()
        print(f"iter {it}: train {l['train']:.3f} | val {l['val']:.3f} | {time.time() - t0:.0f}s")
        if it > 0:
            save(it)
    _, loss = model(*get_batch("train"))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    it += 1
save(it)
print("Fertig, Checkpoint gespeichert.")
