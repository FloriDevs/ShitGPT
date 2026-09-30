"""Chat-Oberflaeche fuer dein Mini-LLM.

  python chat.py            # Chat im Terminal
  python chat.py --web      # Chat im Browser: http://localhost:8000

Das Modell wird nur einmal geladen. Befehle im Terminal: /reset (Verlauf leeren), /quit
Das Chat-Format ist "Nutzer: ...\nBot: ..." - damit das Modell wirklich antwortet,
muss es mit Dialogen in genau diesem Format trainiert werden.
"""
import argparse, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data")
p.add_argument("--out", default="out")
p.add_argument("--temp", type=float, default=0.8)
p.add_argument("--top_k", type=int, default=40)
p.add_argument("--max_new", type=int, default=120)
p.add_argument("--user", default="Nutzer")
p.add_argument("--bot", default="Bot")
p.add_argument("--web", action="store_true")
p.add_argument("--port", type=int, default=8000)
args = p.parse_args()

device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available() else "cpu")


# ------------------------------------------- Modell (identisch zu train.py)
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
        s.head.weight = s.tok.weight

    def forward(s, idx):
        B, T = idx.shape
        x = s.tok(idx) + s.pos(torch.arange(T, device=idx.device))
        for b in s.blocks:
            x = b(x)
        return s.head(s.ln(x))


tok = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))
ckpt_path = os.path.join(args.out, "ckpt.pt")
if not os.path.exists(ckpt_path):
    raise SystemExit("Kein Checkpoint gefunden - erst trainieren.")
ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(tok.get_vocab_size(), **ckpt["cfg"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()
eot = tok.token_to_id("<|eot|>")
print(f"Modell geladen (Iteration {ckpt['iter']}, Device: {device})")

history = ""      # bisheriger Chatverlauf als Text


def visible(text):
    """Antwort bis zum ersten Zeilenumbruch bzw. bis das Modell den Nutzer spielt."""
    t = text.lstrip(" \n")
    done = False
    if "\n" in t:
        t, done = t.split("\n")[0], True
    cut = t.find(f"{args.user}:")
    if cut != -1:
        t, done = t[:cut], True
    return t.rstrip() if done else t, done


@torch.no_grad()
def stream_reply(user_text):
    """Erzeugt die Antwort und liefert sie stueckweise (Generator)."""
    global history
    prompt = f"{history}{args.user}: {user_text}\n{args.bot}:"
    x = torch.tensor([tok.encode(prompt).ids], device=device)
    out_ids, shown = [], ""
    for _ in range(args.max_new):
        logits = model(x[:, -model.block:])[:, -1] / args.temp
        v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float("inf")
        nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
        t = nxt.item()
        if t == eot:
            break
        x = torch.cat([x, nxt], dim=1)
        out_ids.append(t)
        text = tok.decode(out_ids)
        if text.endswith("\ufffd"):          # halbes Unicode-Zeichen abwarten
            continue
        vis, done = visible(text)
        if len(vis) > len(shown):
            yield vis[len(shown):]
            shown = vis
        if done:
            break
    history = (prompt + " " + shown.strip() + "\n")[-1500:]


def reset():
    global history
    history = ""


# ---------------------------------------------------------------- Terminal
def run_terminal():
    print("Chat gestartet. /reset leert den Verlauf, /quit beendet.\n")
    while True:
        try:
            msg = input(f"{args.user}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg == "/quit":
            break
        if msg == "/reset":
            reset()
            print("(Verlauf geleert)")
            continue
        if not msg:
            continue
        print(f"{args.bot}:", end="", flush=True)
        got = False
        for piece in stream_reply(msg):
            got = True
            print(piece, end="", flush=True)
        print("" if got else " (keine Antwort)")


# ---------------------------------------------------------------- Browser
PAGE = """<!doctype html><html lang="de"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shitmeni</title>
<style>
 body{margin:0;background:#15171c;color:#e6e6e6;font:16px system-ui,sans-serif;
      display:flex;flex-direction:column;height:100vh}
 #log{flex:1;overflow-y:auto;padding:16px;max-width:760px;width:100%;margin:0 auto;box-sizing:border-box}
 .m{margin:8px 0;padding:10px 14px;border-radius:12px;max-width:85%;white-space:pre-wrap;line-height:1.4}
 .u{background:#2b6cb0;margin-left:auto}.b{background:#262a33}
 #bar{display:flex;gap:8px;padding:12px;max-width:760px;width:100%;margin:0 auto;box-sizing:border-box}
 input{flex:1;padding:12px;border-radius:10px;border:1px solid #3a3f4b;background:#1d2027;color:inherit;font:inherit}
 button{padding:0 18px;border-radius:10px;border:0;background:#3a3f4b;color:inherit;font:inherit;cursor:pointer}
</style>
<div id="log"></div>
<div id="bar"><input id="in" placeholder="Nachricht ..." autofocus>
<button id="go">Senden</button><button id="rs">Neu</button></div>
<script>
const log=document.getElementById('log'),inp=document.getElementById('in');
function add(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d}
async function send(){
  const t=inp.value.trim();if(!t)return;inp.value='';add(t,'u');
  const b=add('...','b');
  try{const r=await fetch('/chat',{method:'POST',body:JSON.stringify({message:t})});
      b.textContent=(await r.json()).reply||'(keine Antwort)'}
  catch(e){b.textContent='Fehler: '+e}
}
document.getElementById('go').onclick=send;
inp.onkeydown=e=>{if(e.key==='Enter')send()};
document.getElementById('rs').onclick=async()=>{await fetch('/reset',{method:'POST'});log.innerHTML=''};
</script></html>"""


def run_web():
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            data = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(n).decode("utf-8") if n else "{}"
            if self.path == "/reset":
                reset()
                self._send("{}", "application/json")
            else:
                msg = json.loads(body).get("message", "")
                reply = "".join(stream_reply(msg)).strip()
                self._send(json.dumps({"reply": reply}), "application/json")

        def log_message(self, *a):
            pass

    print(f"Browser-Chat: http://localhost:{args.port}  (Strg+C beendet)")
    HTTPServer(("127.0.0.1", args.port), H).serve_forever()


if args.web:
    run_web()
else:
    run_terminal()
