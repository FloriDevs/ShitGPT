"""Baut aus Wikipedia + eigenen .txt-Dateien die Trainingsdaten.

Beispiele:
  python prepare.py --wiki_lang de --wiki_articles 20000 --own_dir eigene_texte
  python prepare.py --wiki_articles 0 --own_dir eigene_texte     # nur eigene Texte

Der Tokenizer wird nur EINMAL trainiert (data/tokenizer.json). Danach wird er
wiederverwendet, damit alte Checkpoints kompatibel bleiben.
"""
import argparse, glob, os, random
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="data")
p.add_argument("--wiki_lang", default="de")           # de, en, ...
p.add_argument("--wiki_articles", type=int, default=20000)
p.add_argument("--own_dir", default="eigene_texte")   # Ordner mit .txt-Dateien
p.add_argument("--own_repeat", type=int, default=1)   # eigene Texte x-fach gewichten
p.add_argument("--vocab_size", type=int, default=4096)
args = p.parse_args()
os.makedirs(args.data_dir, exist_ok=True)


def wiki_texts():
    if args.wiki_articles <= 0:
        return
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{args.wiki_lang}",
                      split="train", streaming=True)
    for i, ex in enumerate(ds):
        if i >= args.wiki_articles:
            break
        yield ex["text"]


def own_texts():
    for path in glob.glob(os.path.join(args.own_dir, "**/*.txt"), recursive=True):
        with open(path, encoding="utf-8", errors="ignore") as f:
            t = f.read().strip()
        if t:
            yield t


docs = list(wiki_texts())
own = list(own_texts())
print(f"{len(docs)} Wikipedia-Artikel, {len(own)} eigene Dateien")
docs += own * max(1, args.own_repeat)
if not docs:
    raise SystemExit("Keine Daten gefunden.")

tok_path = os.path.join(args.data_dir, "tokenizer.json")
if os.path.exists(tok_path):
    tok = Tokenizer.from_file(tok_path)
    print("Vorhandenen Tokenizer wiederverwendet")
else:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|eot|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(docs, trainer)
    tok.save(tok_path)
    print("Neuen Tokenizer trainiert")

eot = tok.token_to_id("<|eot|>")
random.seed(1337)
random.shuffle(docs)
n_val = max(1, len(docs) // 20)          # 5 % Validierung (dokumentweise)
splits = {"val": docs[:n_val], "train": docs[n_val:]}

for name, items in splits.items():
    chunks = []
    for i in range(0, len(items), 1000):
        for enc in tok.encode_batch(items[i:i + 1000]):
            chunks.append(np.array(enc.ids + [eot], dtype=np.uint16))
    arr = np.concatenate(chunks)
    arr.tofile(os.path.join(args.data_dir, f"{name}.bin"))
    print(f"{name}: {len(arr):,} Tokens")
