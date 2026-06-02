"""Sanity check the tokenizer.json we built."""

import json
import sys

with open("tokenizer.json", "r", encoding="utf-8") as f:
    t = json.load(f)

print("Model type:", t["model_type"])
print("Vocab size:", len(t["vocab"]))
print("Merges count:", len(t["merges"]))
print("Special tokens:", t["special_tokens"])

# Invert vocab
inv_vocab = {v: k for k, v in t["vocab"].items()}

# Check TTS special token IDs from config.json
for tid in [151672, 151673, 151671, 151669, 151670, 151675, 151643]:
    name = inv_vocab.get(tid, "NOT FOUND")
    print(f"  ID {tid}: {name}")

# Check codec special tokens
for tid in [2149, 2150, 2154, 2155, 2148]:
    name = inv_vocab.get(tid, "NOT FOUND")
    print(f"  ID {tid}: {name}")
