"""Build tokenizer.json from vocab.json + merges.txt for Qwen3-TTS."""

import json
import sys
from pathlib import Path


def main():
    weights_dir = Path("weights/converted")

    # Load vocab
    with open(weights_dir / "vocab.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    print(f"Vocab size: {len(vocab)}")

    # Load merges
    with open(weights_dir / "merges.txt", "r", encoding="utf-8") as f:
        merges = [line.rstrip() for line in f if line.strip()]
    print(f"Merges count: {len(merges)}")

    # Load tokenizer config for special token IDs
    with open(weights_dir / "tokenizer_config.json", "r", encoding="utf-8") as f:
        tok_config = json.load(f)

    # Build tokenizer.json in the format our Rust code expects
    tokenizer_json = {
        "model_type": "BPE",
        "vocab": vocab,
        "merges": merges,
        "special_tokens": {
            "codec_bos_id": 2149,
            "codec_eos_id": 2150,
            "codec_think_id": 2154,
            "codec_nothink_id": 2155,
            "pad_token_id": 151643,
        },
    }

    # Also check if there are special TTS tokens in the config
    added_tokens = tok_config.get("added_tokens_decoder", {})
    print(f"Added tokens: {len(added_tokens)}")

    # Write output
    output_path = Path("tokenizer.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, ensure_ascii=False)
    print(f"Wrote {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
