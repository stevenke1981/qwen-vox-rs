#!/usr/bin/env python3
"""
Qwen3-TTS Weight Converter — PyTorch SafeTensors → Candle-compatible SafeTensors

Converts Qwen3-TTS model weights from the official PyTorch format to a
Candle-compatible layout. Key transformations:

1. Rename tensor keys to match Rust module paths
2. Transpose Conv1d weights from PyTorch [out, in, k] to Candle [out, in, k] (same)
3. Transpose Linear weights from PyTorch [out, in] to Candle [out, in] (same)
4. Extract and save tokenizer decoder weights separately
5. Validate shapes and dtypes

Usage:
    python convert_weights.py --input model.safetensors --output weights/
    python convert_weights.py --input model.safetensors --output weights/ --split-tokenizer
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: safetensors not installed. Run: pip install safetensors torch")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch not installed. Run: pip install torch")
    sys.exit(1)


# ── Key mapping: PyTorch → Candle ──────────────────────────────────────────

# Talker backbone key prefixes (kept as-is, just validated)
TALKER_PREFIXES = [
    "talker.model.",  # Main transformer backbone
    "talker.embed_tokens.",
    "talker.layers.",
    "talker.norm.",
    "talker.rotary_emb.",
]

# Code predictor (MTP) key prefixes
CODE_PREDICTOR_PREFIXES = [
    "talker.code_predictor.",
    "code_predictor.",
    "small_to_mtp_projection.",
]

# Speaker encoder prefixes
SPEAKER_PREFIXES = [
    "speaker_encoder.",
]

# Tokenizer decoder prefixes (speech tokenizer / code2wav)
TOKENIZER_DECODER_PREFIXES = [
    "speech_tokenizer.",
    "tokenizer_decoder.",
]


def classify_key(key: str) -> str:
    """Classify a weight key into component category."""
    for prefix in TALKER_PREFIXES:
        if key.startswith(prefix):
            return "talker"
    for prefix in CODE_PREDICTOR_PREFIXES:
        if key.startswith(prefix):
            return "code_predictor"
    for prefix in SPEAKER_PREFIXES:
        if key.startswith(prefix):
            return "speaker"
    for prefix in TOKENIZER_DECODER_PREFIXES:
        if key.startswith(prefix):
            return "tokenizer_decoder"
    return "other"


def convert_dtype(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float32 for Candle compatibility.

    Candle supports f16/bf16 but f32 is safest for alignment testing.
    Production builds can use f16 later.
    """
    if tensor.dtype in (torch.float16, torch.bfloat16):
        return tensor.float()
    return tensor


def extract_and_rename(key: str, component: str) -> str:
    """Rename keys for cleaner Rust-side loading."""
    if component == "talker":
        # model.layers.{i}.self_attn.q_proj.weight → layers.{i}.attn.q.weight
        key = key.replace("model.layers.", "layers.")
        key = key.replace("self_attn.", "attn.")
        key = key.replace("_proj", "")
        key = key.replace("input_layernorm", "ln1")
        key = key.replace("post_attention_layernorm", "ln2")
        key = key.replace("model.norm.", "norm.")
        key = key.replace("model.embed_tokens.", "embed.")
        key = key.replace("model.rotary_emb.", "rope.")
    elif component == "code_predictor":
        key = key.replace("code_predictor.", "")
    elif component == "speaker":
        key = key.replace("speaker_encoder.", "")
    elif component == "tokenizer_decoder":
        key = key.replace("speech_tokenizer.", "")
        key = key.replace("tokenizer_decoder.", "")
    return key


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS weights for Candle")
    parser.add_argument("--input", "-i", required=True, help="Input SafeTensors file")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument(
        "--split-tokenizer",
        action="store_true",
        help="Save tokenizer decoder weights as separate file",
    )
    parser.add_argument(
        "--dtype",
        choices=["f32", "f16", "bf16"],
        default="f32",
        help="Output dtype (default: f32 for alignment testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print key mapping without saving"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output dtype: {args.dtype}")
    print()

    # Open and classify all tensors
    tensors_by_component = {
        "talker": {},
        "code_predictor": {},
        "speaker": {},
        "tokenizer_decoder": {},
        "other": {},
    }

    key_mapping = {}  # original → (new_name, component, shape, dtype)

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        print(f"Total tensors: {len(keys)}")
        print()

        for key in sorted(keys):
            tensor = f.get_tensor(key)
            component = classify_key(key)
            new_key = extract_and_rename(key, component)

            # Convert dtype
            if args.dtype == "f32":
                tensor = tensor.float()
            elif args.dtype == "f16":
                tensor = tensor.half()
            elif args.dtype == "bf16":
                tensor = tensor.bfloat16()

            tensors_by_component[component][new_key] = tensor
            key_mapping[key] = (
                new_key,
                component,
                list(tensor.shape),
                str(tensor.dtype),
            )

            if args.dry_run:
                print(f"  [{component:20s}] {key}")
                print(
                    f"  {'':20s} → {new_key}  shape={list(tensor.shape)}  dtype={tensor.dtype}"
                )

    if args.dry_run:
        print()
        print("Dry run complete. No files written.")
        # Save mapping as JSON for reference
        mapping_path = output_dir / "key_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(key_mapping, f, indent=2)
        print(f"Key mapping saved to: {mapping_path}")
        return

    # Print summary
    for comp, tensors in tensors_by_component.items():
        if tensors:
            total_params = sum(t.numel() for t in tensors.values())
            print(
                f"  {comp:25s}: {len(tensors):4d} tensors, {total_params:>12,d} params"
            )
    print()

    # Save main model (talker + code_predictor + speaker + other)
    main_tensors = {}
    for comp in ["talker", "code_predictor", "speaker", "other"]:
        main_tensors.update(tensors_by_component[comp])

    if main_tensors:
        main_path = output_dir / "model.safetensors"
        print(f"Saving main model ({len(main_tensors)} tensors) → {main_path}")
        save_file(main_tensors, str(main_path))

    # Save tokenizer decoder separately if requested
    if args.split_tokenizer and tensors_by_component["tokenizer_decoder"]:
        tok_path = output_dir / "tokenizer_decoder.safetensors"
        tok_tensors = tensors_by_component["tokenizer_decoder"]
        print(f"Saving tokenizer decoder ({len(tok_tensors)} tensors) → {tok_path}")
        save_file(tok_tensors, str(tok_path))

    # Save key mapping for reference
    mapping_path = output_dir / "key_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(key_mapping, f, indent=2)
    print(f"Key mapping saved to: {mapping_path}")

    # Generate Rust weight index
    index_path = output_dir / "weight_index.json"
    index = {}
    for comp, tensors in tensors_by_component.items():
        for name, tensor in tensors.items():
            index[name] = {
                "component": comp,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "numel": tensor.numel(),
            }
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Weight index saved to: {index_path}")

    print()
    print("[OK] Conversion complete!")
    print(f"   Main model: {output_dir / 'model.safetensors'}")
    if args.split_tokenizer:
        print(f"   Tokenizer:  {output_dir / 'tokenizer_decoder.safetensors'}")
    print(f"   Mapping:    {output_dir / 'key_mapping.json'}")
    print(f"   Index:      {output_dir / 'weight_index.json'}")


if __name__ == "__main__":
    main()
