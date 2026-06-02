#!/usr/bin/env python3
"""
Export weights from SafeTensors for alignment testing.

This script extracts specific weight tensors from the converted SafeTensors
files and saves them in a format suitable for Rust-side alignment testing.

Usage:
    python export_weights_for_alignment.py --input weights/converted/model.safetensors \
        --output weights/alignments --component talker
"""

import argparse
import json
import sys
from pathlib import Path

try:
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: pip install safetensors torch")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: pip install torch")
    sys.exit(1)


def export_talker_weights(input_path: Path, output_dir: Path):
    """Export talker backbone weights for alignment."""
    print("  Exporting talker weights...")
    tensors = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        # Export embedding layers
        for key in keys:
            if "embed" in key.lower():
                tensors[key] = f.get_tensor(key).float()

        # Export attention layers (first 4 layers for quick testing)
        for key in keys:
            if "layers.0." in key or "layers.1." in key:
                if "attn" in key or "self_attn" in key:
                    tensors[key] = f.get_tensor(key).float()
                elif "mlp" in key:
                    tensors[key] = f.get_tensor(key).float()
                elif "ln" in key or "norm" in key:
                    tensors[key] = f.get_tensor(key).float()

    if tensors:
        save_file(tensors, str(output_dir / "talker_layers.safetensors"))
        print(f"    Saved {len(tensors)} talker tensors")
    return tensors


def export_code_predictor_weights(input_path: Path, output_dir: Path):
    """Export code predictor (MTP) weights for alignment."""
    print("  Exporting code predictor weights...")
    tensors = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        for key in keys:
            if "code_predictor" in key or "lm_head" in key:
                tensors[key] = f.get_tensor(key).float()

    if tensors:
        save_file(tensors, str(output_dir / "code_predictor.safetensors"))
        print(f"    Saved {len(tensors)} code predictor tensors")
    return tensors


def export_speaker_weights(input_path: Path, output_dir: Path):
    """Export speaker encoder weights for alignment."""
    print("  Exporting speaker encoder weights...")
    tensors = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        for key in keys:
            if "speaker" in key or "asp" in key or "blocks" in key:
                tensors[key] = f.get_tensor(key).float()

    if tensors:
        save_file(tensors, str(output_dir / "speaker_encoder.safetensors"))
        print(f"    Saved {len(tensors)} speaker encoder tensors")
    return tensors


def export_tokenizer_decoder_weights(input_path: Path, output_dir: Path):
    """Export tokenizer decoder (code2wav) weights for alignment."""
    print("  Exporting tokenizer decoder weights...")
    tensors = {}

    with safe_open(str(input_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())

        for key in keys:
            if "decoder" in key:
                tensors[key] = f.get_tensor(key).float()

    if tensors:
        save_file(tensors, str(output_dir / "tokenizer_decoder.safetensors"))
        print(f"    Saved {len(tensors)} tokenizer decoder tensors")
    return tensors


def generate_test_input(output_dir: Path, seq_len: int = 8):
    """Generate a deterministic test input for alignment comparison."""
    print("  Generating test input...")
    torch.manual_seed(42)

    # Fake token sequence (16 codebook layers x seq_len frames)
    tokens = torch.randint(0, 2048, (16, seq_len), dtype=torch.long)
    save_file(
        {"test_tokens": tokens},
        str(output_dir / "test_input.safetensors"),
    )
    print(f"    Saved test tokens: shape={list(tokens.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Export weights for alignment testing")
    parser.add_argument(
        "--input", "-i", required=True, help="Input SafeTensors file (converted)"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for alignment files"
    )
    parser.add_argument(
        "--tokenizer-input", "-t", help="Input SafeTensors file for tokenizer decoder"
    )
    parser.add_argument(
        "--component",
        choices=["all", "talker", "code_predictor", "speaker", "tokenizer"],
        default="all",
        help="Which components to export",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print()

    print("Exporting weights for alignment...")

    if args.component in ("all", "talker"):
        export_talker_weights(input_path, output_dir)

    if args.component in ("all", "code_predictor"):
        export_code_predictor_weights(input_path, output_dir)

    if args.component in ("all", "speaker"):
        export_speaker_weights(input_path, output_dir)

    if args.component in ("all", "tokenizer") and args.tokenizer_input:
        export_tokenizer_decoder_weights(Path(args.tokenizer_input), output_dir)

    # Always generate test input
    generate_test_input(output_dir)

    # Save export metadata
    meta = {
        "input": str(input_path),
        "tokenizer_input": args.tokenizer_input,
        "components": args.component,
    }
    with open(output_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("[OK] Export complete!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
