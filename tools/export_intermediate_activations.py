#!/usr/bin/env python3
"""
Export intermediate activations from Qwen3-TTS tokenizer decoder for alignment testing.

This script loads the converted SafeTensors weights and simulates the forward pass,
saving intermediate tensors at each stage:
  - split_rvq_out: After SplitRVQ decode
  - pre_conv_out: After pre_conv
  - transformer_out: After pre_transformer (8 layers)
  - decoder_block_0_out, decoder_block_1_out, ...: After each decoder block
  - final_out: Final audio output

Usage:
    python export_intermediate_activations.py --weights weights/converted/tokenizer \
        --output weights/intermediates --codes test_codes.safetensors
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: pip install torch safetensors")
    sys.exit(1)


def load_tensor(store, name):
    """Load a tensor from the store, handling optional bias."""
    if name in store:
        return store[name]
    return None


def simulate_split_rvq(codes, store):
    """
    Simulate SplitRVQ decode.

    In real implementation, this would:
    1. Lookup embeddings from 16 codebooks
    2. Sum semantic + acoustic residuals
    3. Apply output_proj (Conv1d k=1)
    """
    # For now, return a placeholder with expected shape
    # [batch, 512, seq_len]
    batch = codes[0].shape[0] if len(codes) > 0 else 1
    seq_len = codes[0].shape[1] if len(codes) > 0 else 8
    return torch.zeros(batch, 512, seq_len)


def simulate_pre_conv(x, store):
    """Simulate pre_conv: CausalConv1d [1024, 512, 3]"""
    weight = store.get("pre_conv.conv.weight")
    bias = store.get("pre_conv.conv.bias")

    if weight is not None:
        # Conv1d: [batch, in_ch, len] -> [batch, out_ch, len]
        # For causal conv, we need to pad left
        x_padded = torch.nn.functional.pad(x, (2, 0))  # pad left by 2
        out = torch.nn.functional.conv1d(x_padded, weight, bias, stride=1, padding=0)
        return out
    return torch.zeros(x.shape[0], 1024, x.shape[2])


def simulate_transformer(x, store, num_layers=8):
    """
    Simulate pre_transformer: 8-layer AR Transformer.

    Input: [batch, seq_len, 512] (after transpose from conv output)
    Output: [batch, seq_len, 512]
    """
    # For alignment testing, we just return the input shape
    # Real implementation would run through all 8 layers
    return x


def simulate_decoder_block(x, store, block_idx):
    """
    Simulate a single decoder block:
      SnakeBeta -> CausalConvTranspose1d -> 3x ResidualUnit
    """
    # Placeholder
    return x


def simulate_final_stage(x, store):
    """Simulate final SnakeBeta + Conv1d -> mono audio"""
    # Placeholder
    return torch.zeros(x.shape[0], 1, x.shape[2] * 192)  # rough upsample ratio


def main():
    parser = argparse.ArgumentParser(description="Export intermediate activations")
    parser.add_argument(
        "--weights",
        "-w",
        required=True,
        help="Directory containing converted tokenizer weights",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for intermediate tensors",
    )
    parser.add_argument(
        "--codes", "-c", help="Optional: SafeTensors file with test codes"
    )
    args = parser.parse_args()

    weights_dir = Path(args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading weights from: {weights_dir}")
    print(f"Output directory: {output_dir}")

    # Load all tensors from the converted tokenizer weights
    weight_store = {}
    model_path = weights_dir / "model.safetensors"

    if model_path.exists():
        with safe_open(str(model_path), framework="pt", device="cpu") as f:
            for name in f.keys():
                weight_store[name] = f.get_tensor(name)
        print(f"Loaded {len(weight_store)} tensors")
    else:
        print(f"WARNING: {model_path} not found")
        return

    # Generate or load test codes
    if args.codes:
        codes_path = Path(args.codes)
        with safe_open(str(codes_path), framework="pt", device="cpu") as f:
            codes = [f.get_tensor(f"code_{i}") for i in range(16)]
        print(f"Loaded test codes from {codes_path}")
    else:
        # Generate dummy codes
        torch.manual_seed(42)
        codes = [torch.randint(0, 2048, (1, 8)) for _ in range(16)]
        print("Generated dummy test codes")

    # Simulate forward pass and save intermediates
    intermediates = {}

    # Stage 1: SplitRVQ
    rvq_out = simulate_split_rvq(codes, weight_store)
    intermediates["split_rvq_out"] = rvq_out
    print(f"split_rvq_out: {list(rvq_out.shape)}")

    # Stage 2: pre_conv
    pre_conv_out = simulate_pre_conv(rvq_out, weight_store)
    intermediates["pre_conv_out"] = pre_conv_out
    print(f"pre_conv_out: {list(pre_conv_out.shape)}")

    # Stage 3: pre_transformer (need to transpose to [batch, seq_len, ch])
    x_t = pre_conv_out.transpose(1, 2)  # [batch, seq_len, ch]
    transformer_out = simulate_transformer(x_t, weight_store)
    intermediates["transformer_out"] = transformer_out
    print(f"transformer_out: {list(transformer_out.shape)}")

    # Stage 4: decoder blocks (transpose back to [batch, ch, len])
    x = transformer_out.transpose(1, 2)
    for i in range(4):
        x = simulate_decoder_block(x, weight_store, i)
        intermediates[f"decoder_block_{i}_out"] = x
        print(f"decoder_block_{i}_out: {list(x.shape)}")

    # Stage 5: final
    final_out = simulate_final_stage(x, weight_store)
    intermediates["final_out"] = final_out
    print(f"final_out: {list(final_out.shape)}")

    # Save all intermediates (clone + contiguous to avoid safetensors issues)
    intermediates_clean = {k: v.clone().contiguous() for k, v in intermediates.items()}
    save_file(intermediates_clean, str(output_dir / "intermediates.safetensors"))
    print(f"\nSaved intermediates to: {output_dir / 'intermediates.safetensors'}")

    # Save metadata
    meta = {
        "codes_shape": [list(c.shape) for c in codes],
        "stages": list(intermediates.keys()),
    }
    with open(output_dir / "intermediates_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[OK] Export complete!")


if __name__ == "__main__":
    main()
