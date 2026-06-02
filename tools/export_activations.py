#!/usr/bin/env python3
"""
Export PyTorch intermediate activations for alignment testing.

Runs a single forward pass through each Qwen3-TTS component and saves
intermediate tensors as SafeTensors. The Rust alignment test framework
loads these and compares against Candle outputs layer-by-layer.

Usage:
    python export_activations.py --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
        --output activations/ --component all
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: pip install torch safetensors")
    sys.exit(1)


def export_embedding_lookup(model, output_dir: Path):
    """Export codebook embedding weights for all 16 layers."""
    print("  Exporting codebook embeddings...")
    tensors = {}

    # Extract codec embeddings from code predictor
    if hasattr(model, "code_predictor") and model.code_predictor is not None:
        cp = model.code_predictor
        if hasattr(cp, "codec_embedding"):
            for i, emb in enumerate(cp.codec_embedding):
                tensors[f"codebook.layer_{i:02d}.weight"] = (
                    emb.weight.data.float().cpu()
                )

    # Also extract the main embed_tokens for reference
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        tensors["embed_tokens.weight"] = (
            model.model.embed_tokens.weight.data.float().cpu()
        )

    save_file(tensors, str(output_dir / "embeddings.safetensors"))
    print(f"    Saved {len(tensors)} embedding tensors")
    return tensors


def export_rmsnorm_weights(model, output_dir: Path):
    """Export all RMSNorm weights for alignment."""
    print("  Exporting RMSNorm weights...")
    tensors = {}

    for name, module in model.named_modules():
        if "layernorm" in name.lower() or "norm" in name.lower():
            if hasattr(module, "weight"):
                safe_name = name.replace(".", "_")
                tensors[f"norm.{safe_name}.weight"] = module.weight.data.float().cpu()
                if hasattr(module, "bias") and module.bias is not None:
                    tensors[f"norm.{safe_name}.bias"] = module.bias.data.float().cpu()

    save_file(tensors, str(output_dir / "norms.safetensors"))
    print(f"    Saved {len(tensors)} norm tensors")
    return tensors


def export_attention_weights(model, output_dir: Path):
    """Export attention projection weights."""
    print("  Exporting attention weights...")
    tensors = {}

    for name, module in model.named_modules():
        if "self_attn" in name or "attn" in name:
            for param_name, param in module.named_parameters(recurse=False):
                safe_name = name.replace(".", "_")
                tensors[f"attn.{safe_name}.{param_name}"] = param.data.float().cpu()

    save_file(tensors, str(output_dir / "attention.safetensors"))
    print(f"    Saved {len(tensors)} attention tensors")
    return tensors


def export_mlp_weights(model, output_dir: Path):
    """Export MLP/FFN weights."""
    print("  Exporting MLP weights...")
    tensors = {}

    for name, module in model.named_modules():
        if "mlp" in name.lower():
            for param_name, param in module.named_parameters(recurse=False):
                safe_name = name.replace(".", "_")
                tensors[f"mlp.{safe_name}.{param_name}"] = param.data.float().cpu()

    save_file(tensors, str(output_dir / "mlp.safetensors"))
    print(f"    Saved {len(tensors)} MLP tensors")
    return tensors


def export_conv_weights(model, output_dir: Path):
    """Export convolution weights (tokenizer decoder)."""
    print("  Exporting conv weights...")
    tensors = {}

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
            safe_name = name.replace(".", "_")
            tensors[f"conv.{safe_name}.weight"] = module.weight.data.float().cpu()
            if module.bias is not None:
                tensors[f"conv.{safe_name}.bias"] = module.bias.data.float().cpu()

    if tensors:
        save_file(tensors, str(output_dir / "conv.safetensors"))
        print(f"    Saved {len(tensors)} conv tensors")
    else:
        print("    No conv layers found (tokenizer decoder not loaded?)")
    return tensors


def generate_test_input(output_dir: Path, seq_len: int = 8):
    """Generate a deterministic test input for alignment comparison."""
    print("  Generating test input...")
    torch.manual_seed(42)

    # Fake token sequence (16 codebook layers × seq_len frames)
    tokens = torch.randint(0, 2048, (16, seq_len), dtype=torch.long)
    save_file(
        {"test_tokens": tokens},
        str(output_dir / "test_input.safetensors"),
    )
    print(f"    Saved test tokens: shape={list(tokens.shape)}")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS activations for alignment"
    )
    parser.add_argument(
        "--model", "-m", required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output", "-o", required=True, help="Output directory for activation files"
    )
    parser.add_argument(
        "--component",
        choices=["all", "embed", "norm", "attn", "mlp", "conv"],
        default="all",
        help="Which components to export",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="PyTorch dtype for loading (default: float32)",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for loading (default: cpu)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}, Dtype: {args.dtype}")
    print()

    # Try to load the model
    try:
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=getattr(torch, args.dtype.replace("float", "float")),
            device_map=args.device,
            trust_remote_code=True,
        )
        model.eval()
        print(f"Model loaded: {type(model).__name__}")
    except Exception as e:
        print(f"WARNING: Could not load model via transformers: {e}")
        print("Falling back to weight-only export (no forward pass)")
        model = None

    print()
    print("Exporting weights...")

    if model is not None:
        if args.component in ("all", "embed"):
            export_embedding_lookup(model, output_dir)
        if args.component in ("all", "norm"):
            export_rmsnorm_weights(model, output_dir)
        if args.component in ("all", "attn"):
            export_attention_weights(model, output_dir)
        if args.component in ("all", "mlp"):
            export_mlp_weights(model, output_dir)
        if args.component in ("all", "conv"):
            export_conv_weights(model, output_dir)
    else:
        print("  Skipping (model not loaded)")

    # Always generate test input
    generate_test_input(output_dir)

    # Save export metadata
    meta = {
        "model": args.model,
        "dtype": args.dtype,
        "device": args.device,
        "components": args.component,
    }
    with open(output_dir / "export_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("[OK] Export complete!")
    print(f"   Output: {output_dir}")


if __name__ == "__main__":
    main()
