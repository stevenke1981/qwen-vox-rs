#!/usr/bin/env python3
"""
Export intermediate activations from Qwen3-TTS-Tokenizer-12Hz for Rust alignment verification.

This script runs a forward pass through the PyTorch tokenizer and saves intermediate
activations at each stage for comparison with the Rust implementation.
"""

import argparse
import os
import sys

try:
    import torch
    from safetensors.torch import save_file
except ImportError:
    print("ERROR: torch/safetensors not installed. Run: pip install torch safetensors")
    sys.exit(1)

try:
    from qwen_tts import Qwen3TTSTokenizer
except ImportError:
    print("ERROR: qwen-tts not installed. Run: pip install qwen-tts")
    sys.exit(1)

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Export intermediate activations from Qwen3-TTS tokenizer"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="weights/intermediates/intermediates.safetensors",
        help="Output path for intermediates",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        help="Model ID or local path",
    )
    parser.add_argument("--device", default="cpu", help="Device to run on")
    args = parser.parse_args()

    os.environ["HF_HOME"] = os.path.abspath(".cache/huggingface")

    print(f"Loading tokenizer from: {args.model}")
    tokenizer = Qwen3TTSTokenizer.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float32,
    )
    print("Tokenizer loaded!")

    # Create test audio (sine wave at 440Hz for 1 second)
    sr = 24000
    t = np.linspace(0, 1, sr, endpoint=False)
    test_audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # Encode
    print("Encoding test audio...")
    codes = tokenizer.encode(test_audio, sr)
    codes_tensor = codes.audio_codes[0]  # [13, 16]
    print(f"Codes shape: {codes_tensor.shape}")

    # Get the internal model
    model = tokenizer.model

    # Run through decoder step by step
    print("Running decoder forward pass...")

    decoder = model.decoder

    # 1. SplitRVQ quantizer
    print("  Stage 1: SplitRVQ quantizer...")
    quantizer_out = decoder.quantizer.decode(codes_tensor)  # [B, 512, T]
    print(f"    quantizer_out shape: {quantizer_out.shape}")

    # 2. Pre-conv
    print("  Stage 2: Pre-conv...")
    pre_conv_out = decoder.pre_conv(quantizer_out)  # [B, 1024, T]
    print(f"    pre_conv_out shape: {pre_conv_out.shape}")

    # 3. Pre-transformer
    print("  Stage 3: Pre-transformer...")
    # Transpose for transformer: [B, C, T] → [B, T, C]
    pt_in = pre_conv_out.transpose(1, 2)
    pt_out = decoder.pre_transformer(pt_in)  # [B, T, 1024]
    pt_out = pt_out.transpose(1, 2)  # [B, 1024, T]
    print(f"    pre_transformer_out shape: {pt_out.shape}")

    # 4. Upsample
    print("  Stage 4: Upsample...")
    up_out = pt_out
    for i, stage in enumerate(decoder.upsample):
        up_out = stage(up_out)
        print(f"    upsample stage {i} shape: {up_out.shape}")

    # 5. Decoder blocks
    print("  Stage 5: Decoder blocks...")
    dec_out = up_out
    for i, block in enumerate(decoder.decoder):
        dec_out = block(dec_out)
        print(f"    decoder_block_{i}_out shape: {dec_out.shape}")

    # 6. Final snake + conv
    print("  Stage 6: Final conv...")
    final_snake_out = decoder.final_snake(dec_out)
    final_out = decoder.final_conv(final_snake_out)
    print(f"    final_out shape: {final_out.shape}")

    # Clamp to [-1, 1]
    final_out = final_out.clamp(-1, 1)

    # Save intermediates
    intermediates = {
        "split_rvq_out": quantizer_out.detach().cpu(),
        "pre_conv_out": pre_conv_out.detach().cpu(),
        "transformer_out": pt_out.detach().cpu(),
        "upsample_out": up_out.detach().cpu(),
        "decoder_block_0_out": decoder.decoder[0](up_out).detach().cpu()
        if len(decoder.decoder) > 0
        else None,
        "final_out": final_out.detach().cpu(),
    }

    # Remove None entries
    intermediates = {k: v for k, v in intermediates.items() if v is not None}

    # Also save the codes
    intermediates["codes"] = codes_tensor.detach().cpu()

    # Save to file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_file(intermediates, args.output)
    print(f"\nSaved intermediates to: {args.output}")

    # Print summary
    print("\nIntermediate shapes:")
    for name, tensor in intermediates.items():
        print(f"  {name}: {list(tensor.shape)}")


if __name__ == "__main__":
    main()
