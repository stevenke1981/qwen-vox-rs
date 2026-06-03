"""Compare official Qwen3-TTS codec frames against Rust-generated frames."""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np


def load_rust_frames(path: pathlib.Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as f:
        payload: Any = json.load(f)
    frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
    arr = np.asarray(frames, dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"{path} must contain a [frames, 16] codec matrix, got {arr.shape}")
    return arr


def load_official_frames(path: pathlib.Path) -> np.ndarray:
    arr = np.asarray(np.load(path), dtype=np.int64)
    if arr.ndim != 2 or arr.shape[1] != 16:
        raise ValueError(f"{path} must contain a [frames, 16] codec matrix, got {arr.shape}")
    return arr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", required=True, help="Official .npy codec frames")
    parser.add_argument("--rust", required=True, help="Rust JSON codec-frame dump")
    parser.add_argument("--show", type=int, default=8, help="Rows to print around the first mismatch")
    args = parser.parse_args()

    official = load_official_frames(pathlib.Path(args.official))
    rust = load_rust_frames(pathlib.Path(args.rust))
    common = min(len(official), len(rust))

    print(
        {
            "official_shape": tuple(official.shape),
            "rust_shape": tuple(rust.shape),
            "common_frames": common,
        }
    )
    if common == 0:
        return 1

    official_common = official[:common]
    rust_common = rust[:common]
    frame_matches = np.all(official_common == rust_common, axis=1)
    codebook_matches = np.mean(official_common == rust_common, axis=0)
    q0_matches = official_common[:, 0] == rust_common[:, 0]

    first_mismatch = None
    mismatch_indices = np.flatnonzero(~frame_matches)
    if mismatch_indices.size:
        first_mismatch = int(mismatch_indices[0])

    print(
        {
            "matching_frames": int(frame_matches.sum()),
            "frame_match_rate": float(frame_matches.mean()),
            "q0_match_rate": float(q0_matches.mean()),
            "first_mismatch": first_mismatch,
        }
    )
    print(
        {
            f"q{i}_match_rate": float(rate)
            for i, rate in enumerate(codebook_matches)
        }
    )

    head = min(args.show, common)
    print({"official_q0_head": official_common[:head, 0].tolist()})
    print({"rust_q0_head": rust_common[:head, 0].tolist()})

    if first_mismatch is not None:
        end = min(common, first_mismatch + max(1, args.show))
        for i in range(first_mismatch, end):
            print(
                {
                    "frame": i,
                    "official": official_common[i].tolist(),
                    "rust": rust_common[i].tolist(),
                    "equal": bool(frame_matches[i]),
                }
            )

    return 0 if frame_matches.all() and len(official) == len(rust) else 2


if __name__ == "__main__":
    raise SystemExit(main())
