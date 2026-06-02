"""Check safetensors key structure for each weight file."""

import json, struct
from pathlib import Path

files = [
    "weights/converted/model.safetensors",
    "weights/alignments/talker_layers.safetensors",
    "weights/alignments/code_predictor.safetensors",
    "weights/alignments/speaker_encoder.safetensors",
    "weights/alignments/tokenizer_decoder.safetensors",
]

for path_str in files:
    path = Path(path_str)
    if not path.exists():
        print(f"\n{path_str}: NOT FOUND")
        continue
    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_len])
    print(f"\n{path_str} ({path.stat().st_size / 1024**3:.1f} GB, {len(header)} keys)")
    for k, v in sorted(header.items()):
        try:
            shape_str = "x".join(str(s) for s in v["shape"])
            print(f"  {k}: {shape_str} {v['dtype']}")
        except Exception as e:
            print(f"  {k}: ERROR - {e} - keys={list(v.keys())}")
