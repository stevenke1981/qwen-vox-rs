"""Check safetensors key structure for each weight file."""

import json, struct
from pathlib import Path

weight_dir = Path("weights")

files = [
    # "model-0.6b/model.safetensors",   # too big
    "converted/model.safetensors",
    "alignments/talker_layers.safetensors",
    "alignments/code_predictor.safetensors",
    "alignments/speaker_encoder.safetensors",
    "alignments/tokenizer_decoder.safetensors",
]

for fname in files:
    path = weight_dir / fname
    if not path.exists():
        print(f"\n{fname}: NOT FOUND")
        continue
    data = path.read_bytes()
    header_len = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8 : 8 + header_len])
    print(f"\n{fname} ({path.stat().st_size / 1024**3:.1f} GB)")
    for k, v in sorted(header.items()):
        shape_str = "x".join(str(s) for s in v["shape"])
        print(f"  {k}: {shape_str} {v['dtype']}")
