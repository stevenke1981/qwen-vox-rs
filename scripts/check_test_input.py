"""Check test_input.safetensors structure."""

import json, struct
from pathlib import Path

path = Path("weights/alignments/test_input.safetensors")
data = path.read_bytes()

header_len = struct.unpack("<Q", data[:8])[0]
header = json.loads(data[8 : 8 + header_len])
print("Keys in test_input:", list(header.keys()))
for k, v in header.items():
    print(
        f"  {k}: shape={v['shape']}, dtype={v['dtype']}, data_offsets={v['data_offsets']}"
    )
