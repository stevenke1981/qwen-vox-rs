#!/usr/bin/env python3
"""Inspect weight_index.json to understand tensor naming convention."""

import json

with open("weights/converted/weight_index.json") as f:
    data = json.load(f)

# Show unique 3-part prefixes
prefixes = set()
for name in data:
    parts = name.split(".")
    if len(parts) >= 3:
        prefixes.add(".".join(parts[:3]))
    else:
        prefixes.add(name)
print("=== Prefixes ===")
for p in sorted(prefixes):
    print(f"  {p}")

print("\n=== Layer numbers ===")
layer_nums = set()
for n in data:
    for part in n.split("."):
        if part.isdigit():
            layer_nums.add(int(part))
print(f"  {sorted(layer_nums)}")

# Show talker backbone layer names (non-code_predictor)
print("\n=== Backbone layer tensor names (first 3 layers) ===")
backbone_layers = {}
for n, v in data.items():
    if "layers." in n and "code_predictor" not in n:
        for part in n.split("."):
            if part.isdigit():
                layer_num = int(part)
                if layer_num < 3:
                    short = n.replace(f".{layer_num}.", ".{i}.")
                    shape = v["shape"]
                    if layer_num not in backbone_layers:
                        backbone_layers[layer_num] = {}
                    backbone_layers[layer_num][short] = shape
                break

for lnum in sorted(backbone_layers):
    print(f"  Layer {lnum}:")
    for name, shape in sorted(backbone_layers[lnum].items()):
        print(f"    {name}: {shape}")

# Show code_predictor weight names
print("\n=== Code predictor tensor names ===")
cp_names = {}
for n, v in data.items():
    if "code_predictor" in n:
        for part in n.split("."):
            if part.isdigit():
                layer_num = int(part)
                if layer_num < 3:
                    if layer_num not in cp_names:
                        cp_names[layer_num] = {}
                    cp_names[layer_num][n] = v["shape"]
                break
        else:
            cp_names[n] = v["shape"]

for name, shape in sorted(cp_names.items()):
    if isinstance(shape, dict):
        print(f"  Layer {name}:")
        for n, s in sorted(shape.items()):
            print(f"    {n}: {s}")
    else:
        print(f"  {name}: {shape}")

# Show special tensors
print("\n=== Special tensors ===")
special_keys = [
    "text_embedding",
    "codec_embedding",
    "codec_head",
    "norm",
    "embed_tokens",
    "text_projection",
    "small_to_mtp",
]
for name, v in sorted(data.items()):
    for sk in special_keys:
        if sk in name:
            print(f"  {name}: {v['shape']}")
            break

# Show tensor count and total params
total_params = sum(v["numel"] for v in data.values())
print(f"\n=== Stats ===")
print(f"  Total tensors: {len(data)}")
print(f"  Total params: {total_params:,}")
print(f"  ~Size: {total_params * 4 / 1024**3:.2f} GB (f32)")
