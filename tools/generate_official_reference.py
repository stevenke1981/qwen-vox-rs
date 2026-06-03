"""Generate an official Qwen3-TTS reference WAV from local HF weights.

This script is intentionally a Python reference path. It loads the upstream
Qwen3-TTS classes from files downloaded into ./out and local weights from
weights/hf_original. Use it to produce reference audio/code frames for the Rust
implementation; do not use it as the Rust runtime path.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib
import sys
import types

import numpy as np
import soundfile as sf
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoFeatureExtractor, AutoModel, AutoTokenizer


def load_module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install_package_namespace(source_dir: pathlib.Path):
    for name in [
        "qwen_tts",
        "qwen_tts.core",
        "qwen_tts.core.models",
        "qwen_tts.core.tokenizer_12hz",
        "qwen_tts.inference",
    ]:
        mod = types.ModuleType(name)
        mod.__path__ = [str(source_dir)]
        sys.modules[name] = mod


def load_official_classes(source_dir: pathlib.Path):
    install_package_namespace(source_dir)

    v2_conf = load_module(
        "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
        source_dir / "configuration_qwen3_tts_tokenizer_v2.py",
    )
    v2_model = load_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2",
        source_dir / "modeling_qwen3_tts_tokenizer_v2.py",
    )

    class Qwen3TTSTokenizer:
        def __init__(self):
            self.model = None
            self.feature_extractor = None
            self.config = None
            self.device = None

        @classmethod
        def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
            inst = cls()
            AutoConfig.register(
                "qwen3_tts_tokenizer_12hz",
                v2_conf.Qwen3TTSTokenizerV2Config,
                exist_ok=True,
            )
            AutoModel.register(
                v2_conf.Qwen3TTSTokenizerV2Config,
                v2_model.Qwen3TTSTokenizerV2Model,
                exist_ok=True,
            )
            inst.feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name_or_path
            )
            filtered = {k: v for k, v in kwargs.items() if k not in ("device_map",)}
            inst.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path, **filtered
            )
            inst.config = inst.model.config
            try:
                inst.device = next(inst.model.parameters()).device
            except StopIteration:
                inst.device = torch.device("cpu")
            return inst

        def decode(self, encoded):
            if isinstance(encoded, list):
                codes = [e["audio_codes"] for e in encoded]
            elif isinstance(encoded, dict):
                codes = encoded["audio_codes"]
            else:
                codes = encoded.audio_codes

            if isinstance(codes, torch.Tensor):
                audio_codes = codes.unsqueeze(0) if codes.dim() == 2 else codes
            else:
                audio_codes = pad_sequence(
                    [c.to(torch.long) for c in codes],
                    batch_first=True,
                    padding_value=0,
                )
            audio_codes = audio_codes.to(self.device)
            with torch.inference_mode():
                wavs = self.model.decode(audio_codes, return_dict=False)[0]
            wavs = [
                w.detach().float().cpu().numpy().astype(np.float32) for w in wavs
            ]
            sr = int(getattr(self.feature_extractor, "sampling_rate", 24000))
            return wavs, sr

    tokenizer_stub = types.ModuleType("qwen_tts.inference.qwen3_tts_tokenizer")
    tokenizer_stub.Qwen3TTSTokenizer = Qwen3TTSTokenizer
    sys.modules["qwen_tts.inference.qwen3_tts_tokenizer"] = tokenizer_stub

    main_conf = load_module(
        "qwen_tts.core.models.configuration_qwen3_tts",
        source_dir / "configuration_qwen3_tts.py",
    )
    processing = load_module(
        "qwen_tts.core.models.processing_qwen3_tts",
        source_dir / "processing_qwen3_tts.py",
    )
    main_model = load_module(
        "qwen_tts.core.models.modeling_qwen3_tts",
        source_dir / "modeling_qwen3_tts.py",
    )
    return main_conf, processing, main_model


def build_assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="chinese")
    parser.add_argument("--speaker", default="vivian")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--weights-dir", default="weights/hf_original")
    parser.add_argument("--source-dir", default="out")
    parser.add_argument("--output", default="out/official_qwen3_reference.wav")
    parser.add_argument("--codes-output", default="")
    parser.add_argument("--argmax", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir).resolve()
    weights_dir = pathlib.Path(args.weights_dir).resolve()
    output = pathlib.Path(args.output).resolve()

    main_conf, processing, main_model = load_official_classes(source_dir)

    AutoConfig.register("qwen3_tts", main_conf.Qwen3TTSConfig, exist_ok=True)
    AutoModel.register(
        main_conf.Qwen3TTSConfig,
        main_model.Qwen3TTSForConditionalGeneration,
        exist_ok=True,
    )

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print({"device": device, "dtype": str(dtype), "weights_dir": str(weights_dir)})

    model = main_model.Qwen3TTSForConditionalGeneration.from_pretrained(
        str(weights_dir),
        torch_dtype=dtype,
        attn_implementation="eager",
        local_files_only=True,
    )
    model.eval()
    if device == "cuda":
        model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), local_files_only=True)
    processor = processing.Qwen3TTSProcessor(tokenizer=tokenizer)

    encoded = processor(
        text=build_assistant_text(args.text),
        return_tensors="pt",
        padding=True,
    )
    input_id = encoded["input_ids"].to(model.device)
    input_ids = [input_id.unsqueeze(0) if input_id.dim() == 1 else input_id]
    print(
        {
            "input_tokens": int(input_ids[0].numel()),
            "speaker": args.speaker,
            "language": args.language,
        }
    )

    if args.argmax:
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "subtalker_dosample": False,
            "repetition_penalty": 1.0,
        }
    else:
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.9,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "repetition_penalty": 1.05,
        }
    print({"generation": generation_kwargs})

    with torch.inference_mode():
        talker_codes_list, _ = model.generate(
            input_ids=input_ids,
            instruct_ids=[None],
            languages=[args.language],
            speakers=[args.speaker],
            non_streaming_mode=True,
            **generation_kwargs,
        )

    print({"generated_frames": [tuple(c.shape) for c in talker_codes_list]})
    if args.codes_output:
        codes_path = pathlib.Path(args.codes_output).resolve()
        codes_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(codes_path, talker_codes_list[0].detach().cpu().numpy())
        print({"codes_output": str(codes_path)})

    wavs, sr = model.speech_tokenizer.decode(
        [{"audio_codes": c} for c in talker_codes_list]
    )
    wav = wavs[0].astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), wav, sr)

    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0
    clip = int(np.sum(np.abs(wav) >= 0.9999))
    zc = int(np.sum((wav[:-1] < 0) != (wav[1:] < 0))) if wav.size > 1 else 0
    print(
        {
            "out": str(output),
            "sr": int(sr),
            "samples": int(wav.size),
            "duration": wav.size / sr,
            "min": float(wav.min()) if wav.size else 0.0,
            "max": float(wav.max()) if wav.size else 0.0,
            "rms_db": 20 * math.log10(max(rms, 1e-12)),
            "clip": clip,
            "zc_rate": zc / max(1, wav.size),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
