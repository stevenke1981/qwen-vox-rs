# Qwen3-TTS / CosyVoice 3 Minnan Hokkien Guide

This note records the Minnan / Hokkien / Taiwan Taigi usage target for this
Rust rewrite. The repository currently implements Qwen3-TTS generation; the
CosyVoice 3 examples are included as feature references, not as an implemented
backend.

## Support Summary

| Item | Qwen3-TTS | Fun-CosyVoice 3.0 |
| --- | --- | --- |
| Minnan support | Explicit Hokkien support | Supports Minnan as one of many dialects |
| Best control style | Voice Design + Instruct | Natural-language Instruct |
| Recommended text | Taiwan Taigi common writing, mixed Han characters and colloquial words | Same; can also transform Mandarin text through instruct |
| Voice clone | 3-second reference audio, cross-language and cross-dialect | 3-second zero-shot reference audio with natural emotion |
| Best use case | Audiobooks, multi-role voices, precise dialect control | Realtime dialogue, natural speech, emotional style |

## Qwen3-TTS Minnan Examples

### Voice Design

Voice Design is the strongest target path for dialect and role control.

```python
from qwen_tts import Qwen3TTSModel
import soundfile as sf
import torch

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

text = "阿公啊，今仔日天氣真好，咱去海邊行一遭好無？"

wavs, sr = model.generate_voice_design(
    text=text,
    language="Chinese",
    instruct=(
        "用純正台灣閩南話（台語）說這句話，60歲左右溫暖慈祥的阿公聲音，"
        "帶點南部腔調、語速自然緩慢、語氣親切溫暖，"
        "帶有明顯台語語調起伏和鼻音。"
    ),
)

sf.write("taiwanese_grandpa.wav", wavs[0], sr)
```

Useful instruct templates:

- `用道地高雄閩南話，30歲年輕女性，活潑甜美，語速稍快，帶笑聲。`
- `用台中腔台語，50歲男性，穩重低沉，帶點感慨語氣。`
- `可愛小女孩台語，8歲左右，奶聲奶氣，拖長尾音，興奮地說。`

### CustomVoice With Dialect Prompt

CustomVoice uses built-in speaker presets. If the selected preset is suitable,
use an instruct-style prompt to strengthen Minnan delivery:

```powershell
.\dist\qwen-vox-cuda.exe generate `
  --device cuda `
  --language chinese `
  --speaker vivian `
  --text "阿公啊，今仔日天氣真好，咱去海邊行一遭好無？" `
  --output out\taiwanese_customvoice.wav
```

Current Rust status: the CLI supports CustomVoice preset generation. A dedicated
`--instruct` or Voice Design path is still future work.

### Voice Clone

Voice clone gives the most realistic speaker identity when given a clean
3-10 second reference clip plus matching transcript.

```powershell
.\dist\qwen-vox-cuda.exe clone `
  --model-dir weights\model-0.6b `
  --device cuda `
  --ref-audio refs\taiwanese_reference.wav `
  --ref-text "今仔日天氣真好，咱來去食飯。" `
  --text "阿公啊，今仔日天氣真好，咱去海邊行一遭好無？" `
  --output out\taiwanese_clone.wav
```

Current Rust status: the clone command validates official Base-model
requirements. Speaker encoder support and dynamic 0.6B talker loading are the
next implementation steps.

## CosyVoice 3 Minnan Reference

CosyVoice 3 is useful as a comparison target for natural, realtime Minnan
speech. It is not implemented in this repository.

```python
for i, item in enumerate(cosyvoice.inference_instruct2(
    text="明仔載咱要去哪裡跈？想食牛肉湯還是蚵仔煎？",
    instruct=(
        "用台灣閩南話（台語）說這句話，要用溫柔親切的女性聲音，"
        "帶點北部腔，語氣開心自然。"
    ),
    prompt_speech_16k=ref_audio,
    stream=False,
)):
    torchaudio.save(f"minnan_{i}.wav", item["tts_speech"], cosyvoice.sample_rate)
```

Strong instruct examples:

- `用純正台灣台語，帶台南腔，老年男性，語速緩慢，語氣懷舊。`
- `年輕女生台語，甜美可愛，語速中等，帶撒嬌感覺。`
- `用台語大聲生氣地說這句話，男性聲音。`

## Practical Minnan Tips

- Prefer Taiwan Taigi common words: `今仔日`, `明仔載`, `食飯`, `跈`, `幾好`,
  `攏總`.
- Mandarin text can work when the instruct explicitly asks for Taiwan Taigi, but
  native Minnan wording usually sounds better.
- Specify region and accent: Taiwan Taigi, Tainan, Kaohsiung, Taipei, Taichung,
  Kinmen.
- Specify age, gender, speaking speed, emotion, and personality.
- For long text, split into 50-150 character sentences and keep the same
  instruct/reference audio for voice consistency.
- Mixed Mandarin and Minnan can be described in one instruct, for example:
  `先用台語說前半句，再用國語說後半句。`

## Rust Feature Mapping

Planned Qwen3-TTS feature mapping for this repository:

| Official feature | Rust CLI target | Status |
| --- | --- | --- |
| CustomVoice preset speaker | `generate --speaker vivian` | Implemented |
| Reproducible generation | `generate --seed 42` | Implemented |
| Lightweight duration scaling | `generate --speed 1.1` | Implemented |
| Base voice clone | `clone --ref-audio --ref-text` | Scaffolded |
| Voice Design / instruct | `generate --instruct ...` | Planned |
| Minnan presets/examples | Documentation and examples | This guide |
