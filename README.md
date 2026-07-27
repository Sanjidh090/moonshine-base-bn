# Moonshine-Base-BN: Tokenizer Transplantation for Bengali ASR

Code and experiments for **"Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR"**, accepted at the [MusIML Workshop, ICML 2026](sanjidh090.github.io/moonshine-base-bn).

Paper: [https://arxiv.org/abs/2607.09598](https://arxiv.org/abs/2607.09598)

This repo adapts [Moonshine-Base](https://huggingface.co/UsefulSensors/moonshine-base) (originally English, GPT-2 BPE tokenizer) to Bengali by transplanting a WordPiece vocabulary from [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert), then fine-tuning on a 1.4M-sample unified Bengali speech corpus. The result is an edge-efficient Bengali ASR model — see the trained checkpoint and full model card on [Hugging Face](https://huggingface.co/Sanjidh090/moonshine-base-bn).

# Project Page: [https://sanjidh090.github.io/moonshine-base-bn/](https://sanjidh090.github.io/moonshine-base-bn/)
<img width="1774" height="848" alt="ChatGPT Image Jun 22, 2026, 02_14_28 AM" src="https://github.com/user-attachments/assets/d9e3082d-eed9-4bba-a0a8-4df022739bed" />

## Results

Evaluated on the held-out test split of [Lipi-Ghor-bn-882-SSTT](https://huggingface.co/datasets/Sanjidh090/Lipi-Ghor-bn-882-SSTT):

| Model | Params | WER (%) ↓ | CER (%) ↓ | RTF ↓ |
|---|---|---|---|---|
| Seamless M4T-v2 (zero-shot) | ~2.3B | 66.71 | 45.54 | — |
| Whisper large-v3 (zero-shot) | ~1.55B | 84.53 | 72.92 | — |
| Meta MMS 1B (zero-shot) | ~1B | 43.06 | 21.16 | — |
| Hishab TITU Conformer Large (zero-shot) | ~120M | 30.51 | 18.23 | — |
| Conformer Baseline (fine-tuned) | ~120M | 24.67 | 15.56 | 0.0120 |
| Faster Whisper Medium (fine-tuned) | ~769M | 21.28 | 11.18 | 0.0190 |
| **Moonshine-Base-BN (this work)** | **~61.5M** | **21.54** | **10.79** | **0.0053** |

> RTF measured on `<CPU/GPU model, precision — fill in>`. Lowest CER of all tested models; matches the WER of a fine-tuned model ~12x its size while running ~3.5x faster than the Faster Whisper Medium CTranslate2 pipeline.

Tokenizer transplantation reduced token fertility from **9.16 → 1.30** tokens/word, directly addressing the autoregressive collapse (repetition/hallucination loops) that high-fertility byte-level tokenizers cause on non-Latin scripts.

## Repository structure

```
.
├── dataset_preparation.md              # Audio chunking / WAV conversion scripts (ffmpeg & pydub variants)
├── tokenizer_fertility_check.py        # Compares Moonshine's original tokenizer vs. BanglaBERT WordPiece on sample Bengali text
├── Vanilla_Baseline_Training.ipynb     # Baseline: Moonshine-Base fine-tuned on Bengali with its original tokenizer (no transplant)
└── tokenizer_replacing_&_finetuning.ipynb  # Core method: vocabulary transplant + fine-tuning pipeline
```

## Method summary

1. **Tokenizer transplant** — replace Moonshine's GPT-2 BPE vocabulary with BanglaBERT's 50k WordPiece vocabulary, re-initializing the embedding/output layers for the new vocab while keeping the encoder-decoder backbone.
2. **Baseline check** — `Vanilla_Baseline_Training.ipynb` establishes what fine-tuning Moonshine on Bengali looks like *without* the transplant, for comparison.
3. **Recovery fine-tuning** — `tokenizer_replacing_&_finetuning.ipynb` fine-tunes the transplanted model on the unified Bengali corpus (LR=5e-5, warmup=2000 steps, early stopping patience=5, checkpoints every 5 epochs).

## Data

Training uses a unified 1.43M-sample Bengali speech corpus (filtered to 1.0–30.0s clips), assembled in part from [Lipi-Ghor-bn-882-SSTT](https://huggingface.co/datasets/Sanjidh090/Lipi-Ghor-bn-882-SSTT) (882 hours, diarized YouTube-sourced Bengali speech) alongside other public Bengali ASR datasets. See `dataset_preparation.md` for the audio preprocessing pipeline — **note:** the scripts there use local Windows paths as an example; edit the `SOURCE_DIR` / `OUTPUT_DIR` variables (or pass them as CLI args) before running.

## Setup

```bash
pip install -r requirements.txt   # <-- add this file: transformers, torch, soundfile, pydub, tqdm, numpy, yt-dlp
```

`dataset_preparation.md`'s scripts additionally require `ffmpeg` on your `PATH`.

## Quickstart: inference

```python
# pip install torch librosa numpy transformers

import torch
import librosa
import numpy as np
from transformers import AutoTokenizer, AutoModelForSpeechSeq2Seq

REPO_ID = "Sanjidh090/moonshine-base-bn"
AUDIO_PATH = "sample.wav"  # replace with your own .wav file

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    REPO_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
).to(device)
model.eval()

# Fallback to verified defaults if the tokenizer config doesn't set these
START_ID = tokenizer.cls_token_id or 2
EOS_ID   = tokenizer.sep_token_id or 3
PAD_ID   = tokenizer.pad_token_id or 0


def transcribe(audio_file):
    audio, _ = librosa.load(audio_file, sr=16000)

    # Pad to a multiple of 320 samples
    remainder = len(audio) % 320
    if remainder:
        audio = np.concatenate([audio, np.zeros(320 - remainder, dtype=np.float32)])

    input_values = torch.tensor(audio).unsqueeze(0).to(device, dtype=dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_values,
            max_new_tokens=2000,
            num_beams=5,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            decoder_start_token_id=START_ID,
            pad_token_id=PAD_ID,
            eos_token_id=EOS_ID
        )

    transcription = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    return transcription


if __name__ == "__main__":
    print(transcribe(AUDIO_PATH))
```

Full usage details, the trained weights, and a live demo Space are on the [Hugging Face model card](https://huggingface.co/Sanjidh090/moonshine-base-bn) ([Space](https://huggingface.co/spaces/Sanjidh090/Moonshine-base-bn-demo)).

## Known limitations

- Domain mismatch between training data (YouTube-sourced) and formal/telephony speech leads to a WER gap versus curated benchmarks.
- Codec compression (as encountered in real-device ONNX deployment) degrades accuracy; not yet corrected for in training.
- BanglaBERT's WordPiece tokenizer emits `[UNK]` for some common Bengali words — this is being investigated as it may understate true fertility gains (see `tokenizer_fertility_check.py`).

## Citation

```bibtex
@inproceedings{hasan2026tokenizer,
  title     = {Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR},
  author    = {Hasan, Sanjid and Rahman, Md. Abdur},
  booktitle = {MusIML Workshop, ICML},
  year      = {2026}
}
```

## See it on arxiv
```bibtex
@misc{hasan2026tokenizertransplantationmitigatingautoregressive,
      title={Tokenizer Transplantation: Mitigating Autoregressive Collapse in Edge-Efficient Bengali ASR}, 
      author={Sanjid Hasan and Md. Abdur Rahman},
      year={2026},
      eprint={2607.09598},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.09598}, 
}
```
## Acknowledgments

Built on [UsefulSensors/Moonshine](https://github.com/usefulsensors/moonshine) and [BanglaBERT](https://huggingface.co/csebuetnlp/banglabert) (CSEBUETNLP). Trained on [Lipi-Ghor-bn-882-SSTT](https://huggingface.co/datasets/Sanjidh090/Lipi-Ghor-bn-882-SSTT), with GPU support from the Department of CSE at Khulna University of Engineering & Technology (KUET).

## License

MIT — see [LICENSE](LICENSE).
