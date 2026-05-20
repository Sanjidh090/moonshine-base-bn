# The dataset in Lipighor 882 hour!

but first let's download lipighor from huggingface.
```python
git lfs install

git clone https://huggingface.co/datasets/Sanjidh090/Lipi-Ghor-bn-882-SSTT
```
then you get files in mp3 format...do some setup and installations and do install audio libraries such as ffmpeg...

then ,
```
"""
mp3_to_wav_chunker.py
=====================
Strategy:
  1. Convert each full MP3 → temporary 16kHz mono WAV  (fixes the ffmpeg error code 69)
  2. Slice the WAV using start/end times from metadata.csv
  3. Write chunked WAVs + clean metadata for training

Input:
  - F:\Dataset\Lipi-Ghor-bn-882-SSTT\metadata.csv
  - F:\Dataset\Lipi-Ghor-bn-882-SSTT\data\<video_id>.mp3

Output:
  - F:\Dataset\lipi-ghor-training-ready\wavs_asr_chunks\wavs\<chunk>.wav
  - F:\Dataset\lipi-ghor-training-ready\wavs_asr_chunks\metadata.csv
  - F:\Dataset\lipi-ghor-training-ready\lipi-ghor\train.tsv / dev.tsv / test.tsv
"""

import csv
import os
import random
import re
import shutil
import subprocess
import unicodedata
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
from tqdm import tqdm

# ── FFmpeg Configuration ──────────────────────────────────────────────────────
venv_bin = Path(r"f:\Dataset\venv\Scripts")
FFMPEG   = str(venv_bin / "ffmpeg.exe")
FFPROBE  = str(venv_bin / "ffprobe.exe")
os.environ["PATH"] += os.pathsep + str(venv_bin)
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
DATASET_ROOT  = Path(r"F:\Dataset\Lipi-Ghor-bn-882-SSTT")
OUTPUT_ROOT   = Path(r"F:\Dataset\lipi-ghor-training-ready")
METADATA_CSV  = DATASET_ROOT / "metadata.csv"
DATA_DIR      = DATASET_ROOT / "data"
TEMP_WAV_DIR  = OUTPUT_ROOT / "temp_full_wavs"   # full converted WAVs (can delete after)

SPLIT       = {"train": 0.90, "dev": 0.05, "test": 0.05}
RANDOM_SEED = 42

# Audio
SAMPLE_RATE = 16_000
MIN_DUR     = 0.0
MAX_DUR     = 30.0
RMS_GATE    = 0.001

# Text filters
MIN_CHARS    = 5
MIN_WORDS    = 3
BN_RATIO_MIN = 0.50
BN_RANGE     = (0x0980, 0x09FF)
ALLOWED_PUNCT = "।॥?!,;:—-\u200c\u200d"
# ═══════════════════════════════════════════════════════════════════════════════


# ── Text helpers ──────────────────────────────────────────────────────────────

def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    text = nfc(str(text)).strip()
    text = re.sub(
        r"[^\u0980-\u09FF\u200c\u200da-zA-Z0-9\s" + re.escape(ALLOWED_PUNCT) + r"]",
        "",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def is_quality_text(text: str) -> bool:
    if len(text) < MIN_CHARS:
        return False
    if len(text.split()) < MIN_WORDS:
        return False
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    bn = sum(1 for c in alpha if BN_RANGE[0] <= ord(c) <= BN_RANGE[1])
    return (bn / len(alpha)) >= BN_RATIO_MIN


# ── Audio helpers ─────────────────────────────────────────────────────────────

def convert_mp3_to_wav(mp3_path: Path, wav_path: Path) -> bool:
    """
    Convert a full MP3 → 16kHz mono WAV using ffmpeg directly.
    This bypasses pydub entirely and avoids the error code 69 bug.
    Returns True on success.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                FFMPEG,
                "-y",                          # overwrite
                "-analyzeduration", "50000000",
                "-probesize", "50000000",
                "-i", str(mp3_path),
                "-ar", str(SAMPLE_RATE),       # 16kHz
                "-ac", "1",                    # mono
                "-sample_fmt", "s16",          # 16-bit PCM
                str(wav_path),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print(f"\n[WARN] MP3→WAV failed for {mp3_path.name}")
            print(result.stderr[-500:])        # show last 500 chars of error
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"\n[WARN] Timeout converting {mp3_path.name}")
        return False
    except Exception as e:
        print(f"\n[WARN] Error converting {mp3_path.name}: {e}")
        return False


def slice_wav(full_wav: Path, start_s: float, end_s: float, out_wav: Path) -> bool:
    """
    Slice a chunk from an already-converted WAV file using soundfile.
    Pure Python — no ffmpeg involved at slice time.
    """
    try:
        # Read only the needed samples (fast, no full-file load)
        start_frame = int(start_s * SAMPLE_RATE)
        end_frame   = int(end_s   * SAMPLE_RATE)

        with sf.SoundFile(str(full_wav)) as f:
            f.seek(start_frame)
            frames = end_frame - start_frame
            audio  = f.read(frames, dtype="float32", always_2d=False)

        if audio.size == 0:
            return False

        sf.write(str(out_wav), audio, SAMPLE_RATE, subtype="PCM_16")
        return True
    except Exception as e:
        print(f"\n[WARN] Slice failed {full_wav.name} [{start_s:.2f}–{end_s:.2f}s]: {e}")
        return False


def rms_of_wav(wav_path: Path) -> float:
    try:
        audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return float(np.sqrt(np.mean(audio ** 2)))
    except Exception:
        return 0.0


def redownload_mp3(video_id: str, mp3_path: Path) -> bool:
    """Redownload from YouTube via yt-dlp if MP3 conversion fails."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    print(f"\n[INFO] Attempting yt-dlp redownload: {video_id} …")
    out_template = str(mp3_path.parent / f"{video_id}.%(ext)s")
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--extract-audio",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "--ffmpeg-location", str(venv_bin),
                "-o", out_template,
                "--no-playlist",
                "--force-overwrites",
                url,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            err = result.stderr.strip().splitlines()
            print(f"[SKIP] yt-dlp failed: {err[-1] if err else 'unknown'}")
            return False
        print(f"[OK]   Downloaded: {mp3_path.name}")
        return mp3_path.exists()
    except subprocess.TimeoutExpired:
        print(f"[SKIP] yt-dlp timeout for {video_id}")
        return False
    except FileNotFoundError:
        raise SystemExit("yt-dlp not found. Run:  pip install yt-dlp")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    random.seed(RANDOM_SEED)

    wavs_dir  = OUTPUT_ROOT / "wavs_asr_chunks" / "wavs"
    meta_dir  = OUTPUT_ROOT / "wavs_asr_chunks"
    lipi_dir  = OUTPUT_ROOT / "lipi-ghor"
    audio_dir = lipi_dir / "audio"

    for d in [wavs_dir, meta_dir, lipi_dir, audio_dir, TEMP_WAV_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Read metadata.csv ─────────────────────────────────────────────
    print("\n[1/5] Reading metadata.csv …")
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata.csv not found at {METADATA_CSV}")

    # Group rows by video_id so we convert each MP3 once
    by_video = defaultdict(list)
    skipped_text = 0

    with open(METADATA_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                video_id = row["video_id"].strip()
                chunk_id = int(row["chunk_id"])
                start    = float(row["start"])
                end      = float(row["end"])
                duration = float(row["duration"])
                text_raw = row["text"].strip()
            except (KeyError, ValueError):
                continue

            # Duration gate
            if not (MIN_DUR <= duration <= MAX_DUR):
                continue

            # Text quality gate
            text = clean_text(text_raw)
            if not is_quality_text(text):
                skipped_text += 1
                continue

            fname = f"{video_id}_chunk_{chunk_id:04d}.wav"
            by_video[video_id].append({
                "video_id": video_id,
                "chunk_id": chunk_id,
                "fname":    fname,
                "start":    start,
                "end":      end,
                "duration": duration,
                "text":     text,
            })

    total_segs = sum(len(v) for v in by_video.values())
    print(f"   ✓ {len(by_video)} unique videos, {total_segs:,} segments after filters")
    print(f"   ✗ {skipped_text:,} segments removed by text/duration filter")

    # ── Step 2: Convert MP3 → full WAV (once per video) ──────────────────────
    print("\n[2/5] Converting MP3s → full 16kHz mono WAVs …")
    bad_videos   = set()
    converted    = 0

    for video_id in tqdm(by_video, desc="Converting MP3→WAV"):
        mp3_path     = DATA_DIR / f"{video_id}.mp3"
        full_wav     = TEMP_WAV_DIR / f"{video_id}.wav"

        # Skip if already converted in a previous run
        if full_wav.exists():
            converted += 1
            continue

        if not mp3_path.exists():
            print(f"\n[WARN] MP3 missing: {mp3_path.name} — trying yt-dlp …")
            if not redownload_mp3(video_id, mp3_path):
                bad_videos.add(video_id)
                continue

        ok = convert_mp3_to_wav(mp3_path, full_wav)
        if not ok:
            # MP3 is corrupt — try redownload then retry once
            print(f"\n[WARN] Conversion failed: {mp3_path.name} — trying yt-dlp …")
            if redownload_mp3(video_id, mp3_path):
                ok = convert_mp3_to_wav(mp3_path, full_wav)

            if not ok:
                bad_videos.add(video_id)
                continue

        converted += 1

    print(f"   ✓ {converted} WAVs ready")
    print(f"   ✗ {len(bad_videos)} videos failed conversion and were skipped")

    # Remove bad videos from candidates
    for vid in bad_videos:
        del by_video[vid]

    # ── Step 3: Slice chunks from WAV + RMS gate + dedup ─────────────────────
    print("\n[3/5] Slicing WAV chunks …")
    seen_texts  = set()
    valid_rows  = []
    skipped_dup = skipped_rms = skipped_err = 0

    all_segs = [seg for segs in by_video.values() for seg in segs]

    for row in tqdm(all_segs, desc="Slicing"):
        full_wav = TEMP_WAV_DIR / f"{row['video_id']}.wav"
        wav_out  = wavs_dir / row["fname"]

        # Dedup by exact transcript
        if row["text"] in seen_texts:
            skipped_dup += 1
            continue

        # Slice (idempotent)
        if not wav_out.exists():
            ok = slice_wav(full_wav, row["start"], row["end"], wav_out)
            if not ok:
                skipped_err += 1
                continue

        # RMS silence gate
        if rms_of_wav(wav_out) < RMS_GATE:
            wav_out.unlink(missing_ok=True)
            skipped_rms += 1
            continue

        seen_texts.add(row["text"])
        valid_rows.append(row)

    print(f"   ✓ {len(valid_rows):,} chunks kept")
    print(f"   ✗ {skipped_dup:,} duplicate transcripts removed")
    print(f"   ✗ {skipped_rms:,} near-silent chunks removed")
    print(f"   ✗ {skipped_err:,} slice errors")

    if not valid_rows:
        raise RuntimeError("No valid rows remain. Check paths and metadata.csv.")

    # ── Step 4: Write metadata.csv ────────────────────────────────────────────
    print("\n[4/5] Writing output metadata.csv …")
    meta_path = meta_dir / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "duration"])
        writer.writeheader()
        for row in valid_rows:
            writer.writerow({
                "file_name": f"wavs/{row['fname']}",
                "text":      row["text"],
                "duration":  f"{row['duration']:.3f}",
            })
    print(f"   ✓ {meta_path}")

    # ── Step 5: Split → TSVs + audio symlinks ────────────────────────────────
    print("\n[5/5] Splitting and writing TSVs …")
    random.shuffle(valid_rows)
    n       = len(valid_rows)
    n_train = int(n * SPLIT["train"])
    n_dev   = int(n * SPLIT["dev"])
    splits  = {
        "train": valid_rows[:n_train],
        "dev":   valid_rows[n_train: n_train + n_dev],
        "test":  valid_rows[n_train + n_dev:],
    }

    total_hrs = 0.0
    for name, rows in splits.items():
        tsv = lipi_dir / f"{name}.tsv"
        with open(tsv, "w", encoding="utf-8") as f:
            for row in rows:
                sym    = audio_dir / row["fname"]
                target = wavs_dir  / row["fname"]
                if not sym.exists():
                    try:
                        sym.symlink_to(target)
                    except (OSError, NotImplementedError):
                        shutil.copy2(str(target), str(sym))
                f.write(f"audio/{row['fname']}\t{row['text']}\n")

        hrs = sum(r["duration"] for r in rows) / 3600
        total_hrs += hrs
        print(f"   {name:<8} {len(rows):>6,} utterances   {hrs:.2f}h")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║               Dataset Preparation Complete ✓                 ║
╠══════════════════════════════════════════════════════════════╣
║  Total chunks  : {len(valid_rows):>7,}                               ║
║  Total hours   : {total_hrs:>7.2f}h                              ║
║  Sample rate   : 16 000 Hz mono WAV                          ║
╠══════════════════════════════════════════════════════════════╣
║  Temp WAVs in  : {str(TEMP_WAV_DIR):<45} ║
║  (safe to delete after confirming output is good)            ║
╚══════════════════════════════════════════════════════════════╝

Paste into your notebook:
  INPUT_ROOT = Path(r"{OUTPUT_ROOT}")
""")


if __name__ == "__main__":
    main()
```
Delete unnecessary files..temp_wavs is keep worthy...it'll be used again to chunk
then follow this
```python
import csv
import re
import random
import shutil
import unicodedata
from pathlib import Path
from tqdm import tqdm
from pydub import AudioSegment

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
# 1. Your Master CSV
INPUT_CSV = Path(r"F:\Dataset\Lipi-Ghor-bn-882-SSTT\metadata.csv")

# 2. Where the FULL WAVs from your screenshot are
WAV_SOURCE_DIR = Path(r"F:\Dataset\lipi-ghor-training-ready\temp_full_wavs")

# 3. Where you want the final chunks and training files to go
OUTPUT_ROOT = Path(r"D:\Dataset\Lipighor_wavs")

SPLIT         = {"train": 0.90, "dev": 0.05, "test": 0.05}
RANDOM_SEED   = 42

# Thresholds
MIN_DUR       = 0.0
MAX_DUR       = 30.0
MIN_CHARS     = 5
BN_RATIO_MIN  = 0.50
ALLOWED_PUNCT = "।॥?!,;:—-\u200c\u200d"
# ═══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text)).strip()
    text = re.sub(r"[^\u0980-\u09FF\u200c\u200da-zA-Z0-9\s" + re.escape(ALLOWED_PUNCT) + r"]", "", text)
    return re.sub(r"\s+", " ", text).strip()

def is_quality_text(text: str) -> bool:
    if len(text) < MIN_CHARS: return False
    alpha = [c for c in text if c.isalpha()]
    if not alpha: return False
    bn = sum(1 for c in alpha if 0x0980 <= ord(c) <= 0x09FF)
    return (bn / len(alpha)) >= BN_RATIO_MIN

def main():
    random.seed(RANDOM_SEED)
    
    # Paths for sliced output
    wavs_out_dir = OUTPUT_ROOT / "wavs_asr_chunks" / "wavs"
    meta_dir     = OUTPUT_ROOT / "wavs_asr_chunks"
    lipi_dir     = OUTPUT_ROOT / "lipi-ghor"
    audio_dir    = lipi_dir / "audio"
    
    for d in [wavs_out_dir, meta_dir, lipi_dir, audio_dir]: 
        d.mkdir(parents=True, exist_ok=True)

    valid_rows = []
    seen_texts = set()
    current_audio = None
    current_video_id = None

    print(f"\n[1/2] Slicing WAVs based on {INPUT_CSV.name}...")
    
    # Sort CSV by video_id to avoid reloading the same large WAV file multiple times
    with open(INPUT_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: x['video_id'])

    for row in tqdm(rows, desc="Processing Segments"):
        video_id = row["video_id"]
        source_wav = WAV_SOURCE_DIR / f"{video_id}.wav"
        
        if not source_wav.exists():
            continue

        # Load audio only when the video_id changes (Efficient)
        if video_id != current_video_id:
            try:
                current_audio = AudioSegment.from_wav(str(source_wav))
                current_video_id = video_id
            except Exception:
                continue

        # Duration Filter
        try:
            start_ms = float(row["start"]) * 1000
            end_ms   = float(row["end"]) * 1000
            duration = (end_ms - start_ms) / 1000
        except ValueError: continue
        
        if not (MIN_DUR <= duration <= MAX_DUR):
            continue

        # Text Quality & Dedup
        text = clean_text(row.get("text", ""))
        if text in seen_texts or not is_quality_text(text):
            continue

        # Slice and Export Chunk
        chunk_fname = f"{video_id}_chunk_{int(row['chunk_id']):04d}.wav"
        chunk_path = wavs_out_dir / chunk_fname
        
        if not chunk_path.exists():
            chunk = current_audio[start_ms:end_ms]
            # Standardize: 16kHz Mono
            chunk = chunk.set_frame_rate(16000).set_channels(1)
            chunk.export(str(chunk_path), format="wav")

        seen_texts.add(text)
        valid_rows.append({"fname": chunk_fname, "wav_path": chunk_path, "text": text, "duration": duration})

    print(f"\n[2/2] Creating Training Splits ({len(valid_rows)} chunks)...")

    # Write Master Metadata
    with open(meta_dir / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "duration"])
        writer.writeheader()
        for r in valid_rows:
            writer.writerow({"file_name": f"wavs/{r['fname']}", "text": r['text'], "duration": f"{r['duration']:.2f}"})

    random.shuffle(valid_rows)
    n = len(valid_rows)
    tr, dv = int(n * SPLIT["train"]), int(n * SPLIT["dev"])
    splits = {"train": valid_rows[:tr], "dev": valid_rows[tr:tr+dv], "test": valid_rows[tr+dv:]}

    for name, items in splits.items():
        with open(lipi_dir / f"{name}.tsv", "w", encoding="utf-8") as f:
            for r in items:
                dest = audio_dir / r["fname"]
                if not dest.exists():
                    try: dest.symlink_to(r["wav_path"])
                    except: shutil.copy2(str(r["wav_path"]), str(dest))
                f.write(f"audio/{r['fname']}\t{r['text']}\n")
        print(f"   {name:<8}: {len(items):>6,} utts")

if __name__ == "__main__":
    main()
```
