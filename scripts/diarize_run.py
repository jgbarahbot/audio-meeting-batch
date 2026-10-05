#!/usr/bin/env python3
"""Speaker diarization + transcription for dependentium audio.

Pipeline: ffmpeg -> 16kHz mono wav -> pyannote speaker-diarization-3.1
-> faster-whisper (medium, local) with word timestamps -> merge words into
speaker turns -> markdown with [HH:MM:SS] Speaker N.
"""
import os, re, subprocess, sys, json

AUDIO = os.environ.get("DIAR_AUDIO") or os.path.expanduser("~/.hermes/cache/audio/audio_7fbe3f6e4044.mp3")
WAV = os.environ.get("DIAR_WAV") or "/tmp/dependentium_16k.wav"
OUT = os.environ.get("DIAR_OUT") or os.path.expanduser("~/Documents/dependentium_2026-09-08.md")
DIAR_CACHE = os.environ.get("DIAR_CACHE") or "/tmp/dependentium_diar.json"

# HF token (needed for pyannote's gated model)
token = None
env_file = os.path.expanduser("~/.hermes/.env")
if os.path.exists(env_file):
    for line in open(env_file):
        m = re.match(r'(?:export\s+)?HF_TOKEN\s*=\s*(\S+)', line)
        if m:
            token = m.group(1).strip().strip('"').strip("'")
if token:
    os.environ["HF_TOKEN"] = token
    os.environ["HF_HUB_TOKEN"] = token
    print("HF token loaded", flush=True)
else:
    print("WARNING: no HF_TOKEN found", flush=True)

# 1. audio -> 16k mono wav (skippable: DIAR_WAV may already be prepared)
if os.environ.get("DIAR_SKIP_FFMPEG") != "1":
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", AUDIO,
         "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", WAV],
        check=True)
print("wav ready", flush=True)

# 2. diarization (in-memory audio: torchcodec file decoding is broken in this venv)
# Thread tuning (benchmark 2026-09-09, 1-min file, i7-12700H 6P+8E):
# 6 threads is fastest (~36 s/min) vs ~40 s/min at the default (all cores).
# Override with OMP_NUM_THREADS=... if desired.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
import numpy as np
import soundfile as sf
import torch
from pyannote.audio import Pipeline

wav_np, sr = sf.read(WAV, dtype="float32")
dur = len(wav_np) / sr
waveform = torch.from_numpy(wav_np).unsqueeze(0)  # (1, n)
audio_dict = {"waveform": waveform, "sample_rate": sr}

pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1")
pipe.to(torch.device("cpu"))
if os.path.exists(DIAR_CACHE):
    spk_spans = [list(x) for x in json.load(open(DIAR_CACHE))]
    print(f"diarization loaded from cache: {len(spk_spans)} turns", flush=True)
else:
    print("running diarization (CPU, ~25min audio)...", flush=True)
    diar = pipe(audio_dict)
    spk_spans = [(turn.start, turn.end, spk)
                 for turn, spk, _ in diar.speaker_diarization.itertracks(yield_label=True)]
    json.dump(spk_spans, open(DIAR_CACHE, "w"))
    print(f"diarization done: {len(spk_spans)} turns (cached)", flush=True)

def speaker_at(t):
    # the speaker with max overlap with [t-0.1, t+0.1]
    t0, t1 = max(0.0, t - 0.1), t + 0.1
    best_sp, best_ov = None, 0.0
    for s, e, sp in spk_spans:
        ov = min(e, t1) - max(s, t0)
        if ov > best_ov:
            best_ov, best_sp = ov, sp
    return best_sp or "SPEAKER_00"

# 3. transcription (whisper.cpp on the Intel iGPU, Vulkan)
# Note: this build emits segment-level timestamps only (no word-level in JSON),
# so speaker turns are built per segment (max-overlap assignment + merge).
import time
WHISPER_BIN = os.environ.get("WHISPER_CLI") or os.path.expanduser(
    "~/tools/whisper.cpp/build/bin/whisper-cli")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL") or os.path.expanduser(
    "~/tools/whisper.cpp/models/ggml-medium.bin")
WHISPER_DEV = os.environ.get("WHISPER_DEV", "0")  # 0 = Intel iGPU (xe driver)
if os.environ.get("WHISPER_GPU") == "0":
    WHISPER_DEV = "-1"  # force CPU


def whisper_cpp(wav, out_json, lang):
    cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav,
           "-l", lang, "-t", "2", "-oj", "-of", out_json]
    if WHISPER_DEV != "-1":
        cmd += ["-dev", WHISPER_DEV]
    t0 = time.time()
    subprocess.run(cmd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dt = time.time() - t0
    d = json.load(open(out_json + ".json"))
    print(f"whisper.cpp ({'GPU dev ' + WHISPER_DEV if WHISPER_DEV != '-1' else 'CPU'}): "
          f"{dt:.0f}s for {len(d['transcription'])} segments", flush=True)
    return d


def whisper_segments(d):
    segs = []
    for seg in d["transcription"]:
        o = seg.get("offsets", {})
        s, e = o.get("from"), o.get("to")
        text = seg.get("text", "").strip()
        if s is None or e is None or not text:
            continue
        segs.append({"start": s / 1000.0, "end": e / 1000.0, "text": text})
    return segs


print("running whisper.cpp (Intel iGPU, Vulkan)...", flush=True)
d1 = whisper_cpp(WAV, WAV + ".d1.json", "auto")
lang = d1["result"]["language"]
print(f"language: {lang} (auto-detected by whisper.cpp)", flush=True)
info = type("L", (), {"language": lang})()
info.language_probability = None
segs = whisper_segments(d1)

turns = []  # (start, end, speaker, text)
cur = None
for sg in segs:
    sp = speaker_at((sg["start"] + sg["end"]) / 2)
    if cur and cur[2] == sp and sg["start"] - cur[1] < 1.5:
        cur = (cur[0], sg["end"], sp, cur[3] + " " + sg["text"])
    else:
        if cur:
            turns.append(cur)
        cur = (sg["start"], sg["end"], sp, sg["text"])
if cur:
    turns.append(cur)

# merge consecutive turns with big gaps into silence markers later in md
print(f"{len(turns)} speaker turns", flush=True)

def ts(t):
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# 4. write markdown
names = sorted({t[2] for t in turns})
nmap = {s: f"Speaker {i+1}" for i, s in enumerate(names)}
with open(OUT, "w", encoding="utf-8") as f:
    f.write(f"# {os.environ.get('DIAR_TITLE', 'Transcripción con diarización')}\n\n")
    f.write(f"- Fuente: `{os.path.basename(AUDIO)}` (~{dur/60:.0f} min)\n")
    prob = f" ({info.language_probability:.0%})" if info.language_probability is not None else ""
    f.write(f"- Idioma detectado: **{info.language}**{prob}\n")
    backend = "whisper.cpp medium on Intel iGPU (Vulkan, xe)" if WHISPER_DEV != "-1" else "whisper.cpp medium (CPU)"
    f.write(f"- Modelos: pyannote/speaker-diarization-3.1 (diarización) + {backend} (transcripción)\n")
    f.write(f"- Números de hablante: {len(names)} ({', '.join(nmap.values())})\n\n")
    f.write("---\n\n")
    for s, e, sp, text in turns:
        text = re.sub(r"\s+", " ", text).strip()
        f.write(f"**[{ts(s)}–{ts(e)}] {nmap[sp]}:** {text}\n\n")
print(f"WROTE {OUT}", flush=True)
