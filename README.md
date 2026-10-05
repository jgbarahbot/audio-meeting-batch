# audio-meeting-batch

Turn **meeting recordings** into **diarized transcripts + meeting summaries**.

A self-contained Hermes Agent skill. It chains `ffmpeg → pyannote/speaker-
diarization-3.1 (who spoke when) → whisper.cpp (what was said) → markdown`
with `[HH:MM:SS] Speaker N` labels, then runs a **proper-noun correction
pass** and writes a **topic-based summary**.

- **Diarization:** `pyannote/speaker-diarization-3.1` on CPU (cached per file).
- **Transcription:** `whisper.cpp` `medium`, on an Intel iGPU (Vulkan) or CPU.
- **Corrections:** a per-project `corrections.yaml` of narrow regex fixes for
  the proper nouns whisper garbles.
- **Summary:** written by the agent — anonymous by default, with an optional
  *named* introductions section when a roster is provided.

## What's in the repo

| path | what |
|---|---|
| `SKILL.md` | the skill itself — full workflow, setup, pitfalls (read this first) |
| `scripts/diarize_run.py` | single-file pipeline (env-var driven) |
| `scripts/diarize_batch.py` | sequential batch driver (resumable) |
| `scripts/proper_noun_correct.py` | proper-noun correction pass (no PyYAML dep) |
| `corrections.example.yaml` | template for your project's corrections |
| `requirements.txt` | venv deps |
| `setup.sh` | one-shot install (venv + optional whisper.cpp) |
| `watchdog_example.py` | example cron watchdog for multi-hour runs |

## Install

```bash
git clone <this-repo> ~/tools/audio-meeting-batch
cd ~/tools/audio-meeting-batch
./setup.sh            # builds the venv; add --whisper to build whisper.cpp
```

Or do it by hand (see `SKILL.md` → "One-time setup" for the full list of
gotchas — two **separate** Hugging Face gated-model licenses are required):

```bash
python3 -m venv ~/tools/diarize-venv
~/tools/diarize-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/tools/diarize-venv/bin/pip install -r requirements.txt
```

You also need: an `HF_TOKEN` in `~/.hermes/.env` (or `HF_HUB_TOKEN` in the
env), both `pyannote` gated models approved, and a `whisper-cli` binary +
model (defaults in `setup.sh`; override with `WHISPER_CLI` / `WHISPER_MODEL`).

## Use

For one file:

```bash
DIAR_AUDIO=meeting.ogg DIAR_WAV=/tmp/m.wav \
DIAR_CACHE=/tmp/diar_m.json DIAR_OUT=out/meeting.md \
DIAR_TITLE="Team Sync — Q3" \
~/tools/diarize-venv/bin/python scripts/diarize_run.py

python3 scripts/proper_noun_correct.py out/meeting.md corrections.yaml
```

For many files (sequential, resumable):

```bash
~/tools/diarize-venv/bin/python scripts/diarize_batch.py \
  out/ "team-sync" 2026-09-08 Spanish a.ogg b.ogg c.ogg
```

`DIAR_CACHE` must be a **file** path (a checkpoint), not a directory.

## Reuse

The pipeline is env-var driven and path-independent, so it runs on any
machine once the venv + whisper.cpp are present. Per-project knowledge lives
in a `corrections.yaml` you keep next to (or in the tree above) your
transcripts — start from `corrections.example.yaml`. Nothing in the repo is
tied to a specific meeting.

## License

MIT — see `LICENSE`.
