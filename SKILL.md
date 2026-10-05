---
name: audio-meeting-batch
description: "Batch meeting audio: diarize, transcribe, summarize."
version: 1.0.0
author: jgbarahbot (Jesús González-Barahona)
license: MIT
metadata:
  hermes:
    tags: [audio, diarization, transcription, meeting, summary, batch, pyannote, whisper]
---

# Audio Meeting Batch (diarize + transcribe + summarize)

Full workflow for processing **one or more audio files** into diarized
transcripts AND meeting summaries. Self-contained: the pipeline scripts
(`scripts/`) are env-var driven, and the one-time setup (venv, gated HF
models, optional GPU whisper) is inlined below, so this skill works on a
fresh machine with no other skills.

Pipeline: `ffmpeg → 16 kHz mono wav → pyannote/speaker-diarization-3.1
(CPU) → whisper.cpp (medium, GPU via Vulkan or CPU) → markdown with
[HH:MM:SS] Speaker N` → **proper-noun correction pass** (always) →
**summary** (written by the agent, per the format below).

## One-time setup

Dedicated venv with **CPU** torch (the GPU is usually busy with other work):

```
python3 -m venv ~/tools/diarize-venv
~/tools/diarize-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/tools/diarize-venv/bin/pip install numpy soundfile pyannote.audio
```

- If `OSError: Could not load this library: .../torchaudio/lib/_torchaudio.abi3.so`
  appears, the pip-default torchaudio doesn't match CPU torch:
  `pip install --force-reinstall --no-deps torchaudio --index-url https://download.pytorch.org/whl/cpu`
- Verify: `~/tools/diarize-venv/bin/python -c "import pyannote.audio, soundfile, torch, torchaudio"`.

**Gated models — TWO separate Hugging Face licenses:** both
`pyannote/speaker-diarization-3.1` AND `pyannote/segmentation-3.0` must be
approved (two different pages; approving one still 403s on the other).
Probe BEFORE a long run with a 30 s slice (`ffmpeg -ss 100 -t 30 ...`) or an
`hf_hub_download` of `pyannote/segmentation-3.0`'s `pytorch_model.bin` —
cheap, and it tells you exactly which repo the user still has to approve.
On 403 `GatedRepoError`, tell the user the EXACT repo URL that still needs
approval, then re-probe after they confirm.

**HF token:** put an uncommented `HF_TOKEN=hf_...` line in `~/.hermes/.env`
(the pipeline reads it with an anchored regex and exports it as
`HF_HUB_TOKEN`). A commented `# HF_TOKEN=...` line is a stale decoy and must
not match. Do NOT pass `use_auth_token=` to `Pipeline.from_pretrained` —
removed in pyannote.audio 3.3 (`TypeError`); the token comes from the env var.

**Transcription backend (whisper.cpp).** The pipeline calls a `whisper-cli`
binary. Default paths (override via env, see below):
`WHISPER_CLI=~/tools/whisper.cpp/build/bin/whisper-cli`,
`WHISPER_MODEL=~/tools/whisper.cpp/models/ggml-medium.bin`,
`WHISPER_DEV=0` (Intel iGPU via Vulkan; `WHISPER_GPU=0` forces CPU).
Build whisper.cpp once:

```
git clone https://github.com/ggml-org/whisper.cpp ~/tools/whisper.cpp
cd ~/tools/whisper.cpp && cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
./models/download-model.sh medium
```

If the iGPU is bound to the `i915` driver it HANGS on compute; on Intel
`xe`-platforms the `xe` driver is required. `lspci -k | grep -A3 vga` should
show `xe`, not `i915`. On a `vk::DeviceLostError` after a kernel/GRUB change
or reboot, check the driver binding. If you have no usable iGPU, just run
with `WHISPER_GPU=0` (CPU) — slower but works.

## Workflow

1. **Ask for the files** if not uploaded yet. If the user specifies a summary
   language, note it; otherwise the summary **defaults to the language of the
   audio** (whisper detects it; it's shown in each transcription header).
   Transcriptions are ALWAYS in the audio's own language (never forced).
2. **On each upload**, note the path and its duration
   (`ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 <file>`).
   Warn when a file is very long (CPU diarization is ~2–3× realtime; a
   93-min file takes hours).
3. **Before launching — don't double-run.** Check for a live pipeline and an
   existing output first (`ps aux | grep -iE 'diariz|whisper-cli|pyannote' | grep -v grep`
   and `ls <expected_out>.md*`). A second launch clobbers the shared output
   and burns a full hour of CPU. If one is in flight, wait on it.
4. **Wait for all files** (or ask whether to start with what's arrived).
   Run the **batch driver** (below) as a background process with a long
   timeout. Files are processed **strictly in order, never in parallel**
   (CPU-only; parallel runs thrash the cores).
5. **Poll** the log; deliver each finished transcription as a MEDIA:
   attachment right away. If the run will take hours, set up a **cron
   watchdog** (below) to ping the room when the output lands.
6. **Proper-noun correction pass (ALWAYS, before summarizing).** Whisper
   garbles proper nouns systematically (places, orgs, people). Run
   `scripts/proper_noun_correct.py <transcript.md> [corrections.yaml]` — it
   applies a YAML list of `{pattern, replacement, why}` regex substitutions,
   rewrites the transcript in place (idempotent), and logs every applied
   change to `<transcript>_corrections.md`. The corrections file is resolved
   by walking up from the transcript dir looking for `corrections.yaml`,
   then falling back to the skill-level `corrections.yaml`.
   For a **new project, build `<project_dir>/corrections.yaml` FIRST** from
   the official source (seminar page, org site) + speaker
   self-introductions — verify names against a web search before writing
   patterns, keep them narrow (word boundaries, full phrases), and re-run
   after adding patterns (cheap, idempotent). Re-read the corrected
   transcript before writing the summary.
7. **Write a summary** per file into `<name>_summary.md` (the transcription
   path with `_summary.md` appended instead of `.md`).
8. **Deliver** the summary files too (MEDIA:).

## Single-file pipeline (`scripts/diarize_run.py`)

Env-var driven (no positional args). Reads:

| var | meaning |
|---|---|
| `DIAR_AUDIO` | source audio (any format ffmpeg reads) |
| `DIAR_WAV` | intermediate 16 kHz mono wav |
| `DIAR_CACHE` | diarization JSON — **a FILE path** (e.g. `/tmp/diar_cache_<name>.json`), NOT a directory |
| `DIAR_OUT` | final transcript `.md` |
| `DIAR_TITLE` | H1 title |
| `DIAR_SKIP_FFMPEG=1` | reuse an existing `DIAR_WAV` |
| `WHISPER_CLI` / `WHISPER_MODEL` / `WHISPER_DEV` / `WHISPER_GPU` | transcription backend (see setup) |

```
DIAR_AUDIO=recording.ogg DIAR_WAV=/tmp/r.wav \
DIAR_CACHE=/tmp/diar_cache_r.json DIAR_OUT=out/recording.md \
DIAR_TITLE="Meeting: Topic" \
<venv-python> scripts/diarize_run.py
```

**`DIAR_CACHE` is a FILE path, a checkpoint.** If the JSON already exists it
is reloaded and only transcription re-runs — a failed/slow re-run does NOT
repeat the long pyannote phase. Passing a directory path crashes with
`FileNotFoundError` when the script tries to write the JSON. Use `/tmp`
paths, not project subdirs.

## Batch driver (`scripts/diarize_batch.py`)

```
<venv-python> scripts/diarize_batch.py OUTDIR BASE DATE [SUMMARY_LANG] FILE...
```

- `OUTDIR`: where results go.
- `BASE`: name stem, e.g. `team-sync` → outputs
  `<BASE>_<DATE>.md` (1 file) or `<BASE>_<DATE>-a.md`, `-b`, ... (N files),
  in argument order.
- `DATE`: e.g. `2026-09-08`.
- `SUMMARY_LANG`: metadata only (the agent uses it when writing summaries);
  the driver does not write summaries.
- Per-file temp paths are hash-based (`/tmp/batch_<sha1-10>_<i>.wav/.json`)
  so reruns are safe: a failed run re-invoked with the same args reuses each
  finished file's diarization cache (pyannote step skipped).
- It calls `scripts/diarize_run.py` per file (env: `DIAR_AUDIO`, `DIAR_WAV`,
  `DIAR_CACHE`, `DIAR_OUT`, `DIAR_TITLE`) and **stops on the first failure**
  with a non-zero exit.

## Launching (Hermes terminal tool quirks)

- `terminal(background=true, notify=true)` is flaky in some sessions — the
  call comes back foreground or rejects `notify`. Fallback that always
  works: launch via `execute_code` with `subprocess.Popen(...,
  start_new_session=True, env=env)`, where `env` is `os.environ` **minus**
  `PYTHONPATH`, `PYTHONHOME`, `VIRTUAL_ENV` (the venv is a different Python;
  inheriting Hermes's `PYTHONPATH` makes the child import the wrong numpy).
  Then poll with `ps -p <pid>` + `tail` of the log. Do NOT use
  `setsid`/`nohup` inside a `terminal` command — the tool rejects shell-level
  background wrappers.
- `notify` is only accepted with `background=true`; a foreground command
  with `notify` is rejected.
- Don't run full-length audio foreground: the terminal cap is 600 s and CPU
  diarization exceeds it by far. For long recordings, use
  `terminal(background=true, notify=true)` so the pipeline can run to
  completion without a foreground timeout killing it.

### Long runs (hours): cron watchdog pattern

When a run will take hours the chat session can't stay open and
notify-on-complete is unreliable. Use a durable **no_agent cron watchdog**:

1. Launch the job detached (Popen fallback); note the PID.
2. Write a watchdog script that prints **nothing** while the job runs and no
   output file exists (no_agent + empty stdout = no delivery). When the
   output file appears (>1 KB) or the PID died with no output, print ONE line
   (READY/FAILED) and write a sibling `.state` file so later ticks stay
   silent (one-shot guard).
3. Create the job: `cronjob_manage(action="create", name="<name>-watch",
   script="<name>_watch.py", no_agent=True, schedule="every 5m", repeat=120,
   deliver="origin", failure_deliver="local", enabled_toolsets=[])` —
   `repeat` bounds it (120×5m ≈ 10 h); empty `failure_deliver` keeps
   scheduler hiccups out of the user's chat.
4. Test the script once by hand (must exit 0 with empty stdout while the job
   runs), and delete any `.state` file after the manual test.
5. On the READY ping, write the summaries, then remove the job if still alive.

## Summary format (attribution rules)

**Topic-based. Default is anonymous**: no "Speaker N", no "X dijo", no person
attribution. Use passive/neutral phrasing: "se dijo…", "se señaló…", "uno de
los participantes…". (User preference, reaffirmed 2026-10-05: "only the
topics and what was said, but not who said it.")

**EXCEPTION — an introductions section** (the opening round where each
attendee self-introduces) is summarized **NAMED**: list each participant
with name + affiliation, verified against an authoritative roster file
(e.g. `introductions.txt`) if the user provides one, else against speaker
self-introductions. The roster is ground truth for BOTH the introductions
summary and the Speaker-N→name diarization map (see below).

So: introductions = named (roster-verified); everything else (debates,
decisions, calendar, org) = anonymous. State this split explicitly at the top
of the summary.

```
# <Meeting name> — summary (or "Resumen — <name>")

- Fuente: `<transcription file>` (~N min, X voces, <lang>)
- Generado con: pyannote 3.1 (diarización) + <backend> (transcripción)

## Temas y decisiones

**1. <Topic>.** Prose paragraph(s) per topic, in meeting order — decisions,
agreements, and pending actions live inside the topic paragraph, in neutral
phrasing ("uno de los participantes", "se acordó que..."; names only if
naturally present in the content).
```

Rules:
- Base the summary ONLY on the transcription (read the full `.md` file).
- The transcription is verbatim — the summary is the place to paraphrase,
  correct mishearings, and structure.
- When a proper noun is garbled by ASR, keep it quoted or flag it; do not
  silently invent the real name unless obvious from context.
- Save to `<transcription without .md>_summary.md`.
- **SINGLE** summary file when one audio session spans multiple files
  (e.g. an a+b pair) — not one per file.
- **Diarization map (optional companion file):** when the user wants the
  intro diarized, resolve each `Speaker N` label to a person against the
  roster + self-introductions and write `<name>_speakers.md` (N → person,
  plus non-human clusters like ambient/video-narration). Don't re-run
  pyannote — the labels are already in the transcript.
- Delivery: post the `.md` files as MEDIA: attachments AND paste the content
  inline in the message.

## CPU tuning & timing

Thread tuning (benchmark, 1-min file, i7-12700H 6P+8E): **6 threads ≈ 36
s/min** (stable) vs ~40 s/min at the default (all cores). `diarize_run.py`
defaults `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=6` (override
via env). P-core pinning gave no extra benefit.

Timing expectations (iGPU transcription, medium model): diarization
dominates. 25-min file ≈ 25–40 min total (pyannote ~36 s/min tuned) +
~13 min whisper on iGPU. 17-min ≈ 20–30 min. 93-min ≈ several hours
(diarization-bound). Tell the user the estimate before starting long ones;
if they object, the lever is a smaller whisper model (ask first) —
diarization speed is fixed (a GPU port of pyannote is only ~1.5–2×, not
worth it). Diarization results are cached per file in `DIAR_CACHE` — after a
crash, rerunning skips the pyannote stage.

## Pitfalls

- Never run files in parallel (user explicitly asked for sequential).
- **Correction patterns corrupt verbatim records:** keep them narrow and
  verified against an official source. After each pass, grep the transcript
  for the KNOWN-GARBLED stems (case-insensitive) and add missing spelling
  variants before re-running (a pattern covering only one spelling leaves the
  rest uncorrected).
- **Recording-specific fixes are NOT systematic patterns.** A correction
  that applies to one recording only (a name that appears once) belongs in
  that recording's transcript/summary, not in the shared `corrections.yaml`.
- `/tmp` is ephemeral: caches die on reboot; the outputs survive.
- The driver hard-stops on first failure: fix, then rerun the SAME command —
  completed files are cheap (cache hit).
- Summary language: default = audio language (read the "Idioma detectado"
  line in each transcription header); override only if the user specified one.
