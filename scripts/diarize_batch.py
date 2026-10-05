#!/usr/bin/env python3
"""Batch driver: diarize + transcribe a group of audio files, sequentially.

Runs the sibling diarize_run.py once per file with per-file env vars
(unique WAV / cache / out per file). Files are processed strictly in
argument order — never in parallel (CPU-only: parallel runs thrash).

Usage:
  diarize_batch.py OUTDIR BASE DATE [SUMMARY_LANG] FILE...

Example:
  ~/tools/diarize-venv/bin/python scripts/diarize_batch.py \\
      out/ team-sync 2026-09-08 Spanish \\
      /path/a.mp3 /path/b.mp3

Naming:
  - 1 file  -> <BASE>_<DATE>.md
  - N files -> <BASE>_<DATE>-a.md, -b, -c, ... (lowercase letters, in order)
  Summaries (written by the agent, not this script) get the same name with
  _summary.md appended: <BASE>_<DATE>-a_summary.md

SUMMARY_LANG is metadata for the agent (default = language of the audio,
detected per file and shown in each transcription header). This script does
not write summaries.
"""
import os
import re
import subprocess
import sys
import hashlib

# Resolve diarize_run.py next to THIS script (portable), falling back to the
# conventional ~/tools location. Runs under whatever python invoked us
# (the venv), so no hardcoded interpreter path.
_HERE = os.path.dirname(os.path.abspath(__file__))
DIARIZE = os.environ.get("DIARIZE_RUN") or next(
    (p for p in (os.path.join(_HERE, "diarize_run.py"),
                 os.path.expanduser("~/tools/diarize_run.py"))
     if os.path.exists(p)), None)
VENV_PY = sys.executable


def suffix(n_files, idx):
    if n_files == 1:
        return ""
    return "-" + chr(ord("a") + idx)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    outdir, base, date, summary_lang = sys.argv[1:5]
    files = sys.argv[5:]
    if not DIARIZE or not os.path.exists(DIARIZE):
        print("ERROR: diarize_run.py not found (set DIARIZE_RUN or place it "
              f"next to this script)", file=sys.stderr)
        sys.exit(1)
    os.makedirs(outdir, exist_ok=True)

    n = len(files)
    for i, audio in enumerate(files):
        if not os.path.exists(audio):
            print(f"ERROR: missing input {audio}", file=sys.stderr)
            sys.exit(1)
    h = hashlib.sha1(" ".join(files).encode()).hexdigest()[:10]

    for i, audio in enumerate(files):
        out = os.path.join(outdir, f"{base}_{date}{suffix(n, i)}.md")
        wav = f"/tmp/batch_{h}_{i}.wav"
        cache = f"/tmp/batch_{h}_{i}.json"
        env = dict(os.environ)
        env.update({
            "DIAR_AUDIO": audio,
            "DIAR_WAV": wav,
            "DIAR_CACHE": cache,
            "DIAR_OUT": out,
            "DIAR_TITLE": "Transcripción con diarización",
        })
        print(f"=== [{i+1}/{n}] {os.path.basename(audio)} -> {out} "
              f"(summary lang: {summary_lang or 'auto'}) ===", flush=True)
        rc = subprocess.call([VENV_PY, "-u", DIARIZE], env=env)
        if rc != 0:
            print(f"ERROR: pipeline failed (rc={rc}) for {audio}", file=sys.stderr)
            print("STOPPED: fix the issue and rerun (finished files keep their cache).",
                  file=sys.stderr)
            sys.exit(rc)
    print(f"BATCH DONE: {n} file(s) in {outdir}", flush=True)


if __name__ == "__main__":
    main()
