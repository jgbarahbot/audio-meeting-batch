#!/usr/bin/env bash
# One-shot setup for audio-meeting-batch.
#
#   ./setup.sh             # venv only (pyannote + CPU torch)
#   ./setup.sh --whisper   # also build whisper.cpp (Vulkan iGPU) + medium model
#   WHISPER_CLI=/path ./setup.sh   # override defaults
#
# Requires: python3 (venv), git, ffmpeg, cmake + a C compiler (for --whisper).
# The Intel iGPU Vulkan path needs the 'xe' driver bound to the GPU (see SKILL.md).
set -euo pipefail

WHISPER_DIR="${WHISPER_DIR:-$HOME/tools/whisper.cpp}"
WHISPER_MODEL_NAME="${WHISPER_MODEL_NAME:-medium}"
WITH_WHISPER=0
for a in "$@"; do
  case "$a" in
    --whisper) WITH_WHISPER=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# 1) venv with CPU torch + pyannote
python3 -m venv ~/tools/diarize-venv
~/tools/diarize-venv/bin/pip install --upgrade pip
~/tools/diarize-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/tools/diarize-venv/bin/pip install -r "$(cd "$(dirname "$0")" && pwd)/requirements.txt"
# fix torchaudio ABI mismatch if it appears (pip default vs CPU torch)
if ! ~/tools/diarize-venv/bin/python -c "import torchaudio" 2>/dev/null; then
  echo "torchaudio mismatch — reinstalling CPU build..."
  ~/tools/diarize-venv/bin/pip install --force-reinstall --no-deps torchaudio \
    --index-url https://download.pytorch.org/whl/cpu
fi
~/tools/diarize-venv/bin/python -c "import pyannote.audio, soundfile, torch, torchaudio; print('venv OK')"

# 2) optional whisper.cpp (Vulkan iGPU)
if [ "$WITH_WHISPER" = 1 ]; then
  if [ ! -d "$WHISPER_DIR" ]; then
    git clone https://github.com/ggml-org/whisper.cpp "$WHISPER_DIR"
  fi
  ( cd "$WHISPER_DIR"
    cmake -B build -DGGML_VULKAN=ON
    cmake --build build -j
    ./models/download-model.sh "$WHISPER_MODEL_NAME"
  )
  echo "whisper-cli: $WHISPER_DIR/build/bin/whisper-cli"
  echo "whisper model: $WHISPER_DIR/models/ggml-${WHISPER_MODEL_NAME}.bin"
fi

echo
echo "Setup complete."
echo "  - Approve BOTH gated HF models: pyannote/speaker-diarization-3.1 AND pyannote/segmentation-3.0"
echo "  - Put 'HF_TOKEN=hf_...' (uncommented) in ~/.hermes/.env"
echo "  - Transcribe with:"
echo "      DIAR_AUDIO=in.ogg DIAR_WAV=/tmp/in.wav DIAR_CACHE=/tmp/diar_in.json DIAR_OUT=out/in.md DIAR_TITLE='Title' \\"
echo "        ~/tools/diarize-venv/bin/python scripts/diarize_run.py"
echo "  - Without a usable iGPU, add WHISPER_GPU=0 to run transcription on CPU."
