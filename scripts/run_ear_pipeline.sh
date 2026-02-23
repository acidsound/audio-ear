#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <reference.wav> <test.wav> [out_dir] [--preset-a A.json --preset-b B.json]"
  exit 1
fi

REF="$1"
TEST="$2"
OUT_DIR="/tmp/ear-pipeline"
shift 2
if [[ $# -gt 0 && "$1" != --* ]]; then
  OUT_DIR="$1"
  shift
fi

PRESET_A=""
PRESET_B=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preset-a)
      PRESET_A="${2:-}"
      shift 2
      ;;
    --preset-b)
      PRESET_B="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 2
      ;;
  esac
done

mkdir -p "$OUT_DIR"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
QUICK="$ROOT_DIR/scripts/ffmpeg_sox_quick_compare.py"
DETAIL="$ROOT_DIR/scripts/wav_ear_compare.py"
PRESET_COMPARE="$ROOT_DIR/scripts/preset_effective_compare.py"

python3 "$QUICK" \
  --ref "$REF" \
  --test "$TEST" \
  --json-out "$OUT_DIR/quick.json" \
  --md-out "$OUT_DIR/quick.md"

python3 "$DETAIL" \
  --ref "$REF" \
  --test "$TEST" \
  --json-out "$OUT_DIR/detail.json" \
  --md-out "$OUT_DIR/detail.md"

if [[ -n "$PRESET_A" && -n "$PRESET_B" ]]; then
  python3 "$PRESET_COMPARE" \
    --a "$PRESET_A" \
    --b "$PRESET_B" \
    --json-out "$OUT_DIR/preset_effective.json" \
    --md-out "$OUT_DIR/preset_effective.md"
fi

echo "Saved reports in: $OUT_DIR"
