#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON:-python}"
REC_REVAL_SPLIT="${REC_REVAL_SPLIT:-val}"
REC_REVAL_CHECKPOINT_MODE="${REC_REVAL_CHECKPOINT_MODE:-best-last}"
REC_REVAL_BATCH_SIZE="${REC_REVAL_BATCH_SIZE:-256}"
REC_REVAL_NUM_WORKERS="${REC_REVAL_NUM_WORKERS:-8}"
REC_REVAL_DECODE_MODE="${REC_REVAL_DECODE_MODE:-greedy}"
REC_REVAL_WORD_EDIT_DISTANCE="${REC_REVAL_WORD_EDIT_DISTANCE:-0}"
REC_REVAL_CHAR_EDIT_DISTANCE="${REC_REVAL_CHAR_EDIT_DISTANCE:-0}"
REC_REVAL_MAX_BATCHES="${REC_REVAL_MAX_BATCHES:-}"

args=(
  --split "${REC_REVAL_SPLIT}"
  --checkpoint-mode "${REC_REVAL_CHECKPOINT_MODE}"
  --batch-size "${REC_REVAL_BATCH_SIZE}"
  --num-workers "${REC_REVAL_NUM_WORKERS}"
  --decode-mode "${REC_REVAL_DECODE_MODE}"
  --word-accuracy-edit-distance "${REC_REVAL_WORD_EDIT_DISTANCE}"
  --character-accuracy-edit-distance "${REC_REVAL_CHAR_EDIT_DISTANCE}"
  --python "${PYTHON_BIN}"
)

if [[ -n "${REC_REVAL_MAX_BATCHES}" ]]; then
  args+=(--max-batches "${REC_REVAL_MAX_BATCHES}")
fi

"${PYTHON_BIN}" tools/revalidate_recognition_checkpoints.py "${args[@]}"

echo "[OK] Recognition ablation revalidation finished."
