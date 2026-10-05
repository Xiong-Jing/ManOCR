#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export OCR_MANCHU_PROJECT_ROOT="${OCR_MANCHU_PROJECT_ROOT:-${PROJECT_ROOT}}"
export OCR_MANCHU_OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"

PYTHON_BIN="${PYTHON:-python}"
HELDOUT_TASKS="${HELDOUT_TASKS:-all}"
REC_EVAL_BATCH_SIZE="${REC_EVAL_BATCH_SIZE:-256}"
DET_EVAL_BATCH_SIZE="${DET_EVAL_BATCH_SIZE:-8}"
EVAL_NUM_WORKERS="${EVAL_NUM_WORKERS:-8}"
CHECKPOINT_POLICY="${CHECKPOINT_POLICY:-best_then_last}"

args=(
  --plan configs/experiments/heldout_all_models.yaml
  --tasks "${HELDOUT_TASKS}"
  --splits test
  --python-bin "${PYTHON_BIN}"
  --checkpoint-policy "${CHECKPOINT_POLICY}"
  --recognition-batch-size "${REC_EVAL_BATCH_SIZE}"
  --detection-batch-size "${DET_EVAL_BATCH_SIZE}"
  --num-workers "${EVAL_NUM_WORKERS}"
)

if [[ "${HELDOUT_ALLOW_MISSING:-0}" == "1" ]]; then
  args+=(--allow-missing)
fi

if [[ "${HELDOUT_DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_heldout_tests.py "${args[@]}"
