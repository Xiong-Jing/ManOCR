#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
REC_LAMBDA_BATCH_SIZE="${REC_LAMBDA_BATCH_SIZE:-96}"
REC_LAMBDA_NUM_WORKERS="${REC_LAMBDA_NUM_WORKERS:-8}"
REC_LAMBDA_EVAL_BATCH_SIZE="${REC_LAMBDA_EVAL_BATCH_SIZE:-256}"
REC_LAMBDA_EVAL_NUM_WORKERS="${REC_LAMBDA_EVAL_NUM_WORKERS:-8}"
REC_LAMBDA_EPOCHS="${REC_LAMBDA_EPOCHS:-}"
REC_LAMBDA_MAX_TRAIN_BATCHES="${REC_LAMBDA_MAX_TRAIN_BATCHES:-}"
REC_LAMBDA_MAX_VAL_BATCHES="${REC_LAMBDA_MAX_VAL_BATCHES:-}"
REC_LAMBDA_EVAL_MAX_BATCHES="${REC_LAMBDA_EVAL_MAX_BATCHES:-}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
# A controlled sensitivity run starts from scratch unless explicitly resumed.
AUTO_RESUME="${AUTO_RESUME:-0}"

CONFIGS=(
  configs/recognition/svtr_dab_otp_lambda_0.yaml
  configs/recognition/svtr_dab_otp_lambda_0p025.yaml
  configs/recognition/svtr_dab_otp_lambda_0p05.yaml
  configs/recognition/svtr_dab_otp_lambda_0p1.yaml
  configs/recognition/svtr_dab_otp_lambda_0p2.yaml
  configs/recognition/svtr_dab_otp_lambda_0p5.yaml
)

NAMES=(
  svtr_dab_otp_lambda_0
  svtr_dab_otp_lambda_0p025
  svtr_dab_otp_lambda_0p05
  svtr_dab_otp_lambda_0p1
  svtr_dab_otp_lambda_0p2
  svtr_dab_otp_lambda_0p5
)

LAMBDAS=(0 0.025 0.05 0.1 0.2 0.5)
METRIC_SOURCES=(SVTR SVTR SVTR SVTR+Lortho SVTR SVTR)

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  for idx in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$idx]}"
    echo "[TRAIN] lambda=${LAMBDAS[$idx]} config=${cfg}"
    train_args=(
      --config "${cfg}"
      --batch-size "${REC_LAMBDA_BATCH_SIZE}"
      --num-workers "${REC_LAMBDA_NUM_WORKERS}"
    )
    if [[ -n "${REC_LAMBDA_EPOCHS}" ]]; then
      train_args+=(--epochs "${REC_LAMBDA_EPOCHS}")
    fi
    if [[ -n "${REC_LAMBDA_MAX_TRAIN_BATCHES}" ]]; then
      train_args+=(--max-train-batches "${REC_LAMBDA_MAX_TRAIN_BATCHES}")
    fi
    if [[ -n "${REC_LAMBDA_MAX_VAL_BATCHES}" ]]; then
      train_args+=(--max-val-batches "${REC_LAMBDA_MAX_VAL_BATCHES}")
    fi
    if [[ "${AUTO_RESUME}" == "1" ]]; then
      train_args+=(--auto-resume)
    fi
    "${PYTHON_BIN}" scripts/train_recognition.py "${train_args[@]}"
  done
fi

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  name="${NAMES[$idx]}"
  checkpoint="${OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Missing required best checkpoint: ${checkpoint}" >&2
    exit 2
  fi

  for split in val test; do
    echo "[EVAL] model=${name} lambda=${LAMBDAS[$idx]} split=${split} metric_source=${METRIC_SOURCES[$idx]} checkpoint=${checkpoint}"
    eval_args=(
      --config "${cfg}"
      --checkpoint "${checkpoint}"
      --split "${split}"
      --batch-size "${REC_LAMBDA_EVAL_BATCH_SIZE}"
      --num-workers "${REC_LAMBDA_EVAL_NUM_WORKERS}"
      --save-predictions
    )
    if [[ -n "${REC_LAMBDA_EVAL_MAX_BATCHES}" ]]; then
      eval_args+=(--max-batches "${REC_LAMBDA_EVAL_MAX_BATCHES}")
    fi
    "${PYTHON_BIN}" scripts/eval_recognition.py "${eval_args[@]}"
  done
done

"${PYTHON_BIN}" tools/summarize_otp_lambda_sensitivity.py \
  --root "${OUTPUT_ROOT}/metrics/recognition"

echo "[OK] OTP lambda-sensitivity ablation finished."
