#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-16}"
DET_NUM_WORKERS="${DET_NUM_WORKERS:-8}"
DET_EVAL_BATCH_SIZE="${DET_EVAL_BATCH_SIZE:-8}"
DET_EVAL_NUM_WORKERS="${DET_EVAL_NUM_WORKERS:-8}"
# A controlled counterpart run starts all six variants from the same training
# budget by default. Set AUTO_RESUME=1 only to continue verified checkpoints
# produced by this same experiment protocol.
AUTO_RESUME="${AUTO_RESUME:-0}"
OUTPUT_SUFFIX="${VSAA_COUNTERPART_SUFFIX:-_vsaa_counterpart}"

CONFIGS=(
  configs/detection/dbnetpp_official_baseline.yaml
  configs/detection/dbnetpp_strip_pooling.yaml
  configs/detection/dbnetpp_coordinate_attention.yaml
  configs/detection/dbnetpp_horizontal_strip.yaml
  configs/detection/dbnetpp_vertical_strip.yaml
  configs/detection/dbnetpp_vsaa.yaml
)

NAMES=(
  dbnetpp_official_baseline
  dbnetpp_strip_pooling
  dbnetpp_coordinate_attention
  dbnetpp_horizontal_strip
  dbnetpp_vertical_strip
  dbnetpp_vsaa
)

for cfg in "${CONFIGS[@]}"; do
  echo "[TRAIN] ${cfg}"
  train_args=(
    --config "${cfg}"
    --batch-size "${DET_BATCH_SIZE}"
    --num-workers "${DET_NUM_WORKERS}"
  )
  if [[ "${AUTO_RESUME}" == "1" ]]; then
    train_args+=(--auto-resume)
  fi
  "${PYTHON_BIN}" scripts/train_detection.py "${train_args[@]}"
done

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  name="${NAMES[$idx]}"
  checkpoint="${OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[ERROR] Missing required best checkpoint: ${checkpoint}" >&2
    exit 2
  fi

  for split in val test; do
    echo "[EVAL] model=${name} split=${split} config=${cfg}"
    "${PYTHON_BIN}" scripts/eval_detection.py \
      --config "${cfg}" \
      --checkpoint "${checkpoint}" \
      --split "${split}" \
      --batch-size "${DET_EVAL_BATCH_SIZE}" \
      --num-workers "${DET_EVAL_NUM_WORKERS}" \
      --output-suffix "${OUTPUT_SUFFIX}" \
      --save-predictions
  done
done

"${PYTHON_BIN}" tools/summarize_vsaa_counterparts.py \
  --root "${OUTPUT_ROOT}/metrics/detection" \
  --suffix "${OUTPUT_SUFFIX}" \
  --output-prefix detection_vsaa_counterparts

echo "[OK] VSAA counterpart ablation finished."
