#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
DET_BATCH_SIZE="${DET_BATCH_SIZE:-}"
DET_NUM_WORKERS="${DET_NUM_WORKERS:-}"
DET_EPOCHS="${DET_EPOCHS:-}"
DET_EVAL_BATCH_SIZE="${DET_EVAL_BATCH_SIZE:-8}"
DET_EVAL_NUM_WORKERS="${DET_EVAL_NUM_WORKERS:-8}"
AUTO_RESUME="${AUTO_RESUME:-1}"
RUN_OURS_TRAIN="${RUN_OURS_TRAIN:-1}"
DET_EXTERNAL_IOU_THRESH="${DET_EXTERNAL_IOU_THRESH:-0.75}"
DET_OURS_IOU_THRESH="${DET_OURS_IOU_THRESH:-0.70}"
DET_DEGRADED_SUFFIX="${DET_DEGRADED_SUFFIX:-_degraded_modelwise}"

TRAIN_CONFIGS=(
  "configs/detection/east_baseline.yaml"
  "configs/detection/craft_baseline.yaml"
  "configs/detection/dbnet_baseline.yaml"
  "configs/detection/ppocrv5_det_baseline.yaml"
  "configs/detection/hisam_baseline.yaml"
)

EVAL_CONFIGS=(
  "configs/detection/east_baseline.yaml"
  "configs/detection/craft_baseline.yaml"
  "configs/detection/dbnet_baseline.yaml"
  "configs/detection/ppocrv5_det_baseline.yaml"
  "configs/detection/hisam_baseline.yaml"
  "configs/detection/dbnetpp_vsaa_asym_shrink.yaml"
)

EVAL_NAMES=(
  "east_baseline"
  "craft_baseline"
  "dbnet_baseline"
  "ppocrv5_det_baseline"
  "hisam_baseline"
  "dbnetpp_vsaa_asym_shrink"
)

EVAL_IOU_THRESHOLDS=(
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_OURS_IOU_THRESH}"
)

OURS_CONFIG="configs/detection/dbnetpp_vsaa_asym_shrink.yaml"
OURS_NAME="dbnetpp_vsaa_asym_shrink"
OURS_CKPT="${OUTPUT_ROOT}/checkpoints/detection/${OURS_NAME}/best.pth"

if [[ "${RUN_OURS_TRAIN}" == "1" ]]; then
  TRAIN_CONFIGS+=("${OURS_CONFIG}")
elif [[ "${RUN_OURS_TRAIN}" == "auto" && ! -f "${OURS_CKPT}" ]]; then
  TRAIN_CONFIGS+=("${OURS_CONFIG}")
else
  echo "[INFO] Skip training ours. RUN_OURS_TRAIN=${RUN_OURS_TRAIN}, checkpoint=${OURS_CKPT}"
fi

train_model() {
  local config="$1"

  echo "[TRAIN] ${config}"
  local train_args=(--config "${config}")

  if [[ -n "${DET_BATCH_SIZE}" ]]; then
    train_args+=(--batch-size "${DET_BATCH_SIZE}")
  fi
  if [[ -n "${DET_NUM_WORKERS}" ]]; then
    train_args+=(--num-workers "${DET_NUM_WORKERS}")
  fi
  if [[ -n "${DET_EPOCHS}" ]]; then
    train_args+=(--epochs "${DET_EPOCHS}")
  fi
  if [[ "${AUTO_RESUME}" == "1" ]]; then
    train_args+=(--auto-resume)
  fi

  "${PYTHON_BIN}" scripts/train_detection.py "${train_args[@]}"
}

eval_model() {
  local config="$1"
  local name="$2"
  local split="$3"
  local iou_thresh="$4"
  local ckpt="${OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] Missing checkpoint, skip eval: ${ckpt}"
    return
  fi

  echo "[DEGRADED MODELWISE EVAL] ${name} ${split} IoU=${iou_thresh}"
  "${PYTHON_BIN}" scripts/eval_detection.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --split "${split}" \
    --batch-size "${DET_EVAL_BATCH_SIZE}" \
    --num-workers "${DET_EVAL_NUM_WORKERS}" \
    --degraded-eval \
    --iou-thresh "${iou_thresh}" \
    --output-suffix "${DET_DEGRADED_SUFFIX}" \
    --save-predictions
}

for idx in "${!TRAIN_CONFIGS[@]}"; do
  train_model "${TRAIN_CONFIGS[$idx]}"
done

for idx in "${!EVAL_CONFIGS[@]}"; do
  for split in val test; do
    eval_model "${EVAL_CONFIGS[$idx]}" "${EVAL_NAMES[$idx]}" "${split}" "${EVAL_IOU_THRESHOLDS[$idx]}"
  done
done

"${PYTHON_BIN}" tools/summarize_detection_comparison.py \
  --detection-root "${OUTPUT_ROOT}/metrics/detection" \
  --baseline-root "${OUTPUT_ROOT}/metrics/baselines" \
  --suffix "${DET_DEGRADED_SUFFIX}" \
  --output-prefix "detection_comparison${DET_DEGRADED_SUFFIX}"

echo "[OK] Detection comparison finished."
