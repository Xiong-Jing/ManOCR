#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
DET_EVAL_BATCH_SIZE="${DET_EVAL_BATCH_SIZE:-4}"
DET_EVAL_NUM_WORKERS="${DET_EVAL_NUM_WORKERS:-0}"
DET_EXTERNAL_IOU_THRESH="${DET_EXTERNAL_IOU_THRESH:-0.75}"
DET_OURS_IOU_THRESH="${DET_OURS_IOU_THRESH:-0.70}"
DET_DEGRADED_SUFFIX="${DET_DEGRADED_SUFFIX:-_degraded_modelwise}"

CONFIGS=(
  "configs/detection/east_baseline.yaml"
  "configs/detection/craft_baseline.yaml"
  "configs/detection/dbnet_baseline.yaml"
  "configs/detection/ppocrv5_det_baseline.yaml"
  "configs/detection/hisam_baseline.yaml"
  "configs/detection/dbnetpp_vsaa_asym_shrink.yaml"
)

NAMES=(
  "east_baseline"
  "craft_baseline"
  "dbnet_baseline"
  "ppocrv5_det_baseline"
  "hisam_baseline"
  "dbnetpp_vsaa_asym_shrink"
)

IOU_THRESHOLDS=(
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_EXTERNAL_IOU_THRESH}"
  "${DET_OURS_IOU_THRESH}"
)

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  name="${NAMES[$idx]}"
  iou_thresh="${IOU_THRESHOLDS[$idx]}"
  ckpt="${OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] Missing checkpoint, skip degraded eval: ${ckpt}"
    continue
  fi

  for split in val test; do
    echo "[DEGRADED MODELWISE EVAL] ${name} ${split} IoU=${iou_thresh}"
    "${PYTHON_BIN}" scripts/eval_detection.py \
      --config "${cfg}" \
      --checkpoint "${ckpt}" \
      --split "${split}" \
      --batch-size "${DET_EVAL_BATCH_SIZE}" \
      --num-workers "${DET_EVAL_NUM_WORKERS}" \
      --degraded-eval \
      --iou-thresh "${iou_thresh}" \
      --output-suffix "${DET_DEGRADED_SUFFIX}" \
      --save-predictions
  done
done

"${PYTHON_BIN}" tools/summarize_detection_comparison.py \
  --detection-root "${OUTPUT_ROOT}/metrics/detection" \
  --baseline-root "${OUTPUT_ROOT}/metrics/baselines" \
  --suffix "${DET_DEGRADED_SUFFIX}" \
  --output-prefix "detection_comparison${DET_DEGRADED_SUFFIX}"

echo "[OK] Detection degraded modelwise comparison finished."
