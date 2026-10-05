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
DET_EVAL_SUFFIX="${DET_EVAL_SUFFIX:-_strict}"

CONFIGS=(
  "configs/detection/dbnetpp_official_baseline.yaml"
  "configs/detection/dbnetpp_vsaa.yaml"
  "configs/detection/dbnetpp_asym_shrink.yaml"
  "configs/detection/dbnetpp_vsaa_asym_shrink.yaml"
)

NAMES=(
  "dbnetpp_official_baseline"
  "dbnetpp_vsaa"
  "dbnetpp_asym_shrink"
  "dbnetpp_vsaa_asym_shrink"
)

for idx in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$idx]}"
  name="${NAMES[$idx]}"
  ckpt="${OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[WARN] Missing checkpoint, skip strict eval: ${ckpt}"
    continue
  fi

  for split in val test; do
    echo "[STRICT EVAL] ${name} ${split} IoU=0.75"
    "${PYTHON_BIN}" scripts/eval_detection.py \
      --config "${cfg}" \
      --checkpoint "${ckpt}" \
      --split "${split}" \
      --batch-size "${DET_EVAL_BATCH_SIZE}" \
      --num-workers "${DET_EVAL_NUM_WORKERS}" \
      --output-suffix "${DET_EVAL_SUFFIX}"
  done
done

"${PYTHON_BIN}" tools/summarize_detection_formal.py \
  --root "${OUTPUT_ROOT}/metrics/detection" \
  --suffix "${DET_EVAL_SUFFIX}" \
  --output-prefix "detection_ablation${DET_EVAL_SUFFIX}" \
  --allow-missing

FINAL_EXP="dbnetpp_vsaa_asym_shrink"
if [[ -f "${OUTPUT_ROOT}/metrics/detection/${FINAL_EXP}/metrics.json" \
  && -f "${OUTPUT_ROOT}/metrics/detection/${FINAL_EXP}/eval_val${DET_EVAL_SUFFIX}.json" \
  && -f "${OUTPUT_ROOT}/metrics/detection/${FINAL_EXP}/eval_test${DET_EVAL_SUFFIX}.json" ]]; then
  "${PYTHON_BIN}" tools/plot_final_detection_model.py \
    --root "${OUTPUT_ROOT}/metrics/detection" \
    --out-dir "${OUTPUT_ROOT}/visualizations/paper_figures/main_models" \
    --exp-name "${FINAL_EXP}" \
    --label "DBNet++ + VSAA + AS" \
    --suffix "${DET_EVAL_SUFFIX}"
else
  echo "[WARN] Skip final detection plots because required metrics files are missing."
fi

echo "[OK] Detection strict ablation finished."
