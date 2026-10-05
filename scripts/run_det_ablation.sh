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
DET_EVAL_SUFFIX="${DET_EVAL_SUFFIX:-_strict}"

CONFIGS=(
  configs/detection/dbnetpp_official_baseline.yaml
  configs/detection/dbnetpp_vsaa.yaml
  configs/detection/dbnetpp_asym_shrink.yaml
  configs/detection/dbnetpp_vsaa_asym_shrink.yaml
)
NAMES=(
  dbnetpp_official_baseline
  dbnetpp_vsaa
  dbnetpp_asym_shrink
  dbnetpp_vsaa_asym_shrink
)
for cfg in "${CONFIGS[@]}"; do
  echo "[TRAIN] ${cfg}"
  train_args=(--config "${cfg}")
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
done

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  for split in val test; do
    echo "[STRICT EVAL] ${name} ${split} IoU=0.75"
    "${PYTHON_BIN}" scripts/eval_detection.py \
      --config "configs/detection/${name}.yaml" \
      --checkpoint "${OUTPUT_ROOT}/checkpoints/detection/${name}/best.pth" \
      --split "${split}" \
      --batch-size "${DET_EVAL_BATCH_SIZE}" \
      --num-workers "${DET_EVAL_NUM_WORKERS}" \
      --output-suffix "${DET_EVAL_SUFFIX}" \
      --save-predictions
  done
done

"${PYTHON_BIN}" tools/summarize_detection_formal.py \
  --root "${OUTPUT_ROOT}/metrics/detection" \
  --suffix "${DET_EVAL_SUFFIX}" \
  --output-prefix "detection_ablation${DET_EVAL_SUFFIX}" \
  --allow-missing
"${PYTHON_BIN}" tools/plot_detection_formal_curves.py \
  --root "${OUTPUT_ROOT}/metrics/detection" \
  --out-dir "${OUTPUT_ROOT}/visualizations/paper_figures"

"${PYTHON_BIN}" tools/plot_final_detection_model.py \
  --root "${OUTPUT_ROOT}/metrics/detection" \
  --out-dir "${OUTPUT_ROOT}/visualizations/paper_figures/main_models" \
  --exp-name "dbnetpp_vsaa_asym_shrink" \
  --label "DBNet++ + VSAA + AS" \
  --suffix "${DET_EVAL_SUFFIX}"

echo "[OK] Detection ablation finished."
