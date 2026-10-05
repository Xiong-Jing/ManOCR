#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
REC_EVAL_BATCH_SIZE="${REC_EVAL_BATCH_SIZE:-256}"
REC_EVAL_NUM_WORKERS="${REC_EVAL_NUM_WORKERS:-8}"
REC_DECODE_MODE="${REC_DECODE_MODE:-lexicon}"
REC_DECODE_MAX_EDIT_DISTANCE="${REC_DECODE_MAX_EDIT_DISTANCE:-2}"
REC_DECODE_LENGTH_DELTA="${REC_DECODE_LENGTH_DELTA:-2}"
REC_CTC_RERANK_MAX_EDIT_DISTANCE="${REC_CTC_RERANK_MAX_EDIT_DISTANCE:-2}"
REC_CTC_RERANK_LENGTH_DELTA="${REC_CTC_RERANK_LENGTH_DELTA:-2}"
REC_CTC_RERANK_MAX_CANDIDATES="${REC_CTC_RERANK_MAX_CANDIDATES:-40}"
REC_CTC_RERANK_PRIOR_WEIGHT="${REC_CTC_RERANK_PRIOR_WEIGHT:-0.04}"
REC_EVAL_MAX_BATCHES="${REC_EVAL_MAX_BATCHES:-}"

CONFIGS=(
  configs/recognition/svtr_official_baseline.yaml
  configs/recognition/svtr_official_dab.yaml
  configs/recognition/svtr_official_lortho.yaml
  configs/recognition/svtr_official_dab_lortho.yaml
)
NAMES=(
  svtr_official_baseline
  svtr_official_dab
  svtr_official_lortho
  svtr_official_dab_lortho
)

append_eval_args() {
  local -n args_ref=$1

  if [[ -n "${REC_EVAL_MAX_BATCHES}" ]]; then
    args_ref+=(--max-batches "${REC_EVAL_MAX_BATCHES}")
  fi

  case "${REC_DECODE_MODE}" in
    lexicon)
      args_ref+=(
        --disable-ctc-rerank
        --decode-max-edit-distance "${REC_DECODE_MAX_EDIT_DISTANCE}"
        --decode-length-delta "${REC_DECODE_LENGTH_DELTA}"
      )
      ;;
    ctc)
      args_ref+=(
        --enable-ctc-rerank
        --ctc-rerank-max-edit-distance "${REC_CTC_RERANK_MAX_EDIT_DISTANCE}"
        --ctc-rerank-length-delta "${REC_CTC_RERANK_LENGTH_DELTA}"
        --ctc-rerank-max-candidates "${REC_CTC_RERANK_MAX_CANDIDATES}"
        --ctc-rerank-prior-weight "${REC_CTC_RERANK_PRIOR_WEIGHT}"
      )
      ;;
    greedy)
      args_ref+=(--disable-lexicon)
      ;;
    *)
      echo "[ERROR] Unsupported REC_DECODE_MODE=${REC_DECODE_MODE}. Use lexicon, ctc, or greedy." >&2
      exit 1
      ;;
  esac
}

for idx in "${!NAMES[@]}"; do
  name="${NAMES[$idx]}"
  config="${CONFIGS[$idx]}"
  checkpoint="${OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"

  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${OUTPUT_ROOT}/checkpoints/recognition/${name}/last.pth"
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[WARN] Missing checkpoint, skip eval: ${name}"
    continue
  fi

  for split in val test; do
    echo "[EVAL] ${name} ${split} checkpoint=${checkpoint} decode=${REC_DECODE_MODE}"
    eval_args=(
      --config "${config}"
      --checkpoint "${checkpoint}"
      --split "${split}"
      --batch-size "${REC_EVAL_BATCH_SIZE}"
      --num-workers "${REC_EVAL_NUM_WORKERS}"
      --save-predictions
    )
    append_eval_args eval_args
    "${PYTHON_BIN}" scripts/eval_recognition.py "${eval_args[@]}"
  done
done

"${PYTHON_BIN}" tools/summarize_recognition_formal.py --root "${OUTPUT_ROOT}/metrics/recognition"
"${PYTHON_BIN}" tools/plot_recognition_formal_curves.py \
  --root "${OUTPUT_ROOT}/metrics/recognition" \
  --out-dir "${OUTPUT_ROOT}/visualizations/paper_figures"

echo "[OK] Recognition ablation eval-only finished."
