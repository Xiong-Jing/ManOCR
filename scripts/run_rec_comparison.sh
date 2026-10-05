#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON:-python}"
OUTPUT_ROOT="${OCR_MANCHU_OUTPUT_ROOT:-outputs}"
REC_BATCH_SIZE="${REC_BATCH_SIZE:-}"
REC_MAX_TRAIN_BATCH_SIZE="${REC_MAX_TRAIN_BATCH_SIZE:-128}"
REC_DAB_BATCH_SIZE="${REC_DAB_BATCH_SIZE:-96}"
REC_NUM_WORKERS="${REC_NUM_WORKERS:-}"
REC_EPOCHS="${REC_EPOCHS:-}"
REC_EVAL_BATCH_SIZE="${REC_EVAL_BATCH_SIZE:-256}"
REC_EVAL_NUM_WORKERS="${REC_EVAL_NUM_WORKERS:-8}"
AUTO_RESUME="${AUTO_RESUME:-1}"
REC_DECODE_MODE="${REC_DECODE_MODE:-lexicon}"
REC_DECODE_MAX_EDIT_DISTANCE="${REC_DECODE_MAX_EDIT_DISTANCE:-2}"
REC_DECODE_LENGTH_DELTA="${REC_DECODE_LENGTH_DELTA:-2}"
REC_CTC_RERANK_MAX_EDIT_DISTANCE="${REC_CTC_RERANK_MAX_EDIT_DISTANCE:-2}"
REC_CTC_RERANK_LENGTH_DELTA="${REC_CTC_RERANK_LENGTH_DELTA:-2}"
REC_CTC_RERANK_MAX_CANDIDATES="${REC_CTC_RERANK_MAX_CANDIDATES:-40}"
REC_CTC_RERANK_PRIOR_WEIGHT="${REC_CTC_RERANK_PRIOR_WEIGHT:-0.04}"
REC_EVAL_MAX_BATCHES="${REC_EVAL_MAX_BATCHES:-}"
REC_COMPARISON_WORD_EDIT_DISTANCE="${REC_COMPARISON_WORD_EDIT_DISTANCE:-}"
REC_COMPARISON_CHAR_EDIT_DISTANCE="${REC_COMPARISON_CHAR_EDIT_DISTANCE:-}"
RUN_OURS_TRAIN="${RUN_OURS_TRAIN:-auto}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"

TRAIN_CONFIGS=(
  "configs/recognition/crnn_baseline.yaml"
  "configs/recognition/parseq_baseline.yaml"
  "configs/recognition/abinet_baseline.yaml"
  "configs/recognition/svtrv2_baseline.yaml"
  "configs/recognition/svtrv2_nrtr_baseline.yaml"
  "configs/recognition/dcm_baseline.yaml"
)

EVAL_CONFIGS=(
  "configs/recognition/crnn_baseline.yaml"
  "configs/recognition/parseq_baseline.yaml"
  "configs/recognition/abinet_baseline.yaml"
  "configs/recognition/svtrv2_baseline.yaml"
  "configs/recognition/svtrv2_nrtr_baseline.yaml"
  "configs/recognition/dcm_baseline.yaml"
  "configs/recognition/svtr_official_dab_lortho.yaml"
)

EVAL_NAMES=(
  "crnn_baseline"
  "parseq_baseline"
  "abinet_baseline"
  "svtrv2_baseline"
  "svtrv2_nrtr_baseline"
  "dcm_baseline"
  "svtr_official_dab_lortho"
)

OURS_CONFIG="configs/recognition/svtr_official_dab_lortho.yaml"
OURS_NAME="svtr_official_dab_lortho"
OURS_CKPT="${OUTPUT_ROOT}/checkpoints/recognition/${OURS_NAME}/best.pth"
if [[ ! -f "${OURS_CKPT}" ]]; then
  OURS_CKPT="${OUTPUT_ROOT}/checkpoints/recognition/${OURS_NAME}/last.pth"
fi

if [[ "${RUN_OURS_TRAIN}" == "1" ]]; then
  TRAIN_CONFIGS+=("${OURS_CONFIG}")
elif [[ "${RUN_OURS_TRAIN}" == "auto" && ! -f "${OURS_CKPT}" ]]; then
  TRAIN_CONFIGS+=("${OURS_CONFIG}")
else
  echo "[INFO] Skip training ours. RUN_OURS_TRAIN=${RUN_OURS_TRAIN}, checkpoint=${OURS_CKPT}"
fi

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

train_model() {
  local config="$1"

  echo "[TRAIN] ${config}"
  local train_args=(--config "${config}")

  local effective_batch_size
  if [[ "${config}" == *"svtr_official_dab.yaml" || "${config}" == *"svtr_official_dab_lortho.yaml" ]]; then
    effective_batch_size="${REC_DAB_BATCH_SIZE}"
  else
    effective_batch_size="${REC_BATCH_SIZE:-${REC_MAX_TRAIN_BATCH_SIZE}}"
  fi

  if (( effective_batch_size > REC_MAX_TRAIN_BATCH_SIZE )); then
    echo "[WARN] REC_BATCH_SIZE=${effective_batch_size} exceeds safe cap ${REC_MAX_TRAIN_BATCH_SIZE}; using ${REC_MAX_TRAIN_BATCH_SIZE}"
    effective_batch_size="${REC_MAX_TRAIN_BATCH_SIZE}"
  fi
  echo "[INFO] Effective train batch size for ${config}: ${effective_batch_size}"
  train_args+=(--batch-size "${effective_batch_size}")

  if [[ -n "${REC_NUM_WORKERS}" ]]; then
    train_args+=(--num-workers "${REC_NUM_WORKERS}")
  fi
  if [[ -n "${REC_EPOCHS}" ]]; then
    train_args+=(--epochs "${REC_EPOCHS}")
  fi
  if [[ "${AUTO_RESUME}" == "1" ]]; then
    train_args+=(--auto-resume)
  fi

  "${PYTHON_BIN}" scripts/train_recognition.py "${train_args[@]}"
}

eval_model() {
  local config="$1"
  local name="$2"
  local split="$3"
  local checkpoint="${OUTPUT_ROOT}/checkpoints/recognition/${name}/best.pth"

  if [[ ! -f "${checkpoint}" ]]; then
    checkpoint="${OUTPUT_ROOT}/checkpoints/recognition/${name}/last.pth"
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[WARN] Missing checkpoint, skip eval: ${checkpoint}"
    return
  fi

  echo "[EVAL] ${name} ${split} checkpoint=${checkpoint} decode=${REC_DECODE_MODE}"
  local eval_args=(
    --config "${config}"
    --checkpoint "${checkpoint}"
    --split "${split}"
    --batch-size "${REC_EVAL_BATCH_SIZE}"
    --num-workers "${REC_EVAL_NUM_WORKERS}"
    --save-predictions
  )

  if [[ -n "${REC_COMPARISON_WORD_EDIT_DISTANCE}" ]]; then
    eval_args+=(--word-accuracy-edit-distance "${REC_COMPARISON_WORD_EDIT_DISTANCE}")
  fi

  if [[ -n "${REC_COMPARISON_CHAR_EDIT_DISTANCE}" ]]; then
    eval_args+=(--character-accuracy-edit-distance "${REC_COMPARISON_CHAR_EDIT_DISTANCE}")
  fi

  append_eval_args eval_args
  "${PYTHON_BIN}" scripts/eval_recognition.py "${eval_args[@]}"
}

if [[ "${SKIP_TRAIN}" == "1" ]]; then
  echo "[INFO] Skip all recognition comparison training. SKIP_TRAIN=1"
else
  for config in "${TRAIN_CONFIGS[@]}"; do
    train_model "${config}"
  done
fi

for idx in "${!EVAL_CONFIGS[@]}"; do
  for split in val test; do
    eval_model "${EVAL_CONFIGS[$idx]}" "${EVAL_NAMES[$idx]}" "${split}"
  done
done

"${PYTHON_BIN}" tools/summarize_recognition_comparison.py \
  --root "${OUTPUT_ROOT}/metrics/recognition"

"${PYTHON_BIN}" tools/summarize_external_baselines.py \
  --root "${OUTPUT_ROOT}/metrics/baselines"

"${PYTHON_BIN}" tools/plot_recognition_comparison_curves.py \
  --root "${OUTPUT_ROOT}/metrics/recognition" \
  --out-dir "${OUTPUT_ROOT}/visualizations/paper_figures"

echo "[OK] Recognition comparison finished."
