#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
FULL_OCR_WORKSPACE="${FULL_OCR_WORKSPACE:-full_ocr_workspace}"

DET_CONFIG="configs/detection/dbnetpp_vsaa_asym_shrink.yaml"
DET_CKPT="outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth"
REC_CONFIG="configs/recognition/svtr_official_dab_lortho.yaml"
REC_CKPT="outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth"
if [[ ! -f "${REC_CKPT}" ]]; then
  REC_CKPT="outputs/checkpoints/recognition/svtr_official_dab_lortho/last.pth"
fi
FULL_OCR_INPUT="${FULL_OCR_INPUT:-${FULL_OCR_WORKSPACE}/input}"
FULL_OCR_JSON="${FULL_OCR_JSON:-${FULL_OCR_WORKSPACE}/full_page_predictions.json}"
FULL_OCR_TEXT_DIR="${FULL_OCR_TEXT_DIR:-${FULL_OCR_WORKSPACE}/page_texts}"
FULL_OCR_VIS_DIR="${FULL_OCR_VIS_DIR:-${FULL_OCR_WORKSPACE}}"

mkdir -p "${FULL_OCR_WORKSPACE}" "${FULL_OCR_INPUT}" "${FULL_OCR_TEXT_DIR}"
rm -f "${FULL_OCR_JSON}"
rm -rf "${FULL_OCR_TEXT_DIR}" "${FULL_OCR_VIS_DIR}/annotated" "${FULL_OCR_VIS_DIR}/sequence" "${FULL_OCR_VIS_DIR}/combined"
mkdir -p "${FULL_OCR_TEXT_DIR}" "${FULL_OCR_VIS_DIR}/annotated" "${FULL_OCR_VIS_DIR}/sequence" "${FULL_OCR_VIS_DIR}/combined"

"${PYTHON_BIN}" scripts/infer_full_ocr.py \
  --det-config "${DET_CONFIG}" \
  --det-checkpoint "${DET_CKPT}" \
  --rec-config "${REC_CONFIG}" \
  --rec-checkpoint "${REC_CKPT}" \
  --input "${FULL_OCR_INPUT}" \
  --output "${FULL_OCR_JSON}" \
  --text-output-dir "${FULL_OCR_TEXT_DIR}" \
  --device cuda

"${PYTHON_BIN}" tools/visualize_full_ocr_sequence.py \
  --input-json "${FULL_OCR_JSON}" \
  --output-dir "${FULL_OCR_VIS_DIR}" \
  --text-dir "${FULL_OCR_TEXT_DIR}" \
  --side-by-side

echo "[OK] Full OCR inference finished."
echo "[OK] Input: ${FULL_OCR_INPUT}"
echo "[OK] JSON: ${FULL_OCR_JSON}"
echo "[OK] Page texts: ${FULL_OCR_TEXT_DIR}"
echo "[OK] Annotated images: ${FULL_OCR_VIS_DIR}/annotated"
echo "[OK] Sequence images: ${FULL_OCR_VIS_DIR}/sequence"
echo "[OK] Combined images: ${FULL_OCR_VIS_DIR}/combined"
