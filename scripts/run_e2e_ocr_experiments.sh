#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${OCR_MANCHU_PROJECT_ROOT:-/root/code/OCR_Manchu}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

args=()
if [[ "${E2E_DRY_RUN:-0}" == "1" ]]; then
  args+=(--dry-run)
fi

"${PYTHON_BIN}" scripts/run_e2e_ocr_experiments.py "${args[@]}"
