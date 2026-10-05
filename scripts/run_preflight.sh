#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON:-python}"
PATHS_CONFIG="${OCR_MANCHU_PATHS_CONFIG:-configs/paths/remote_server.yaml}"

echo "[PREFLIGHT] Raw data and configuration"
"${PYTHON_BIN}" tools/preflight_remote.py \
  --config "${PATHS_CONFIG}" \
  --stage raw

echo "[PREPARE] Detection data"
"${PYTHON_BIN}" tools/prepare_detection_data.py \
  --config "${PATHS_CONFIG}"

echo "[PREPARE] Recognition data"
"${PYTHON_BIN}" tools/prepare_recognition_data.py \
  --config "${PATHS_CONFIG}"

echo "[PREFLIGHT] Processed data, model configs, and experiment scripts"
"${PYTHON_BIN}" tools/preflight_remote.py \
  --config "${PATHS_CONFIG}" \
  --stage all

echo "[OK] Preflight and data preparation finished."
