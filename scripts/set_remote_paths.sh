#!/usr/bin/env bash
# Source this file; executing it in a child shell cannot update the caller.
# Usage: source /root/code/OCR_Manchu/scripts/set_remote_paths.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use: source /root/code/OCR_Manchu/scripts/set_remote_paths.sh" >&2
  exit 2
fi

export OCR_MANCHU_PROJECT_ROOT=/root/code/OCR_Manchu
export OCR_MANCHU_DET_ROOT=/root/code/Manchu_Detection_Data
export OCR_MANCHU_REC_ROOT=/root/code/Manchu_Recognition_Data
export OCR_MANCHU_OUTPUT_ROOT=/root/code/OCR_Manchu/outputs
export OCR_MANCHU_PATHS_CONFIG=/root/code/OCR_Manchu/configs/paths/remote_server.yaml
export PYTHONPATH="${OCR_MANCHU_PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

printf '[PATHS] project=%s\n[PATHS] detection=%s\n[PATHS] recognition=%s\n[PATHS] outputs=%s\n' \
  "${OCR_MANCHU_PROJECT_ROOT}" "${OCR_MANCHU_DET_ROOT}" \
  "${OCR_MANCHU_REC_ROOT}" "${OCR_MANCHU_OUTPUT_ROOT}"
