# Manchu-OCR Core Code Package

This directory is a code-only copy of the Manchu OCR research project, prepared
on 2026-10-05. It is intended for source review, experiment reproduction, and
transfer without bundled experiment outputs or trained parameters.

## Included

- `src/`: detection, recognition, end-to-end OCR, losses, metrics, and utilities
- `scripts/`: training, validation, testing, inference, and experiment runners
- `tools/`: data preparation, diagnostics, summaries, and plotting code
- `configs/`: model, comparison, ablation, path, and experiment configurations
- `tests/`: automated tests for the retained code
- `docs/` and `experiments/`: experiment and reproduction documentation
- `data/`: path templates and text manifests only; no image datasets
- `resources/`: source/templates required to rebuild language resources
- root-level packaging, dependency, scoring, and project documentation files

## Excluded

- the complete `outputs/` tree, including checkpoints, metrics, logs,
  predictions, tables, plots, and exported paper results
- all trained weights and serialized model files (`.pth`, `.pt`, `.ckpt`,
  `.onnx`, `.engine`, `.safetensors`, and `.h5`)
- `full_ocr_workspace/`, which contains sample images and generated predictions
- image, PDF, CSV/TSV, NumPy binary, log, and cache artifacts
- Python bytecode, pytest caches, IDE settings, and generated package metadata
- the external Manchu detection and recognition datasets

The dataset and output paths in configuration files are retained as
reproduction settings. Update the relevant files under `configs/paths/` when
running the package in a different environment.
