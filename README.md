# OCR_Manchu

PyTorch source code for Manchu document OCR experiments. The repository supports word-level text detection, word recognition, ablation studies, and an end-to-end page OCR pipeline.

The codebase is organized for paper reproduction rather than as a minimal demo. It contains runnable project implementations for the proposed detection and recognition models used in the formal ablation experiments.

## Main Contributions Implemented

### Detection

Main detector:

- `DBNet++ + VSAA + AS`
- Config: `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`
- Key modules:
  - DBNet++-style detector: `src/manchu_ocr/models/detection/dbnetpp.py`
  - DBFPN with ASF and VSAA: `src/manchu_ocr/models/detection/necks/db_fpn.py`
  - VSAA module: `src/manchu_ocr/models/detection/modules/vsaa.py`
  - Asymmetric shrink label generation: `src/manchu_ocr/data/label_generators/db_label_generator.py`

Detection ablation models:

- `DBNet++`: `configs/detection/dbnetpp_official_baseline.yaml`
- `DBNet++ + VSAA`: `configs/detection/dbnetpp_vsaa.yaml`
- `DBNet++ + AS`: `configs/detection/dbnetpp_asym_shrink.yaml`
- `DBNet++ + VSAA + AS`: `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`

### Recognition

Main recognizer:

- `SVTR + DAB + Lortho`
- Config: `configs/recognition/svtr_official_dab_lortho.yaml`
- Key modules:
  - SVTR recognizer: `src/manchu_ocr/models/recognition/svtr_official.py`
  - Diacritic-aware branch: `src/manchu_ocr/models/recognition/branches/diacritic_aware_branch.py`
  - Cross-attention fusion: `src/manchu_ocr/models/recognition/modules/cross_attention_fusion.py`
  - Orthographic transition loss: `src/manchu_ocr/losses/orthographic_transition_loss.py`

Recognition ablation models:

- `SVTR`: `configs/recognition/svtr_official_baseline.yaml`
- `SVTR + DAB`: `configs/recognition/svtr_official_dab.yaml`
- `SVTR + Lortho`: `configs/recognition/svtr_official_lortho.yaml`
- `SVTR + DAB + Lortho`: `configs/recognition/svtr_official_dab_lortho.yaml`

## Repository Layout

```text
OCR_Manchu/
  configs/                 Experiment, path, and model configs
  scripts/                 Training, evaluation, inference, and pipeline scripts
  tools/                   Data preparation, summaries, diagnostics, and plotting tools
  src/manchu_ocr/          Main Python package
  docs/                    Data format, protocol, migration, and method notes
  resources/               Charset and orthographic resources
  data/manifests/          Example manifest files
  full_ocr_workspace/      Local full-page OCR input/output workspace
  outputs/                 Metrics, logs, predictions, figures, and checkpoints
```

## Environment

Recommended setup:

- Python >= 3.9
- PyTorch with CUDA
- Reference GPU: NVIDIA GeForce RTX 4090 24 GB
- Reference CUDA version: 12.8
- Linux server recommended for full training

Install:

```bash
cd /root/code/OCR_Manchu
pip install -e .
pip install -r requirements.txt
```

For Windows local checks, install the same dependencies in a conda or venv environment and run commands from the repository root.

## Data Layout

Default remote layout:

```text
/root/code/OCR_Manchu
/root/code/Manchu_Detection_Data
/root/code/Manchu_Recognition_Data
```

Default local Windows layout:

```text
C:/Users/ahs/Desktop/OCR_Manchu
C:/Users/ahs/Desktop/Manchu_Detection_Data
C:/Users/ahs/Desktop/Manchu_Recognition_Data
```

Path configs:

- Remote: `configs/paths/remote_server.yaml`
- Local Windows: `configs/paths/local_windows.yaml`

You can override remote paths without editing YAML:

```bash
export OCR_MANCHU_PROJECT_ROOT=/root/code/OCR_Manchu
export OCR_MANCHU_DET_ROOT=/root/code/Manchu_Detection_Data
export OCR_MANCHU_REC_ROOT=/root/code/Manchu_Recognition_Data
export OCR_MANCHU_OUTPUT_ROOT=/root/code/OCR_Manchu/outputs
```

### Detection Data

Expected structure:

```text
Manchu_Detection_Data/
  raw/
    images/
    annotations_json/
  processed/
    annotations_clean/
    train.txt
    val.txt
    test.txt
```

Detection manifest format:

```text
image_path<TAB>clean_annotation_json
```

Clean annotation JSON contains image metadata and word-level polygons. In the current dataset, the annotation label value can be simply `manchu`; detection training uses polygon geometry rather than class labels.

### Recognition Data

Expected structure:

```text
Manchu_Recognition_Data/
  raw/
    word_images/
    labels.xlsx
  processed/
    labels.csv
    charset.txt
    transition_matrix.npy
    train.txt
    val.txt
    test.txt
```

Recognition manifest format:

```text
image_path<TAB>text_label
```

## Data Preparation and Preflight

Run the full preflight on the remote server before formal training:

```bash
bash scripts/run_preflight.sh
```

This command checks raw paths, prepares detection data, prepares recognition data, checks processed manifests, verifies the transition matrix, and validates model configs.

Manual detection preparation:

```bash
python tools/prepare_detection_data.py --config configs/paths/remote_server.yaml
python tools/check_detection_dataloader.py --config configs/paths/remote_server.yaml --save-vis
python tools/check_db_label_generator.py --config configs/paths/remote_server.yaml --save-vis
```

Manual recognition preparation:

```bash
python tools/prepare_recognition_data.py --config configs/paths/remote_server.yaml
python tools/build_charset.py --config configs/paths/remote_server.yaml
python tools/build_transition_matrix.py --config configs/paths/remote_server.yaml
python tools/check_recognition_dataloader.py --config configs/paths/remote_server.yaml
```

## Training and Evaluation

All shell scripts set `PYTHONPATH` automatically. Run them from the repository root.

### Detection Ablation

```bash
bash scripts/run_det_ablation.sh
```

This trains and evaluates:

- DBNet++
- DBNet++ + VSAA
- DBNet++ + AS
- DBNet++ + VSAA + AS


Useful overrides:

```bash
DET_EPOCHS=100 DET_BATCH_SIZE=1 DET_EVAL_BATCH_SIZE=8 bash scripts/run_det_ablation.sh
AUTO_RESUME=1 bash scripts/run_det_ablation.sh
```

### Recognition Ablation

```bash
bash scripts/run_rec_ablation.sh
```

This trains and evaluates:

- SVTR
- SVTR + DAB
- SVTR + Lortho
- SVTR + DAB + Lortho


Useful overrides:

```bash
REC_EPOCHS=200 REC_BATCH_SIZE=128 REC_DAB_BATCH_SIZE=96 bash scripts/run_rec_ablation.sh
AUTO_RESUME=1 bash scripts/run_rec_ablation.sh
```

## Full OCR Inference

The full OCR pipeline performs page-level detection, crop rectification, recognition, text output, and visualization.

Default workspace:

```text
full_ocr_workspace/
  input/       Put page images here
  annotated/   Red detection boxes on original pages
  sequence/    Recognized text sequence visualizations
  combined/    Combined visualization images
  page_texts/  Text output per page
```

Run:

```bash
bash scripts/run_full_pipeline.sh
```

The script uses:

- Detection checkpoint: `outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth`
- Recognition checkpoint: `outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth`

Override input/output workspace:

```bash
FULL_OCR_WORKSPACE=full_ocr_workspace bash scripts/run_full_pipeline.sh
```

## Reproducing Paper Tables and Figures

After training and evaluation:

```bash
python tools/summarize_detection_formal.py \
  --root outputs/metrics/detection \
  --suffix _degraded_modelwise \
  --output-prefix detection_ablation_degraded_modelwise \
  --allow-missing

python tools/summarize_recognition_formal.py --root outputs/metrics/recognition
python tools/generate_paper_tables.py
```

Important generated files:

```text
outputs/metrics/paper_tables.md
outputs/metrics/detection/detection_ablation_degraded_modelwise.md
outputs/metrics/recognition/recognition_ablation_formal.md
```

Generate curves:

```bash
python tools/plot_detection_formal_curves.py
python tools/plot_recognition_formal_curves.py
python tools/plot_final_detection_model.py
python tools/plot_final_recognition_model.py
```

Figure outputs are saved under:

```text
outputs/visualizations/paper_figures/
outputs/visualizations/paper_figures/main_models/
```


## Checkpoints and Large Files

The training and evaluation scripts expect checkpoints under:

```text
outputs/checkpoints/detection/<experiment>/best.pth
outputs/checkpoints/recognition/<experiment>/best.pth
```

Large checkpoints and datasets may be excluded from a public source-code release. If checkpoints are not included, rerun the training scripts before evaluation or full OCR inference.

## Additional Documentation

- Data format: `docs/data_format.md`
- Experiment protocol: `docs/experiment_protocol.md`
- Full OCR pipeline: `docs/full_ocr_pipeline.md`
- Remote server migration: `docs/remote_server_migration.md`
- Method notes: `docs/method_notes.md`
- Reproduction notes: `docs/reproduction.md`

## License

See `LICENSE`.
