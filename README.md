# OCR_Manchu

PyTorch source code for Manchu document OCR experiments. The repository supports word-level text detection, word recognition, ablation studies, external baseline comparisons, and an end-to-end page OCR pipeline.

The codebase is organized for paper reproduction rather than as a minimal demo. It contains runnable project implementations for the proposed models and the comparison baselines used in the experiments.

## Main Contributions Implemented

### Detection

Main detector:

- `DBNet++ + VSAA + AS`
- Config: `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`
- Key modules:
  - DBNet++-style detector: `src/manchu_ocr/models/detection/dbnetpp.py`
  - DBFPN with ASF and VSAA: `src/manchu_ocr/models/detection/necks/db_fpn.py`
  - VSAA module: `src/manchu_ocr/models/detection/modules/vsaa.py`
  - Direction-module counterparts: `src/manchu_ocr/models/detection/modules/directional_attention.py`
  - Asymmetric shrink label generation: `src/manchu_ocr/data/label_generators/db_label_generator.py`

Detection ablation models:

- `DBNet++`: `configs/detection/dbnetpp_official_baseline.yaml`
- `DBNet++ + VSAA`: `configs/detection/dbnetpp_vsaa.yaml`
- `DBNet++ + AS`: `configs/detection/dbnetpp_asym_shrink.yaml`
- `DBNet++ + VSAA + AS`: `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`

VSAA counterpart models (fixed DBNet++ base, only the post-fusion module changes):

- No attention: `configs/detection/dbnetpp_official_baseline.yaml`
- Strip Pooling: `configs/detection/dbnetpp_strip_pooling.yaml`
- Coordinate Attention: `configs/detection/dbnetpp_coordinate_attention.yaml`
- Horizontal-only strip: `configs/detection/dbnetpp_horizontal_strip.yaml`
- Vertical-only strip: `configs/detection/dbnetpp_vertical_strip.yaml`
- VSAA: `configs/detection/dbnetpp_vsaa.yaml`

Detection comparison baselines:

- EAST: `configs/detection/east_baseline.yaml`
- CRAFT: `configs/detection/craft_baseline.yaml`
- DBNet: `configs/detection/dbnet_baseline.yaml`
- PP-OCRv5 Det: `configs/detection/ppocrv5_det_baseline.yaml`
- Hi-SAM: `configs/detection/hisam_baseline.yaml`

### Recognition

Main recognizer:

- `SVTR + DAB + OTP`
- Config: `configs/recognition/svtr_official_dab_lortho.yaml`
- Loss: `L_total = L_CTC + lambda_align * L_align + 0.1 * L_OTP`
- The OTP weight is fixed at `lambda=0.1` from epoch 1 through the end of
  training; there is no OTP activation, warm-up, or decay schedule.
- A checkpoint trained under the former dynamic OTP schedule is not a valid
  fixed-lambda run. Retrain this main experiment from epoch 1 for formal results;
  only resume checkpoints already created with the fixed-0.1 configuration.
- Key modules:
  - SVTR official-style recognizer: `src/manchu_ocr/models/recognition/svtr_official.py`
  - Diacritic-aware branch: `src/manchu_ocr/models/recognition/branches/diacritic_aware_branch.py`
  - Cross-attention fusion: `src/manchu_ocr/models/recognition/modules/cross_attention_fusion.py`
  - Orthographic transition loss: `src/manchu_ocr/losses/orthographic_transition_loss.py`

Recognition ablation models:

- `SVTR`: `configs/recognition/svtr_official_baseline.yaml`
- `SVTR + DAB`: `configs/recognition/svtr_official_dab.yaml`
- `SVTR + OTP`: `configs/recognition/svtr_official_lortho.yaml`
- `SVTR + DAB + OTP`: `configs/recognition/svtr_official_dab_lortho.yaml`

OTP lambda-sensitivity models (Our recognizer with DAB fixed):

- `lambda = 0, 0.025, 0.05, 0.1, 0.2, 0.5`
- Plan: `configs/experiments/rec_otp_lambda_sensitivity.yaml`
- Runner: `scripts/run_rec_otp_lambda_sensitivity.sh`

Recognition comparison baselines:

- CRNN: `configs/recognition/crnn_baseline.yaml`
- PARSeq: `configs/recognition/parseq_baseline.yaml`
- ABINet: `configs/recognition/abinet_baseline.yaml`
- SVTRv2: `configs/recognition/svtrv2_baseline.yaml`
- SVTRv2 + NRTR: `configs/recognition/svtrv2_nrtr_baseline.yaml`
- DCM: `configs/recognition/dcm_baseline.yaml`

## Unified Strict Evaluation Protocol

All models now use the same validation and test definitions. Model-specific
metric tolerances and evaluation degradation profiles have been removed.

Recognition metrics:

- `WA`: exact whole-word match rate.
- `CA = (N - S - D) / N`: correct reference-character rate; insertions do not
  reduce CA because they are not reference characters.
- `CER = (S + D + I) / N`: standard corpus-level edit-distance CER.
- `S`, `D`, and `I` are substitutions, deletions, and insertions from an
  optimal Levenshtein alignment; `N` is the total number of reference
  characters. No per-sample error allowance or positional-character exception
  is deducted.

Detection metrics:

- Validation and test use original split images without synthetic degradation.
- Every detector uses the same one-to-one matching threshold, `IoU=0.75`.
- Training augmentation is unchanged and remains controlled by each training
  config; only validation/test degradation was removed.

The standalone scorer follows the same recognition definitions:

```bash
python3 score_recognition.py predictions.csv \
  --model MODEL_NAME \
  --split test \
  --output recognition_scores.csv
```

Romanized Manchu whitespace is ignored by default as formatting. Use
`--space-policy count` only if spaces are real token-boundary characters.

## Repository Layout

```text
OCR_Manchu/
  configs/                 Experiment, path, and baseline configs
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
- GPU used in formal experiments: RTX 4090 24 GB
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

### VSAA Counterpart Ablation

```bash
bash scripts/run_det_vsaa_counterparts.sh
```

This controlled experiment keeps the DBNet++ backbone, ASF/FPN, head, split,
100 epochs, AdamW settings and 1056×768 input fixed. All six variants use the
same strict validation/test protocol: no evaluation degradation and IoU 0.75.
It starts all six variants from scratch by default; use `AUTO_RESUME=1` only to
continue checkpoints created by the same controlled run.

### Detection Comparison

```bash
bash scripts/run_det_comparison.sh
```

This trains/evaluates:

- EAST
- CRAFT
- DBNet
- PP-OCRv5 Det
- Hi-SAM
- Ours: DBNet++ + VSAA + AS

To evaluate only with existing checkpoints:

```bash
RUN_OURS_TRAIN=0 bash scripts/run_det_comparison.sh
```

### Recognition Ablation

```bash
bash scripts/run_rec_ablation.sh
```

This trains and evaluates:

- SVTR
- SVTR + DAB
- SVTR + OTP
- SVTR + DAB + OTP


Useful overrides:

```bash
REC_EPOCHS=200 REC_BATCH_SIZE=128 REC_DAB_BATCH_SIZE=96 bash scripts/run_rec_ablation.sh
AUTO_RESUME=1 bash scripts/run_rec_ablation.sh
```

### OTP Lambda Sensitivity (Paper Section 3.3)

```bash
bash scripts/run_rec_otp_lambda_sensitivity.sh
```

This six-run sweep keeps the Our-recognizer SVTR+DAB architecture, recognition
dataset, optimizer, input size and training budget fixed. Positive lambda values
are constant from epoch 1; `lambda=0` keeps DAB but disables OTP. Every lambda
value now uses the same exact-match WA and standard CA/CER. The
configured validation interval is 50 epochs (with final-epoch validation), and
`best.pth` is selected by strict WA. The runner evaluates both validation and
test and writes a JSON/CSV/Markdown aggregate.

Useful overrides:

```bash
REC_LAMBDA_BATCH_SIZE=96 REC_LAMBDA_EVAL_BATCH_SIZE=192 \
  bash scripts/run_rec_otp_lambda_sensitivity.sh
SKIP_TRAIN=1 bash scripts/run_rec_otp_lambda_sensitivity.sh
```

### Recognition Comparison

```bash
bash scripts/run_rec_comparison.sh
```

This trains/evaluates:

- CRNN
- PARSeq
- ABINet
- SVTRv2
- SVTRv2 + NRTR
- DCM
- Ours: SVTR + DAB + OTP (`lambda=0.1`, fixed)

To skip training and evaluate existing checkpoints:

```bash
SKIP_TRAIN=1 bash scripts/run_rec_comparison.sh
```

### Final Held-out Test (All Models)

After selecting checkpoints and settings only from validation results, verify the
complete 29-model run plan:

```bash
HELDOUT_DRY_RUN=1 bash scripts/run_all_heldout_tests.sh
```

Then evaluate the held-out test split once:

```bash
bash scripts/run_all_heldout_tests.sh
```

The plan in `configs/experiments/heldout_all_models.yaml` covers all sixteen
recognition and all thirteen detection configurations. Missing checkpoints or test
manifests stop the strict run before evaluation begins. Each result JSON records
the model, `validation`/`test` split, config, checkpoint, manifest, complete
metrics, elapsed time, software versions, and actual GPU information.

## Zero-shot General OCR APIs

Qwen2.5-VL-7B-Instruct, GOT-OCR2.0 and PaddleOCR-VL can be evaluated without
training on the same recognition validation/test word crops:

```bash
python3 -m pip install -r requirements-api.txt
python3 scripts/run_lmm_zero_shot.py --dry-run --max-samples 20 --run-tag smoke20
bash scripts/run_lmm_zero_shot.sh --splits validation --run-tag full_v1
bash scripts/run_lmm_zero_shot.sh --splits test --run-tag full_v1
```

The API runner uses a frozen zero-shot protocol, never sends ground-truth labels,
supports retry/resume without repeating successful requests, and emits no formal
metric result while any API sample remains unresolved. All three models use the
same strict exact WA, reference-character CA, and `(S+D+I)/N` CER. Provider keys,
GOT hosting preparation, costs, output paths, and the required validation-before-
test procedure are documented in
[`docs/lmm_zero_shot_api_experiment.md`](docs/lmm_zero_shot_api_experiment.md).


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

## Quantitative E2E OCR

Run the four DBNet++/Ours-detector x SVTR/Ours-recognizer combinations on the
unchanged validation and held-out test page splits:

```bash
E2E_DRY_RUN=1 bash scripts/run_e2e_ocr_experiments.sh
bash scripts/run_e2e_ocr_experiments.sh
```

The formal protocol uses detector-predicted crops only. Every combination uses
original page images, detection IoU 0.75, exact-match E2E WA, reference-character
CA, and standard `(S+D+I)/N` E2E CER, without any model-specific tolerance. It
also reports column-major page CER/edit similarity, missed words, and false
positives. Configuration is in
`configs/experiments/e2e_ocr.yaml`; comparison outputs are written under
`outputs/metrics/e2e_ocr/`.

Prepare real GT word transcriptions without changing the existing page split:

```bash
python tools/prepare_e2e_transcriptions.py export --save-review-crops
# Fill transcriptions.csv and mark reviewed rows status=done.
python tools/prepare_e2e_transcriptions.py apply
python tools/prepare_e2e_transcriptions.py audit
```

The generated review crops are annotation aids only. Formal OCR inference still
uses detector predictions exclusively. A generic label such as `text` causes
the E2E preflight to stop before GPU inference.

## Reproducing Paper Tables and Figures

After training and evaluation:

```bash
python tools/summarize_detection_formal.py \
  --root outputs/metrics/detection \
  --output-prefix detection_ablation_strict \
  --allow-missing

python tools/summarize_detection_comparison.py \
  --detection-root outputs/metrics/detection \
  --baseline-root outputs/metrics/baselines \
  --output-prefix detection_comparison_strict

python tools/summarize_recognition_formal.py --root outputs/metrics/recognition
python tools/summarize_recognition_comparison.py --root outputs/metrics/recognition
python tools/summarize_external_baselines.py --root outputs/metrics/baselines
python tools/generate_paper_tables.py
```

Important generated files:

```text
outputs/metrics/paper_tables.md
outputs/metrics/detection/detection_ablation_strict.md
outputs/metrics/detection/detection_comparison_strict.md
outputs/metrics/recognition/recognition_ablation_formal.md
outputs/metrics/recognition/recognition_comparison.md
outputs/metrics/baselines/external_baselines.md
```

Generate curves:

```bash
python tools/plot_detection_formal_curves.py
python tools/plot_recognition_formal_curves.py
python tools/plot_recognition_comparison_curves.py
python tools/plot_final_detection_model.py
python tools/plot_final_recognition_model.py
```

Recognition training-loss figures contain three curves for each model: raw CTC
loss, raw OTP loss, and the actual optimized total loss. Training history also
stores `train_weighted_ortho_loss = lambda_ortho * train_ortho_loss` for audit.
For models without OTP, the OTP curve is zero. A positive-lambda `metrics.json`
created before OTP component logging cannot be reconstructed faithfully and must
be regenerated by rerunning training.

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

## Notes on Baseline Implementations

The comparison baselines in this repository are project-runnable implementations aligned with the same dataset, manifests, training loop, and evaluation scripts. They are intended to support controlled experiments in this codebase. Official pretrained weights for third-party models are not bundled.

## Additional Documentation

- Data format: `docs/data_format.md`
- Experiment protocol: `docs/experiment_protocol.md`
- Full OCR pipeline: `docs/full_ocr_pipeline.md`
- Remote server migration: `docs/remote_server_migration.md`
- Method notes: `docs/method_notes.md`
- Reproduction notes: `docs/reproduction.md`

## License

See `LICENSE`.
