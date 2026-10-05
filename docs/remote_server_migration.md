# Remote Server Migration

This project is now configured for a Linux remote server with an RTX 4090-class GPU.

## Default Directory Layout

```text
/root/code/OCR_Manchu
/root/code/Manchu_Detection_Data
/root/code/Manchu_Recognition_Data
```

If your server uses different paths, set these variables before running commands:

```bash
export OCR_MANCHU_PROJECT_ROOT=/root/code/OCR_Manchu
export OCR_MANCHU_DET_ROOT=/root/code/Manchu_Detection_Data
export OCR_MANCHU_REC_ROOT=/root/code/Manchu_Recognition_Data
export OCR_MANCHU_OUTPUT_ROOT=/root/code/OCR_Manchu/outputs
```

The path config is:

```text
configs/paths/remote_server.yaml
```

## Verify Paths

```bash
python tools/check_paths.py
```

## Recommended Preflight

Run this before formal training:

```bash
bash scripts/run_preflight.sh
```

It checks raw data, regenerates processed data, verifies manifests and transition matrix, and validates all formal ablation model configs.

## Prepare Detection Data

```bash
python tools/prepare_detection_data.py
python tools/check_detection_dataloader.py --save-vis
python tools/check_db_label_generator.py --save-vis
```

Expected current split:

```text
train: 400
val: 50
test: 50
```

## Prepare Recognition Data

```bash
python tools/prepare_recognition_data.py
python tools/build_charset.py --config configs/paths/remote_server.yaml
python tools/build_transition_matrix.py --config configs/paths/remote_server.yaml
python tools/check_recognition_dataloader.py
```

## Formal Training

Detection:

```bash
bash scripts/run_det_ablation.sh
bash scripts/run_det_vsaa_counterparts.sh
```

Recognition:

```bash
bash scripts/run_rec_ablation.sh
bash scripts/run_rec_otp_lambda_sensitivity.sh
```

Full OCR:

```bash
bash scripts/run_full_pipeline.sh
```

## Current RTX 4090 Batch Settings

Detection:

```text
batch_size: 16
num_workers: 8
```

Recognition:

```text
batch_size: 256
num_workers: 8
```

If the server reports CUDA OOM, reduce:

```text
detection batch_size: 16 -> 12 -> 8
recognition batch_size: 256 -> 192 -> 128
```

You can reduce batch size without editing YAML:

```bash
DET_BATCH_SIZE=8 DET_EVAL_BATCH_SIZE=4 bash scripts/run_det_ablation.sh
REC_BATCH_SIZE=128 REC_EVAL_BATCH_SIZE=128 bash scripts/run_rec_ablation.sh
REC_LAMBDA_BATCH_SIZE=96 REC_LAMBDA_EVAL_BATCH_SIZE=192 \
  bash scripts/run_rec_otp_lambda_sensitivity.sh
```

## Final Held-out Test for Every Model

The formal held-out plan is:

```text
configs/experiments/heldout_all_models.yaml
```

It explicitly lists all sixteen recognition and all thirteen detection configurations.
Run a strict preflight first; missing manifests or checkpoints stop the entire run
before any test model is evaluated:

```bash
HELDOUT_DRY_RUN=1 bash scripts/run_all_heldout_tests.sh
```

Then run the held-out test once:

```bash
bash scripts/run_all_heldout_tests.sh
```

The runner prefers `best.pth` (selected on validation) and falls back to
`last.pth` only when a model has no saved best checkpoint. The exact checkpoint
path is recorded in every result. To require `best.pth` for all models:

```bash
CHECKPOINT_POLICY=best_only bash scripts/run_all_heldout_tests.sh
```

RTX 4090 evaluation overrides remain available without editing YAML:

```bash
REC_EVAL_BATCH_SIZE=192 DET_EVAL_BATCH_SIZE=4 EVAL_NUM_WORKERS=8 \
  bash scripts/run_all_heldout_tests.sh
```

Per-model outputs remain under `outputs/metrics/{recognition,detection}/<model>/`.
An aggregate execution audit is written under `outputs/metrics/experiment_runs/`.

## Four-Pipeline E2E OCR Evaluation

After the two detector and two recognizer checkpoints exist, validate all page
annotations and commands without loading a model:

```bash
python tools/prepare_e2e_transcriptions.py export --save-review-crops
# Fill processed/e2e/transcriptions.csv and mark every reviewed row done.
python tools/prepare_e2e_transcriptions.py apply
python tools/prepare_e2e_transcriptions.py audit
E2E_DRY_RUN=1 bash scripts/run_e2e_ocr_experiments.sh
```

Then run validation and held-out test for all four combinations:

```bash
bash scripts/run_e2e_ocr_experiments.sh
```

This preserves the current detection validation/test page membership through
enriched E2E manifests, prefers each model's `best.pth`, and records both
checkpoint paths. Every pipeline uses original page images, detector IoU 0.75,
exact-match WA, and standard CA/CER without model-specific tolerance. Generic
`text` labels fail preflight. Results and the aggregate CSV are written under
`outputs/metrics/e2e_ocr/`.

