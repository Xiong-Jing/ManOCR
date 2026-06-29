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
```

Recognition:

```bash
bash scripts/run_rec_ablation.sh
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
```

