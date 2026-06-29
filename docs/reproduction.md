# Reproduction

## 1. Configure Paths

Edit:

```text
configs/paths/remote_server.yaml
```

Verify paths:

```bash
python tools/check_paths.py --config configs/paths/remote_server.yaml
```

## 2. Prepare Data

Recommended one-command preflight and preparation:

```bash
bash scripts/run_preflight.sh
```

Manual equivalent:

Detection:

```bash
python tools/prepare_detection_data.py --config configs/paths/remote_server.yaml
python tools/check_detection_dataloader.py --config configs/paths/remote_server.yaml --save-vis
python tools/check_db_label_generator.py --config configs/paths/remote_server.yaml --save-vis
```

The detection preparation step reads split and cleaning parameters from `configs/paths/remote_server.yaml`. The current remote setup expects at least 500 raw images and 500 raw JSON files.

Recognition:

```bash
python tools/prepare_recognition_data.py --config configs/paths/remote_server.yaml
python tools/build_charset.py --config configs/paths/remote_server.yaml
python tools/build_transition_matrix.py --config configs/paths/remote_server.yaml
python tools/check_recognition_dataloader.py --config configs/paths/remote_server.yaml
```

## 3. Write Formal 4090 Configs

```bash
python tools/write_4090_yamls.py
```

## 4. Train and Evaluate

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

## 5. Generate Paper Tables

```bash
python tools/generate_paper_tables.py
```

