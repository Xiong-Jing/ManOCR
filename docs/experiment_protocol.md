# Experiment Protocol

## Hardware

Formal experiments use RTX 4090 settings:

- Recognition evaluation batch size: 256
- Detection evaluation batch size: 8
- CUDA required

Both values can be reduced through environment variables if the 24 GB GPU runs
out of memory. Reducing evaluation batch size does not change the metric logic.

## Validation and Held-out Test

Validation and test are evaluated by the same task evaluator. Model loading,
decoding/post-processing, metric configuration, batch size, and worker count stay
fixed; only `val_list` versus `test_list` changes. Checkpoints must be selected
without consulting test performance.

All detection models use original images and fixed IoU 0.75 on validation/test.
All recognition models use exact-match WA, `(N-S-D)/N` CA, and `(S+D+I)/N`
CER. The runner does not introduce model-specific tolerances or test-specific
thresholds.

Audit all commands without running inference:

```bash
HELDOUT_DRY_RUN=1 bash scripts/run_all_heldout_tests.sh
```

Run the final held-out test for all sixteen recognition and all thirteen detection models:

```bash
bash scripts/run_all_heldout_tests.sh
```

Every `eval_val.json` or `eval_test.json` uses the common
`ocr_manchu.experiment_result.v1` schema and includes model name, public split
name (`validation` or `test`), config/checkpoint/manifest paths, metrics, runtime,
host, Python/PyTorch/CUDA versions, and measured GPU information.

The hosted zero-shot OCR group is separate from checkpoint-based training runs.
It evaluates Qwen2.5-VL-7B-Instruct, GOT-OCR2.0 and PaddleOCR-VL on the recognition
word-crop manifests with no training. It uses zero-tolerance exact WA, reference-
character CA and edit-distance CER for every provider, records the API model/task
instead of a local checkpoint, and refuses to publish metrics until all API calls
succeed. See `docs/lmm_zero_shot_api_experiment.md`.

Minimal result structure:

```json
{
  "schema_version": "ocr_manchu.experiment_result.v1",
  "model_name": "svtr_official_dab_lortho",
  "split": "test",
  "checkpoint_path": ".../best.pth",
  "manifest_path": ".../processed/test.txt",
  "metrics": {},
  "runtime_seconds": 0.0,
  "runtime": {}
}
```

## Detection

Run:

```bat
scripts\run_det_ablation.bat
```

Experiments:

- DBNet++
- DBNet++ + VSAA
- DBNet++ + AS
- DBNet++ + VSAA + AS

VSAA counterpart experiments are kept in a separate controlled plan so they do
not get mixed with the VSAA/AS factorial ablation:

```bash
bash scripts/run_det_vsaa_counterparts.sh
```

Every counterpart uses the same no-degradation, IoU 0.75 evaluation protocol.

## Recognition

Run:

```bat
scripts\run_rec_ablation.bat
```

Experiments:

- SVTR
- SVTR + DAB
- SVTR + OTP
- SVTR + DAB + OTP

The Section 3.3 OTP-weight sweep is a separate controlled recognition ablation:

```bash
bash scripts/run_rec_otp_lambda_sensitivity.sh
```

It trains `lambda = 0, 0.025, 0.05, 0.1, 0.2, 0.5` on the recognition dataset.
All architecture/training fields are copied from Our recognizer; only OTP
enablement and its constant weight change. Every lambda uses the same strict
recognition metrics. The main `SVTR + DAB + OTP` configuration fixes lambda at
0.1 from epoch 1 with no OTP warm-up or decay.

## Full OCR

Run:

```bat
scripts\run_full_pipeline.bat
```

## Quantitative E2E OCR

The four detector-recognizer combinations and RTX 4090 paths are defined in
`configs/experiments/e2e_ocr.yaml`. Each pipeline uses original page images,
fixed detector IoU 0.75, exact-match E2E WA, and standard edit-distance CER.
Validation and test differ only in the enriched
E2E manifest (`val_list` or `test_list`), whose page membership is copied from
the corresponding detection split; detector/recognizer configs, selected checkpoints, post-processing,
crop padding, reading-order logic, matching, and metrics stay identical.

```bash
E2E_DRY_RUN=1 bash scripts/run_e2e_ocr_experiments.sh
bash scripts/run_e2e_ocr_experiments.sh
```

No ground-truth box is used to produce a recognition crop. Ground-truth boxes
are loaded only for post-inference matching. All GT words need a transcription.
The runner fails its data preflight when annotations contain only generic
detection labels.

Generate the same-split enriched manifests before the E2E run:

```bash
python tools/prepare_e2e_transcriptions.py export --save-review-crops
python tools/prepare_e2e_transcriptions.py apply
python tools/prepare_e2e_transcriptions.py audit
```
