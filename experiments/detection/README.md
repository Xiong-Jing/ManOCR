# Detection Experiments

Formal detection experiments are defined in:

```text
configs/experiments/det_ablation.yaml
```

Run:

```bat
scripts\run_det_ablation.bat
```

Final metrics are written to:

```text
outputs/metrics/detection/detection_ablation_formal.md
outputs/metrics/detection/detection_ablation_formal.json
```

Final held-out test for all detection models:

```bash
HELDOUT_TASKS=detection bash scripts/run_all_heldout_tests.sh
```

The same `scripts/eval_detection.py` implementation is used for validation and
test; only `--split val` versus `--split test` changes.
