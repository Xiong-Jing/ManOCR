# Recognition Experiments

Formal recognition experiments are defined in:

```text
configs/experiments/rec_ablation.yaml
```

Run:

```bat
scripts\run_rec_ablation.bat
```

Final metrics are written to:

```text
outputs/metrics/recognition/recognition_ablation_formal.md
outputs/metrics/recognition/recognition_ablation_formal.json
```

Final held-out test for all recognition models:

```bash
HELDOUT_TASKS=recognition bash scripts/run_all_heldout_tests.sh
```

The same `scripts/eval_recognition.py` implementation is used for validation and
test; only `--split val` versus `--split test` changes.
