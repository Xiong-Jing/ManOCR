# Manchu OCR

PyTorch implementation for **Morphology-Aware Detection and Orthography-Guided Recognition for Low-Resource Manchu Historical OCR**.

The framework localizes vertically written Manchu words and transcribes word images into Romanized text. The repository provides model implementations, dataset preparation, controlled ablations, component evaluation, and page-level OCR experiments.

## Methods

**Detection: DBNet++ + VSAA + AS.** A ResNet-34 backbone and DBFPN with adaptive-scale fusion produce multi-scale features. Vertical Strip-Aware Attention combines long vertical and short horizontal convolution responses into a spatial attention map. Asymmetric Shrinkage generates training targets with direction-dependent inward offsets: stem-side edges use the standard DB distance, and terminal edges use one third of that distance. AS operates during target generation and does not alter inference post-processing.

**Recognition: SVTR + DAB + OTP.** A diacritic-aware detail branch complements the visual sequence encoder through cross-attention. The recognition objective is `L_total = L_CTC + λ L_OTP`. The main model uses `λ = 0.1`. OTP uses a fixed transition prior constructed exclusively from training transcriptions. Add-one smoothing, explicit local rule weights, and a numerical floor preserve positive transition probabilities. The default rule coefficient is `ρ = 0.05`; it controls prior construction independently of λ. SVTR ablations and sensitivity experiments use greedy CTC decoding without lexicon correction.

| Component | Implementation |
|---|---|
| Detector | [DBNet++](src/manchu_ocr/models/detection/dbnetpp.py) |
| Feature fusion | [DBFPN](src/manchu_ocr/models/detection/necks/db_fpn.py) |
| Directional attention | [PaperVSAA](src/manchu_ocr/models/detection/modules/vsaa.py) |
| Asymmetric targets | [Edge-offset shrinkage](src/manchu_ocr/data/label_generators/edge_shrink.py) |
| Recognizer | [SVTR](src/manchu_ocr/models/recognition/svtr_official.py) |
| Detail branch | [DAB](src/manchu_ocr/models/recognition/branches/diacritic_aware_branch.py) |
| Transition prior | [Orthographic prior](src/manchu_ocr/data/orthography.py) |
| OTP objective | [Transition loss](src/manchu_ocr/losses/orthographic_transition_loss.py) |

## Installation

Use Python 3.10 or later and install the package from the repository root:

```bash
pip install -e .
pip install -r requirements.txt
```

GPU training requires a CUDA-enabled PyTorch installation. Model and dataset tests can run on CPU. Shell experiment runners target Bash; Python entry points can also be invoked from PowerShell using `python` or `py`. Hosted OCR evaluation has separate client dependencies:

```bash
pip install -r requirements-api.txt
```

## Dataset and partitions

The detection dataset consists of **3,853 printed Manchu pages** with word-level annotations. Its partition contract is defined in [detection_3853.yaml](configs/datasets/detection_3853.yaml).

| Split | Source pages |
|---|---:|
| Training | 3,083 |
| Validation | 385 |
| Test | 385 |
| Total | 3,853 |

Pages are shuffled deterministically with seed 42. Recognition crops inherit the partition of their source page. A page shared by detection and recognition has the same partition in both tasks. End-to-end validation and test manifests identify independent page subsets within their respective partitions.

```text
Manchu_Detection_Data/
  raw/{images,annotations_json}/
  processed/
    annotations_clean/
    train.txt, val.txt, test.txt
    e2e/{annotations,val.txt,test.txt}

Manchu_Recognition_Data/
  raw/{word_images,labels.xlsx}
  processed/
    labels.csv, charset.txt
    transition_matrix.npy, transition_matrix.json
    train.txt, val.txt, test.txt
```

Manifest formats:

```text
Detection / page OCR: image_path<TAB>annotation_json_path
Word recognition:     image_path<TAB>Romanized_transcription
```

Page OCR annotations contain word polygons and real transcriptions. A generic detection class such as `manchu` or `text` is not a transcription.

### Paths

[remote_server.yaml](configs/paths/remote_server.yaml) is the default paths configuration. These environment variables override machine-specific locations:

```bash
export OCR_MANCHU_PROJECT_ROOT=/path/to/Manchu-OCR
export OCR_MANCHU_DET_ROOT=/path/to/Manchu_Detection_Data
export OCR_MANCHU_REC_ROOT=/path/to/Manchu_Recognition_Data
export OCR_MANCHU_OUTPUT_ROOT=/path/to/Manchu-OCR/outputs
```

The equivalent PowerShell syntax is `$env:OCR_MANCHU_DET_ROOT = 'D:/data/Manchu_Detection_Data'`. A [Windows paths configuration](configs/paths/local_windows.yaml) is available for direct path editing. Model configurations select their paths file through `experiment.paths_config`.

### Preparation

Source-page provenance is supplied as a CSV with `image_path,source_page_id` columns. Include page images and their recognition crops, using one shared page ID for all crops from the same original page.

```csv
image_path,source_page_id
/data/pages/page001.png,page001
/data/words/page001_word01.png,page001
/data/words/page001_word02.png,page001
```

```bash
python tools/prepare_detection_data.py --config configs/paths/remote_server.yaml
python tools/prepare_recognition_data.py --config configs/paths/remote_server.yaml --page-map /data/source_pages.csv
python tools/build_transition_matrix.py --config configs/paths/remote_server.yaml --rho 0.05
python tools/check_dataset.py --config configs/paths/remote_server.yaml
python tools/audit_page_splits.py --config configs/paths/remote_server.yaml --page-map /data/source_pages.csv
```

Detection preparation checks valid page count and exact split sizes before replacing prepared annotations or manifests. Recognition preparation checks crop provenance and preserves detection split assignments. Transition artifacts record training-manifest, charset, prior parameters, and content hashes.

Detection training augmentation samples brightness, contrast, anisotropic scaling, and smooth displacements on an 8×8 control grid. Pixels and polygon vertices share the same geometric transform. Validation and test images use deterministic resizing and normalization.

## Training and component evaluation

| Task | Main configuration | Epochs | Batch size | Optimizer | Learning rate |
|---|---|---:|---:|---|---:|
| Detection | [DBNet++ + VSAA + AS](configs/detection/dbnetpp_vsaa_asym_shrink.yaml) | 100 | 16 | AdamW | 1e-4 |
| Recognition | [SVTR + DAB + OTP](configs/recognition/svtr_official_dab_lortho.yaml) | 200 | 128 | AdamW | 6e-5 |

```bash
python scripts/train_detection.py --config configs/detection/dbnetpp_vsaa_asym_shrink.yaml
python scripts/train_recognition.py --config configs/recognition/svtr_official_dab_lortho.yaml
```

Detection validation runs every epoch. Recognition validation runs every 50 epochs and at the final epoch. Detection checkpoints are selected by validation F-measure; recognition checkpoints are selected by validation WA, with lower CER breaking ties.

```bash
python scripts/eval_detection.py \
  --config configs/detection/dbnetpp_vsaa_asym_shrink.yaml \
  --checkpoint outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth --split test

python scripts/eval_recognition.py \
  --config configs/recognition/svtr_official_dab_lortho.yaml \
  --checkpoint outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth --split test --save-predictions
```

Resume training with `--resume /path/to/last.pth`. Checkpoints store optimizer, scheduler, scaler, model configuration, and selection state. Protocol and transition-prior checks prevent incompatible checkpoints from entering an experiment.

### Comparison and ablation suites

| Experiment | Models / settings | Runner |
|---|---|---|
| Detection ablation | DBNet++, +VSAA, +AS, +VSAA+AS | `bash scripts/run_det_ablation.sh` |
| Directional counterparts | None, Strip Pooling, Coordinate Attention, horizontal strip, vertical strip, VSAA | `bash scripts/run_det_vsaa_counterparts.sh` |
| Detection comparison | EAST, CRAFT, DBNet, PP-OCRv5 Det, Hi-SAM, proposed detector | `bash scripts/run_det_comparison.sh` |
| Recognition ablation | SVTR, +DAB, +OTP, +DAB+OTP | `bash scripts/run_rec_ablation.sh` |
| Recognition comparison | CRNN, PARSeq, ABINet, DCM, SVTRv2, SVTRv2+NRTR, proposed recognizer | `bash scripts/run_rec_comparison.sh` |
| OTP loss coefficient | λ = 0, 0.025, 0.05, 0.1, 0.2, 0.5 | `bash scripts/run_rec_otp_lambda_sensitivity.sh` |

Comparison models are project implementations integrated with the same data and evaluation framework. Third-party pretrained weights are not bundled. Experimental configurations specify each implementation and decoder.

### Rule-weight sensitivity

[The ρ experiment](configs/experiments/rec_otp_rho_sensitivity.yaml) evaluates `ρ ∈ {0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1}` at fixed `λ=0.1`. Each setting has a separate transition prior and checkpoint directory.

```bash
python scripts/run_rec_otp_rho_sensitivity.py --stage all --dry-run
python scripts/run_rec_otp_rho_sensitivity.py --stage prepare
python scripts/run_rec_otp_rho_sensitivity.py --stage train
python scripts/run_rec_otp_rho_sensitivity.py --stage evaluate --splits val
python scripts/run_rec_otp_rho_sensitivity.py --stage evaluate --splits test
python scripts/run_rec_otp_rho_sensitivity.py --stage summarize --splits val test
```

`ρ=1` retains the smoothed corpus prior and OTP; `λ=0` removes OTP. Model selection uses validation data. The test split is reserved for evaluation with the selected settings.

## Metrics

- **Detection:** precision, recall, and F-measure with one-to-one matching at IoU 0.75.
- **Word accuracy (WA):** exact transcription matches divided by reference words.
- **Character accuracy (CA):** `(N−S−D)/N`.
- **Character error rate (CER):** `(S+D+I)/N`, using corpus-level minimum-edit alignment.
- **Page transcription score (PTS):** mean `100 × max(0, 1−(Sw+Dw+Iw)/Nw)` over pages, with word-level edit counts and reference word count.

For two-stage end-to-end OCR, a word is correct only when localization and transcription both match. Missed words contribute deletions; unmatched predicted regions contribute insertions. Scores are stored as ratios except PTS, which uses a 0–100 scale.

## Page OCR

### Two-stage pipeline

```bash
bash scripts/run_full_pipeline.sh
python scripts/run_e2e_ocr_experiments.py --splits validation --dry-run
python scripts/run_e2e_ocr_experiments.py --splits validation
python scripts/run_e2e_ocr_experiments.py --splits test
```

[e2e_ocr.yaml](configs/experiments/e2e_ocr.yaml) defines four detector–recognizer combinations. Reading order is top to bottom within a column and left to right across columns. Page transcription review tools:

```bash
python tools/prepare_e2e_transcriptions.py export --save-review-crops
python tools/prepare_e2e_transcriptions.py apply
python tools/prepare_e2e_transcriptions.py audit
```

### Ground-truth crop comparison

```bash
python scripts/eval_crop_oracle.py \
  --manifest /data/e2e/test_manual.txt \
  --det-config configs/detection/dbnetpp_vsaa_asym_shrink.yaml \
  --det-checkpoint outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth \
  --rec-config configs/recognition/svtr_official_dab_lortho.yaml \
  --rec-checkpoint outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth \
  --split test --device cuda
```

The two conditions share pages, recognition weights, crop padding, preprocessing, and decoding. Output includes per-word records and differences between predicted-crop and manually annotated GT-crop scores.

### Zero-shot whole-page OCR

[lmm_page_zero_shot.yaml](configs/experiments/lmm_page_zero_shot.yaml) defines hosted Qwen2.5-VL-7B-Instruct, GOT-OCR2.0, and PaddleOCR-VL-1.5 evaluation on `e2e_data` page manifests.

```bash
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test --prepare-only
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test --dry-run
python scripts/run_lmm_zero_shot.py --config configs/experiments/lmm_page_zero_shot.yaml --splits test
```

Provider credentials use `DASHSCOPE_API_KEY`, `HF_TOKEN` where required, and `PADDLEOCR_ACCESS_TOKEN`. Endpoint overrides are specified in the configuration. Requests contain a complete original page without reference text or in-context examples. Successful responses are cached by sample and protocol identity. Transport failures remain retryable; empty or refused model outputs are scored as empty transcriptions.

Text-only API responses support token-aligned WA, character CER, and PTS. They do not provide spatial Detection F or IoU-matched WA without actual word polygons. Metric definitions and input provenance accompany result files.

## Error analysis and computational cost

```bash
python tools/analyze_e2e_errors.py export --records /path/e2e_records_test.json --review outputs/review/errors.csv
python tools/analyze_e2e_errors.py summarize --records /path/e2e_records_test.json --review outputs/review/errors.csv

python tools/measure_model_cost.py --config configs/detection/dbnetpp_vsaa_asym_shrink.yaml --output outputs/metrics/cost/detector.json
python tools/measure_model_cost.py --config configs/recognition/svtr_official_dab_lortho.yaml --charset /data/processed/charset.txt --output outputs/metrics/cost/recognizer.json
```

Error attribution uses six manually reviewed categories: head–tail truncation, adjacent-column merging, missed words, false-positive word regions, dot/circle confusion, and rare orthographic variants. Shares use categorized error instances as their denominator.

Parameter counts exclude buffers, optimizer state, and duplicate EMA copies. Optional `--benchmark --device cuda` measures model-forward latency with warm-up and synchronized timing. Training histories record per-epoch training and validation durations; standalone evaluations record their elapsed time and runtime environment.

## Outputs and tests

```text
outputs/
  checkpoints/{detection,recognition}/<experiment>/{best,last}.pth
  metrics/{detection,recognition}/<experiment>/
  metrics/{e2e_ocr,crop_oracle,lmm_zero_shot}/
  experiments/otp_rho/
  predictions/
  visualizations/
```

Result records include configuration, checkpoint and manifest references, metric definitions, runtime details, and applicable content hashes. Summarization and plotting tools are under `tools/`.

```bash
python -m pytest -q
```

## License

See [LICENSE](LICENSE).
