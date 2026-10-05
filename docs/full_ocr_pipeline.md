# Manchu Full-Page OCR Pipeline

## Input

Full-page OCR accepts one page image or a directory of page images.

Required model files:

- Detection config: `configs/detection/dbnetpp_vsaa_asym_shrink.yaml`
- Detection checkpoint: `outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth`
- Recognition config: `configs/recognition/svtr_official_dab_lortho.yaml`
- Recognition checkpoint: `outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth`

## Processing Channel

1. Page image is loaded and converted to RGB.
2. Detection transform resizes and pads the page to the detector input size.
3. The detector predicts text-region maps.
4. Post-processing extracts word boxes.
5. Predicted boxes are restored from resized coordinates to original page coordinates.
6. Boxes are sorted in Manchu page reading order: columns left-to-right, words top-to-bottom inside each column.
7. Each box is cropped from the original page image with dynamic padding.
8. The recognizer predicts the text for each word crop.
9. Words are grouped into coarse columns.
10. Output JSON and optional page text files are written.

The pipeline result explicitly records `uses_ground_truth_crops: false` and each
word record has `crop_source: detector_prediction` so E2E runs can audit crop
provenance.

## Output

The JSON output contains:

- `image_path`: source page path
- `items`: word-level OCR results
- `columns`: column-level grouped results
- `page_text`: one text line per page column
- `flat_text`: all recognized words in reading order

Each word-level item contains:

- `index`
- `detector_index` (original detector order, retained for matching audit)
- `box`
- `crop_source`
- `det_score`
- `raw_text`
- `text`
- `rec_confidence`

## Inference Command

```bash
export PYTHONPATH=/root/code/OCR_Manchu/src:$PYTHONPATH
cd /root/code/OCR_Manchu

python scripts/infer_full_ocr.py \
  --det-config configs/detection/dbnetpp_vsaa_asym_shrink.yaml \
  --det-checkpoint outputs/checkpoints/detection/dbnetpp_vsaa_asym_shrink/best.pth \
  --rec-config configs/recognition/svtr_official_dab_lortho.yaml \
  --rec-checkpoint outputs/checkpoints/recognition/svtr_official_dab_lortho/best.pth \
  --input /root/code/Manchu_Detection_Data/raw/images \
  --output outputs/predictions/full_ocr/full_page_predictions.json \
  --text-output-dir outputs/predictions/full_ocr/page_texts \
  --device cuda
```

## Quantitative E2E Evaluation

The formal 2 x 2 experiment is configured in
`configs/experiments/e2e_ocr.yaml`:

- DBNet++ + SVTR
- Ours detector + Ours recognizer
- DBNet++ + Ours recognizer
- Ours detector + SVTR

Both validation and test retain the existing page split through the generated
`e2e_data` manifests in `configs/paths/remote_server.yaml`. The detector runs on
the whole page and the recognizer receives only detector-predicted crops.
Matching uses the same greedy bbox-IoU rule as detection evaluation. Every
pipeline uses original page images and the fixed strict threshold IoU 0.75.

Run a strict server preflight and then all eight evaluations:

```bash
E2E_DRY_RUN=1 bash scripts/run_e2e_ocr_experiments.sh
bash scripts/run_e2e_ocr_experiments.sh
```

E2E WA is the number of IoU-matched words with exactly correct transcription
divided by all ground-truth words. E2E CER is the standard `(S+D+I)/N` value.
Missed words contribute full deletions and false-positive-box text contributes
full insertions. Compatibility aliases such as `e2e_exact_word_accuracy` and
`e2e_strict_cer` equal the primary strict metrics. Page text uses a single
space between words and a newline between columns; page-level CER remains strict.

Each run writes `eval_val.json` or `eval_test.json` under
`outputs/metrics/e2e_ocr/<pipeline-id>/`. The runner also writes
`e2e_comparison_latest.csv` and `e2e_comparison_latest.json` with the four
pipelines, split, both checkpoint paths, manifest, E2E WA/CER, page score/CER,
missed words, false positives, detection F-measure, and runtime.

Formal E2E scoring requires every page word annotation to contain `text` or
`transcription`. Existing detection-only JSON whose label is only `text` cannot
produce recognition ground truth. Prepare separate, enriched annotations while
preserving the original page split:

```bash
python tools/prepare_e2e_transcriptions.py export --save-review-crops
# Review the CSV and every GT crop; fill transcription and set status=done.
python tools/prepare_e2e_transcriptions.py apply
python tools/prepare_e2e_transcriptions.py audit
```

By default this prepares validation and test only. Add
`--splits train val test` if a train-split E2E audit is also required. Source
detection JSON is never overwritten. The generated GT crops are used solely for
human annotation/review and never become recognizer inputs during E2E scoring.
