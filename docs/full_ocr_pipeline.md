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

## Output

The JSON output contains:

- `image_path`: source page path
- `items`: word-level OCR results
- `columns`: column-level grouped results
- `page_text`: one text line per page column
- `flat_text`: all recognized words in reading order

Each word-level item contains:

- `index`
- `box`
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