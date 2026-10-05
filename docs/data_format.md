# Data Format

## Detection

Manifest:

```text
image_path<TAB>annotation_json
```

The remote formal dataset configuration expects at least 500 raw page images and 500 corresponding JSON annotations under the paths defined in `configs/paths/remote_server.yaml`.

Annotation JSON:

```json
{
  "image_path": "...",
  "image_width": 768,
  "image_height": 1056,
  "polygons": [
    {
      "label": "text",
      "points": [[10, 20], [30, 20], [30, 100], [10, 100]],
      "bbox": [10, 20, 30, 100]
    }
  ]
}
```

## Recognition

Manifest:

```text
word_image_path<TAB>label
```

The CTC blank index is fixed to `0`; charset files should contain non-blank characters only.

## Full OCR

Manifest:

```text
page_image_path<TAB>page_annotation_json
```

Full OCR annotation may use one of these top-level keys:

- `items`
- `polygons`
- `shapes`

For inference-only output, text is optional. For formal E2E validation/test,
every ground-truth word must include a real Romanized Manchu transcription;
generic detection labels such as `text`, `word`, or `manchu` are rejected before
either GPU model is loaded:

```json
{
  "items": [
    {
      "points": [[10, 20], [30, 20], [30, 100], [10, 100]],
      "text": "transcription"
    }
  ]
}
```

The page manifest reuses the detection page split. Enrich the JSON annotations
in place (or point the same split manifest at enriched copies); do not construct
recognition crops from the ground-truth polygons. Ground-truth polygons are used
only after inference for IoU matching and scoring.

The project uses enriched copies so the detection annotations remain untouched:

```text
processed/e2e/transcriptions.csv
processed/e2e/annotations/{val,test}/*.json
processed/e2e/{val,test}.txt
```

Create the CSV and optional review crops with
`tools/prepare_e2e_transcriptions.py export --save-review-crops`. Fill every
`transcription`, change `status` from `todo` to `done`, then run `apply` and
`audit`. Review crops are GT-based annotation aids only; they are never consumed
by the E2E recognizer.


