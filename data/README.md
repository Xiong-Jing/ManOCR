# Data Layout

This repository keeps only code/configuration. Large images, annotations, and processed manifests are stored outside the repository and referenced through:

```text
configs/paths/remote_server.yaml
```

## Detection Manifest

Format:

```text
image_path<TAB>clean_annotation_json
```

Clean annotation JSON:

```json
{
  "image_path": "...",
  "image_width": 1000,
  "image_height": 1500,
  "polygons": [
    {
      "label": "text",
      "points": [[0, 0], [10, 0], [10, 50], [0, 50]],
      "bbox": [0, 0, 10, 50]
    }
  ]
}
```

## Recognition Manifest

Format:

```text
word_image_path<TAB>transcription
```

## Full OCR Manifest

Format:

```text
page_image_path<TAB>page_annotation_json
```

Formal E2E validation/test requires every word polygon to contain a real `text`
or `transcription`. A generic detection label such as `text` is not treated as
ground-truth transcription. `scripts/eval_full_ocr.py` validates the complete
manifest first and stops before model loading when any transcription is missing.

Use `python tools/prepare_e2e_transcriptions.py export --save-review-crops`,
complete the generated CSV, then run the same tool with `apply` and `audit`.
This writes separate `e2e_data` annotations/manifests and preserves the original
detection JSON and page split.

