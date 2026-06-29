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

If the page annotation contains `text` or `transcription`, `scripts/eval_full_ocr.py` reports end-to-end recognition metrics. Otherwise, it reports detection matching metrics only.

