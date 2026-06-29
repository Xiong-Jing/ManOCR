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

Each item should include polygon points and may include text:

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


