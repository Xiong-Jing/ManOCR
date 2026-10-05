import argparse
import json
import math
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont


DEFAULT_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_font(font_path: str | None, font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []

    if font_path:
        candidates.append(font_path)

    candidates.extend(DEFAULT_FONT_CANDIDATES)

    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), font_size)

    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text_by_width(
    text: str,
    draw: ImageDraw.ImageDraw,
    font,
    max_width: int,
) -> List[str]:
    words = text.split()

    if not words:
        return [""]

    lines = []
    current = words[0]

    for word in words[1:]:
        candidate = f"{current} {word}"
        width, _ = text_size(draw, candidate, font)

        if width <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word

    lines.append(current)
    return lines


def normalize_page_records(data) -> List[Dict]:
    if isinstance(data, dict) and "items" in data:
        return [data]

    if isinstance(data, list):
        return data

    raise ValueError("Full OCR JSON should be a page dict or a list of page dicts.")


def get_page_title(page: Dict, fallback_idx: int) -> str:
    image_path = page.get("image_path", "")

    if image_path:
        return Path(image_path).stem

    return f"page_{fallback_idx:04d}"


def find_text_file_for_page(
    page: Dict,
    text_dir: Path | None,
    fallback_idx: int,
) -> Path | None:
    if text_dir is None or not text_dir.exists():
        return None

    image_path = page.get("image_path", "")

    candidates = []

    if image_path:
        candidates.append(text_dir / f"{Path(image_path).stem}.txt")

    candidates.append(text_dir / f"page_{fallback_idx:04d}.txt")
    candidates.append(text_dir / f"{fallback_idx:04d}.txt")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    txt_files = sorted(text_dir.glob("*.txt"))

    if 0 <= fallback_idx - 1 < len(txt_files):
        return txt_files[fallback_idx - 1]

    return None


def build_columns_from_text_file(text_path: Path) -> List[Dict]:
    lines = text_path.read_text(encoding="utf-8").splitlines()
    columns = []

    for idx, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        columns.append(
            {
                "column_index": idx,
                "text": line,
            }
        )

    return columns


def build_columns_from_items(page: Dict, text_path: Path | None = None) -> List[Dict]:
    if text_path is not None:
        columns = build_columns_from_text_file(text_path)

        if columns:
            return columns

    columns = page.get("columns")

    if columns:
        return columns

    text = page.get("page_text") or page.get("flat_text") or ""

    if text:
        return [
            {
                "column_index": idx,
                "text": line.strip(),
            }
            for idx, line in enumerate(text.splitlines())
            if line.strip()
        ]

    items = page.get("items", [])
    return [
        {
            "column_index": 0,
            "text": " ".join(str(item.get("text", "")).strip() for item in items if str(item.get("text", "")).strip()),
        }
    ]


def render_sequence_image(
    page: Dict,
    output_path: Path,
    text_path: Path | None = None,
    font_path: str | None = None,
    font_size: int = 28,
    width: int = 1400,
    row_padding: int = 18,
    margin: int = 36,
    title: str | None = None,
) -> None:
    font = load_font(font_path, font_size)
    title_font = load_font(font_path, int(round(font_size * 1.15)))
    index_font = load_font(font_path, max(14, int(round(font_size * 0.72))))

    columns = build_columns_from_items(page, text_path=text_path)
    scratch = Image.new("RGB", (width, 100), "white")
    draw = ImageDraw.Draw(scratch)

    index_width = 100
    text_max_width = width - margin * 2 - index_width
    line_gap = max(8, font_size // 3)

    rendered_rows = []
    total_height = margin

    if title:
        _, title_h = text_size(draw, title, title_font)
        total_height += title_h + margin

    for col in columns:
        text = str(col.get("text", "")).strip()
        wrapped = wrap_text_by_width(text, draw, font, text_max_width)
        line_heights = [text_size(draw, line, font)[1] for line in wrapped]
        row_h = sum(line_heights) + line_gap * max(0, len(wrapped) - 1) + row_padding * 2
        row_h = max(row_h, font_size + row_padding * 2)
        rendered_rows.append((col, wrapped, row_h))
        total_height += row_h

    total_height += margin
    image = Image.new("RGB", (width, total_height), color=(248, 246, 238))
    draw = ImageDraw.Draw(image)

    y = margin

    if title:
        draw.text((margin, y), title, fill=(32, 34, 35), font=title_font)
        _, title_h = text_size(draw, title, title_font)
        y += title_h + margin

    for row_idx, (col, wrapped, row_h) in enumerate(rendered_rows):
        x0 = margin
        y0 = y
        x1 = width - margin
        y1 = y + row_h
        fill = (255, 255, 252) if row_idx % 2 == 0 else (243, 240, 231)
        outline = (210, 202, 186)

        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline=outline, width=1)

        col_idx = int(col.get("column_index", row_idx))
        index_text = f"col {col_idx + 1}"
        draw.text((x0 + 18, y0 + row_padding), index_text, fill=(122, 94, 59), font=index_font)

        text_x = x0 + index_width
        text_y = y0 + row_padding

        for line in wrapped:
            draw.text((text_x, text_y), line, fill=(34, 37, 39), font=font)
            _, line_h = text_size(draw, line, font)
            text_y += line_h + line_gap

        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def resize_keep_height(image: Image.Image, target_height: int) -> Image.Image:
    scale = target_height / float(image.height)
    width = max(1, int(round(image.width * scale)))
    return image.resize((width, target_height), Image.Resampling.LANCZOS)


def render_side_by_side(
    page: Dict,
    sequence_image_path: Path,
    annotated_image_path: Path,
    output_path: Path,
    gap: int = 24,
) -> None:
    image_path = page.get("image_path")

    if not image_path or not Path(image_path).exists() or not annotated_image_path.exists():
        return

    source = Image.open(image_path).convert("RGB")
    annotated = Image.open(annotated_image_path).convert("RGB")
    sequence = Image.open(sequence_image_path).convert("RGB")

    source = resize_keep_height(source, sequence.height)
    annotated = resize_keep_height(annotated, sequence.height)

    width = source.width + annotated.width + sequence.width + gap * 2
    height = max(source.height, annotated.height, sequence.height)
    canvas = Image.new("RGB", (width, height), color=(238, 235, 226))
    canvas.paste(source, (0, 0))
    canvas.paste(annotated, (source.width + gap, 0))
    canvas.paste(sequence, (source.width + annotated.width + gap * 2, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_item_to_column(page: Dict) -> Dict[int, int]:
    item_to_column = {}

    for column in page.get("columns", []):
        col_idx = int(column.get("column_index", len(item_to_column)))

        for item in column.get("items", []):
            if "index" in item:
                item_to_column[int(item["index"])] = col_idx

    return item_to_column


def palette(index: int) -> tuple[int, int, int]:
    colors = [
        (215, 48, 39),
        (26, 152, 80),
        (49, 130, 189),
        (245, 130, 32),
        (145, 61, 181),
        (0, 150, 170),
        (180, 120, 30),
        (90, 90, 90),
    ]
    return colors[index % len(colors)]


def render_annotated_page(
    page: Dict,
    output_path: Path,
    font_path: str | None = None,
) -> None:
    image_path = page.get("image_path")

    if not image_path or not Path(image_path).exists():
        return

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = (220, 20, 20)

    for item in page.get("items", []):
        points = item.get("box", [])

        if len(points) < 3:
            continue

        pts = [(float(x), float(y)) for x, y in points]
        closed = pts + [pts[0]]

        for offset in range(3):
            shifted = [(x + offset, y + offset) for x, y in closed]
            draw.line(shifted, fill=color, width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/visualizations/full_ocr_sequence")
    parser.add_argument(
        "--text-dir",
        type=str,
        default=None,
        help="Directory containing page TXT outputs. Sequence images are generated from these TXT files when available.",
    )
    parser.add_argument("--font-path", type=str, default=None)
    parser.add_argument("--font-size", type=int, default=28)
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--side-by-side", action="store_true")
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    pages = normalize_page_records(load_json(Path(args.input_json)))

    if args.max_pages is not None:
        pages = pages[: args.max_pages]

    output_dir = Path(args.output_dir)
    text_dir = Path(args.text_dir) if args.text_dir else output_dir / "page_texts"
    sequence_dir = output_dir / "sequence"
    annotated_dir = output_dir / "annotated"
    combined_dir = output_dir / "combined"

    for idx, page in enumerate(pages, start=1):
        text_path = find_text_file_for_page(page, text_dir=text_dir, fallback_idx=idx)
        title = text_path.stem if text_path is not None else get_page_title(page, idx)
        safe_name = f"{idx:04d}_{title}"
        safe_name = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in safe_name)

        seq_path = sequence_dir / f"{safe_name}.png"
        annotated_path = annotated_dir / f"{safe_name}.png"

        render_sequence_image(
            page=page,
            output_path=seq_path,
            text_path=text_path,
            font_path=args.font_path,
            font_size=args.font_size,
            width=args.width,
            title=title,
        )

        render_annotated_page(
            page=page,
            output_path=annotated_path,
            font_path=args.font_path,
        )

        if args.side_by_side:
            render_side_by_side(
                page=page,
                sequence_image_path=seq_path,
                annotated_image_path=annotated_path,
                output_path=combined_dir / f"{safe_name}.png",
            )

        print(f"[OK] {seq_path}")

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
