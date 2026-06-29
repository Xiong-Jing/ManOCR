import argparse
import json
from pathlib import Path

from manchu_ocr.metrics.recognition_metrics import compute_recognition_metrics


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Predictions file should contain a list: {path}")

    return data


def fmt_metrics(metrics: dict) -> str:
    return (
        f"WA={metrics['word_accuracy'] * 100:.2f}, "
        f"CA={metrics['character_accuracy'] * 100:.2f}, "
        f"CER={metrics['cer'] * 100:.2f}, "
        f"EditDist={metrics['edit_distance']:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    records = load_records(pred_path)

    preds = [str(item.get("prediction", "")) for item in records]
    labels = [str(item.get("label", "")) for item in records]

    reversed_labels = [label[::-1] for label in labels]
    reversed_preds = [pred[::-1] for pred in preds]

    original_metrics = compute_recognition_metrics(preds, labels)
    reversed_label_metrics = compute_recognition_metrics(preds, reversed_labels)
    reversed_pred_metrics = compute_recognition_metrics(reversed_preds, labels)

    reverse_exact = []
    for item, pred, label in zip(records, preds, labels):
        if pred == label[::-1] and pred != label:
            reverse_exact.append(
                {
                    "prediction": pred,
                    "label": label,
                    "reversed_label": label[::-1],
                    "image_path": item.get("image_path", ""),
                }
            )

    result = {
        "predictions": str(pred_path).replace("\\", "/"),
        "num_samples": len(records),
        "original": original_metrics,
        "compare_to_reversed_labels": reversed_label_metrics,
        "compare_reversed_predictions_to_labels": reversed_pred_metrics,
        "reverse_exact_count": len(reverse_exact),
        "reverse_exact_examples": reverse_exact[: args.max_examples],
    }

    lines = []
    lines.append("# Recognition Label Order Diagnosis")
    lines.append("")
    lines.append(f"- Predictions: `{result['predictions']}`")
    lines.append(f"- Samples: {len(records)}")
    lines.append("")
    lines.append("| Evaluation | WA | CA | CER | EditDist |")
    lines.append("|---|---:|---:|---:|---:|")

    for name, metrics in [
        ("Prediction vs Label", original_metrics),
        ("Prediction vs Reversed Label", reversed_label_metrics),
        ("Reversed Prediction vs Label", reversed_pred_metrics),
    ]:
        lines.append(
            f"| {name} "
            f"| {metrics['word_accuracy'] * 100:.2f} "
            f"| {metrics['character_accuracy'] * 100:.2f} "
            f"| {metrics['cer'] * 100:.2f} "
            f"| {metrics['edit_distance']:.4f} |"
        )

    lines.append("")
    lines.append(
        f"- Exact reverse matches: {len(reverse_exact)} / {len(records)} "
        f"({100.0 * len(reverse_exact) / max(len(records), 1):.2f}%)"
    )

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- If `Prediction vs Reversed Label` is much better than `Prediction vs Label`, "
        "the label order is likely reversed."
    )
    lines.append(
        "- If both reversed-order metrics are close to or worse than the original, "
        "the label order is probably not the main bottleneck."
    )

    if reverse_exact:
        lines.append("")
        lines.append("## Reverse Exact Examples")
        lines.append("")
        lines.append("| Prediction | Label | Reversed Label | Image |")
        lines.append("|---|---|---|---|")
        for item in reverse_exact[: args.max_examples]:
            lines.append(
                f"| `{item['prediction']}` | `{item['label']}` "
                f"| `{item['reversed_label']}` | `{item['image_path']}` |"
            )

    text = "\n".join(lines)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        json_path = out_path.with_suffix(".json")
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Saved markdown to: {out_path}")
        print(f"Saved json to: {json_path}")

    print(text)


if __name__ == "__main__":
    main()
