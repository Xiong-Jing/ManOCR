import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, List, Tuple


def levenshtein_ops(a: str, b: str) -> Tuple[int, List[Tuple[str, str, str]]]:
    """
    Return edit distance and coarse edit operations from prediction a to label b.
    Operations are tuples: (op, pred_char, label_char).
    """
    n = len(a)
    m = len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                ops.append(("sub", a[i - 1], b[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1

    ops.reverse()
    return dp[n][m], ops


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def percent(num: int, den: int) -> float:
    return 100.0 * num / max(den, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    records = load_records(Path(args.predictions))

    total = len(records)
    empty_pred = 0
    exact = 0
    one_edit = 0
    len_diff_counter = Counter()
    label_len_stats = defaultdict(lambda: {"total": 0, "correct": 0, "edit": 0})
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()
    examples = []

    for record in records:
        pred = str(record.get("prediction", ""))
        label = str(record.get("label", ""))

        if pred == "":
            empty_pred += 1
        if pred == label:
            exact += 1

        dist, ops = levenshtein_ops(pred, label)
        if dist == 1:
            one_edit += 1

        len_diff_counter[len(pred) - len(label)] += 1

        bucket = len(label)
        label_len_stats[bucket]["total"] += 1
        label_len_stats[bucket]["correct"] += int(pred == label)
        label_len_stats[bucket]["edit"] += dist

        for op, pred_char, label_char in ops:
            if op == "sub":
                substitutions[(pred_char, label_char)] += 1
            elif op == "ins":
                insertions[label_char] += 1
            elif op == "del":
                deletions[pred_char] += 1

        if pred != label and len(examples) < 50:
            examples.append(
                {
                    "prediction": pred,
                    "label": label,
                    "edit_distance": dist,
                    "image_path": record.get("image_path", ""),
                }
            )

    lines = []
    lines.append("# Recognition Error Analysis")
    lines.append("")
    lines.append(f"- Predictions: `{args.predictions}`")
    lines.append(f"- Samples: {total}")
    lines.append(f"- Exact correct: {exact} ({percent(exact, total):.2f}%)")
    lines.append(f"- Empty predictions: {empty_pred} ({percent(empty_pred, total):.2f}%)")
    lines.append(f"- One-edit wrong/all: {one_edit} ({percent(one_edit, total):.2f}%)")
    lines.append("")

    lines.append("## Accuracy By Label Length")
    lines.append("")
    lines.append("| Label Len | Samples | WA | CER |")
    lines.append("|---:|---:|---:|---:|")
    for label_len in sorted(label_len_stats):
        stat = label_len_stats[label_len]
        wa = stat["correct"] / max(stat["total"], 1)
        cer = stat["edit"] / max(stat["total"] * label_len, 1)
        lines.append(f"| {label_len} | {stat['total']} | {wa * 100:.2f} | {cer * 100:.2f} |")

    lines.append("")
    lines.append("## Length Difference")
    lines.append("")
    lines.append("| pred_len - label_len | Count |")
    lines.append("|---:|---:|")
    for diff, count in len_diff_counter.most_common(args.top_k):
        lines.append(f"| {diff} | {count} |")

    lines.append("")
    lines.append("## Top Substitutions")
    lines.append("")
    lines.append("| Pred | Label | Count |")
    lines.append("|---|---|---:|")
    for (pred_char, label_char), count in substitutions.most_common(args.top_k):
        lines.append(f"| `{pred_char}` | `{label_char}` | {count} |")

    lines.append("")
    lines.append("## Top Insertions")
    lines.append("")
    lines.append("| Missing Label Char | Count |")
    lines.append("|---|---:|")
    for label_char, count in insertions.most_common(args.top_k):
        lines.append(f"| `{label_char}` | {count} |")

    lines.append("")
    lines.append("## Top Deletions")
    lines.append("")
    lines.append("| Extra Pred Char | Count |")
    lines.append("|---|---:|")
    for pred_char, count in deletions.most_common(args.top_k):
        lines.append(f"| `{pred_char}` | {count} |")

    lines.append("")
    lines.append("## First Wrong Examples")
    lines.append("")
    lines.append("| Prediction | Label | EditDist | Image |")
    lines.append("|---|---|---:|---|")
    for item in examples:
        lines.append(
            f"| `{item['prediction']}` | `{item['label']}` "
            f"| {item['edit_distance']} | `{item['image_path']}` |"
        )

    text = "\n".join(lines)

    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved to: {out_path}")

    print(text)


if __name__ == "__main__":
    main()
