"""Export residual events for human attribution and summarize reviewed instances."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from manchu_ocr.data.orthography import sha256_file

CATEGORIES = ("head_tail_truncation", "adjacent_column_merge", "missed_word",
              "false_positive_word_region", "dot_circle_confusion", "rare_orthographic_variant")


def residual_events(records):
    rows = []
    for page in records:
        image = page["image_path"]
        groups = (
            ("recognition_mismatch", [m for m in page["matches"] if not m["text_exact"]]),
            ("missed_word", page["missed_ground_truth_items"]),
            ("false_positive_word_region", page["false_positive_predictions"]),
        )
        for kind, items in groups:
            for index, item in enumerate(items):
                event_id = hashlib.sha256(f"{image}\0{kind}\0{index}".encode()).hexdigest()[:24]
                rows.append({"event_id": event_id, "instance_id": event_id, "image_path": image,
                             "event_type": kind,
                             "evidence_json": json.dumps(item, ensure_ascii=False, sort_keys=True),
                             "category": "", "reviewed": "", "reviewer": "", "notes": ""})
    if len({r["event_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate pages/events in E2E records")
    return rows


def summarize_review(rows, expected):
    by_id = {r["event_id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {r["event_id"] for r in expected}:
        raise ValueError("Review rows must cover every residual event exactly once")
    instances = {}
    for original in expected:
        row = by_id[original["event_id"]]
        if any(row[k] != original[k] for k in ("image_path", "event_type", "evidence_json")):
            raise ValueError(f"Evidence modified: {row['event_id']}")
        if row["reviewed"] != "yes" or not row["reviewer"].strip() or row["category"] not in CATEGORIES:
            raise ValueError(f"Unreviewed/invalid attribution: {row['event_id']}")
        instance_id = row["instance_id"].strip()
        if not instance_id:
            raise ValueError("instance_id cannot be empty")
        identity = (row["image_path"], row["category"])
        if instance_id in instances and instances[instance_id] != identity:
            raise ValueError("Grouped instance must belong to one page and one predominant cause")
        instances[instance_id] = identity
    counts = Counter(category for _, category in instances.values())
    n = len(instances)
    return {"num_residual_events": len(rows), "num_reviewed_instances": n,
            "denominator": "manually categorized error instances, not all words/pages",
            "categories": [{"category": c, "count": counts[c],
                            "share_percent": 100 * counts[c] / n if n else 0.0} for c in CATEGORIES]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export", "summarize"])
    parser.add_argument("--records", required=True, help="eval_full_ocr --save-records output")
    parser.add_argument("--review", required=True, help="Human review CSV")
    parser.add_argument("--output", default="outputs/metrics/e2e_error_analysis.json")
    args = parser.parse_args()
    expected = residual_events(json.loads(Path(args.records).read_text(encoding="utf-8")))
    review = Path(args.review)
    if args.action == "export":
        if review.exists():
            raise FileExistsError(f"Review already exists: {review}")
        review.parent.mkdir(parents=True, exist_ok=True)
        with review.open("w", encoding="utf-8-sig", newline="") as handle:
            fields = list(expected[0]) if expected else ["event_id", "instance_id", "image_path", "event_type", "evidence_json", "category", "reviewed", "reviewer", "notes"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(expected)
        review.with_suffix(".meta.json").write_text(json.dumps({"records_sha256": sha256_file(args.records)}, indent=2), encoding="utf-8")
    else:
        metadata = json.loads(review.with_suffix(".meta.json").read_text(encoding="utf-8"))
        if metadata["records_sha256"] != sha256_file(args.records):
            raise ValueError("Source E2E records changed since review export")
        with review.open(encoding="utf-8-sig", newline="") as handle:
            report = summarize_review(list(csv.DictReader(handle)), expected)
        report.update(metadata)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
