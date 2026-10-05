from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENTS = [
    (0.0, "svtr_dab_otp_lambda_0", "unified strict"),
    (0.025, "svtr_dab_otp_lambda_0p025", "unified strict"),
    (0.05, "svtr_dab_otp_lambda_0p05", "unified strict"),
    (0.1, "svtr_dab_otp_lambda_0p1", "unified strict"),
    (0.2, "svtr_dab_otp_lambda_0p2", "unified strict"),
    (0.5, "svtr_dab_otp_lambda_0p5", "unified strict"),
]


def load_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation result: {path}")
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    required = {
        "model_name",
        "split",
        "config_path",
        "checkpoint_path",
        "manifest_path",
        "metrics",
        "runtime",
        "runtime_seconds",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise KeyError(f"{path} is missing required result fields: {missing}")
    return result


def validate_metric_policy(
    result: dict[str, Any],
    *,
    experiment: str,
    lambda_value: float,
) -> None:
    metrics = result["metrics"]
    if metrics.get("word_accuracy") != metrics.get("exact_word_accuracy"):
        raise ValueError(f"{experiment} result does not use exact-match WA")
    if metrics.get("character_accuracy") != metrics.get("strict_character_accuracy"):
        raise ValueError(f"{experiment} result does not use strict CA")
    if metrics.get("cer") != metrics.get("strict_cer"):
        raise ValueError(f"{experiment} result does not use strict CER")


def metric_fields(prefix: str, result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    return {
        f"{prefix}_WA": float(metrics["word_accuracy"]),
        f"{prefix}_CA": float(metrics["character_accuracy"]),
        f"{prefix}_CER": float(metrics["cer"]),
        f"{prefix}_exact_WA": float(metrics["exact_word_accuracy"]),
        f"{prefix}_strict_CA": float(metrics["strict_character_accuracy"]),
        f"{prefix}_strict_CER": float(metrics["strict_cer"]),
        f"{prefix}_S": int(metrics["substitutions"]),
        f"{prefix}_D": int(metrics["deletions"]),
        f"{prefix}_I": int(metrics["insertions"]),
        f"{prefix}_N": int(metrics["reference_characters"]),
        f"{prefix}_runtime_seconds": float(result["runtime_seconds"]),
    }


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the controlled Our-recognizer OTP lambda sweep."
    )
    parser.add_argument("--root", default="outputs/metrics/recognition")
    parser.add_argument(
        "--output-prefix",
        default="recognition_otp_lambda_sensitivity",
    )
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    rows: list[dict[str, Any]] = []
    for lambda_value, experiment, metric_source in EXPERIMENTS:
        result_paths = {
            split: root / experiment / f"eval_{split}.json"
            for split in ("val", "test")
        }
        missing_paths = [path for path in result_paths.values() if not path.is_file()]
        if missing_paths and args.allow_missing:
            continue
        if missing_paths:
            raise FileNotFoundError(
                "Missing lambda-sensitivity result(s): "
                + ", ".join(str(path) for path in missing_paths)
            )

        validation = load_result(result_paths["val"])
        test = load_result(result_paths["test"])
        validate_metric_policy(
            validation,
            experiment=experiment,
            lambda_value=lambda_value,
        )
        validate_metric_policy(
            test,
            experiment=experiment,
            lambda_value=lambda_value,
        )
        if validation["checkpoint_path"] != test["checkpoint_path"]:
            raise ValueError(
                f"{experiment} validation/test used different checkpoints: "
                f"{validation['checkpoint_path']} vs {test['checkpoint_path']}"
            )

        row: dict[str, Any] = {
            "lambda": lambda_value,
            "experiment": experiment,
            "OTP_enabled": lambda_value > 0.0,
            "metric_logic": metric_source,
            "checkpoint_path": test["checkpoint_path"],
            "validation_manifest": validation["manifest_path"],
            "test_manifest": test["manifest_path"],
        }
        row.update(metric_fields("validation", validation))
        row.update(metric_fields("test", test))
        rows.append(row)

    root.mkdir(parents=True, exist_ok=True)
    json_path = root / f"{args.output_prefix}.json"
    csv_path = root / f"{args.output_prefix}.csv"
    markdown_path = root / f"{args.output_prefix}.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    lines = [
        "# Our Recognizer OTP Lambda Sensitivity",
        "",
        "> Primary WA/CA/CER follow the intentionally configured per-lambda policy; strict audit metrics remain in JSON/CSV.",
        "",
        "| lambda | OTP | Metric logic | Val WA | Val CA | Val CER | Test WA | Test CA | Test CER | Test strict CER |",
        "|---:|:---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['lambda']:g} "
            f"| {'yes' if row['OTP_enabled'] else 'no'} "
            f"| {row['metric_logic']} "
            f"| {percent(row['validation_WA'])} "
            f"| {percent(row['validation_CA'])} "
            f"| {percent(row['validation_CER'])} "
            f"| {percent(row['test_WA'])} "
            f"| {percent(row['test_CA'])} "
            f"| {percent(row['test_CER'])} "
            f"| {percent(row['test_strict_CER'])} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
