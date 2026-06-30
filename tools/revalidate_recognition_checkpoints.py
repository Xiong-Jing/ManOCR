import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

from manchu_ocr.utils.config import load_yaml


DEFAULT_CONFIGS = [
    "configs/recognition/svtr_official_baseline.yaml",
    "configs/recognition/svtr_official_dab.yaml",
    "configs/recognition/svtr_official_lortho.yaml",
    "configs/recognition/svtr_official_dab_lortho.yaml",
]


def safe_tag(path: Path) -> str:
    tag = path.stem
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    return tag


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def discover_checkpoints(ckpt_dir: Path, mode: str) -> List[Path]:
    if not ckpt_dir.exists():
        return []

    if mode == "best-last":
        candidates = [ckpt_dir / "best.pth", ckpt_dir / "last.pth"]
    elif mode == "all":
        candidates = sorted(ckpt_dir.glob("*.pth"))
    else:
        raise ValueError(f"Unsupported checkpoint mode: {mode}")

    seen = set()
    result = []

    for path in candidates:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)

    return result


def run_eval(
    python_bin: str,
    config_path: Path,
    checkpoint_path: Path,
    split: str,
    batch_size: int,
    num_workers: int,
    decode_mode: str,
    word_accuracy_edit_distance: int,
    character_accuracy_edit_distance: int,
    word_accuracy_require_first_char_match: bool,
    word_accuracy_require_second_char_match: bool,
    word_accuracy_require_last_char_match: bool,
    max_batches: int | None,
    output_suffix: str,
) -> None:
    cmd = [
        python_bin,
        "scripts/eval_recognition.py",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--output-suffix",
        output_suffix,
        "--word-accuracy-edit-distance",
        str(word_accuracy_edit_distance),
        "--character-accuracy-edit-distance",
        str(character_accuracy_edit_distance),
    ]

    if word_accuracy_require_first_char_match:
        cmd.append("--word-accuracy-require-first-char-match")
    if word_accuracy_require_second_char_match:
        cmd.append("--word-accuracy-require-second-char-match")
    if word_accuracy_require_last_char_match:
        cmd.append("--word-accuracy-require-last-char-match")

    if max_batches is not None:
        cmd.extend(["--max-batches", str(max_batches)])

    if decode_mode == "lexicon":
        cmd.extend(["--disable-ctc-rerank"])
    elif decode_mode == "ctc":
        cmd.extend(["--enable-ctc-rerank"])
    elif decode_mode == "greedy":
        cmd.extend(["--disable-lexicon"])
    else:
        raise ValueError(f"Unsupported decode mode: {decode_mode}")

    print("[EVAL]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def plot_revalidated_curves(rows: List[Dict], out_dir: Path, split: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    by_exp: Dict[str, List[Dict]] = {}
    for row in rows:
        by_exp.setdefault(row["experiment"], []).append(row)

    metrics = [
        ("word_accuracy", "Word Accuracy (%)", f"fig_recognition_revalidated_{split}_wa.png"),
        ("character_accuracy", "Character Accuracy (%)", f"fig_recognition_revalidated_{split}_ca.png"),
        ("cer", "CER (%)", f"fig_recognition_revalidated_{split}_cer.png"),
    ]

    for metric_key, ylabel, filename in metrics:
        plt.figure(figsize=(8, 5))
        plotted = False

        for exp_name, exp_rows in by_exp.items():
            exp_rows = sorted(exp_rows, key=lambda item: (item["epoch"], item["checkpoint_tag"]))
            xs = [item["epoch"] for item in exp_rows]
            ys = [item[metric_key] for item in exp_rows]

            if metric_key in {"word_accuracy", "character_accuracy", "cer"}:
                ys = [value * 100 for value in ys]

            plt.plot(xs, ys, marker="o", label=exp_name)
            plotted = True

        if not plotted:
            plt.close()
            continue

        plt.xlabel("Checkpoint Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        output_path = out_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OK] saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint-mode", default="best-last", choices=["best-last", "all"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--decode-mode", default="lexicon", choices=["lexicon", "ctc", "greedy"])
    parser.add_argument("--word-accuracy-edit-distance", type=int, default=None)
    parser.add_argument("--character-accuracy-edit-distance", type=int, default=None)
    parser.add_argument(
        "--word-accuracy-require-first-char-match",
        action="store_true",
    )
    parser.add_argument(
        "--word-accuracy-require-second-char-match",
        action="store_true",
    )
    parser.add_argument(
        "--word-accuracy-require-last-char-match",
        action="store_true",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out-dir", default="outputs/metrics/recognition/revalidated")
    parser.add_argument("--fig-dir", default="outputs/visualizations/paper_figures")
    args = parser.parse_args()

    all_rows = []

    for config in args.configs:
        config_path = Path(config)
        cfg = load_yaml(config_path)
        paths_cfg = load_yaml(cfg["experiment"]["paths_config"])

        exp_name = cfg["experiment"]["name"]
        metrics_cfg = cfg.get("metrics", {})
        word_accuracy_edit_distance = (
            int(args.word_accuracy_edit_distance)
            if args.word_accuracy_edit_distance is not None
            else int(metrics_cfg.get("word_accuracy_edit_distance", 0))
        )
        character_accuracy_edit_distance = (
            int(args.character_accuracy_edit_distance)
            if args.character_accuracy_edit_distance is not None
            else int(metrics_cfg.get("character_accuracy_edit_distance", 0))
        )
        word_accuracy_require_first_char_match = bool(
            args.word_accuracy_require_first_char_match
            or metrics_cfg.get("word_accuracy_require_first_char_match", False)
        )
        word_accuracy_require_second_char_match = bool(
            args.word_accuracy_require_second_char_match
            or metrics_cfg.get("word_accuracy_require_second_char_match", False)
        )
        word_accuracy_require_last_char_match = bool(
            args.word_accuracy_require_last_char_match
            or metrics_cfg.get("word_accuracy_require_last_char_match", False)
        )
        output_root = Path(paths_cfg["outputs"]["root"])
        ckpt_dir = output_root / "checkpoints" / "recognition" / exp_name
        metrics_dir = output_root / "metrics" / "recognition" / exp_name

        checkpoints = discover_checkpoints(ckpt_dir, mode=args.checkpoint_mode)

        if not checkpoints:
            print(f"[WARN] no checkpoints found: {ckpt_dir}")
            continue

        for checkpoint_path in checkpoints:
            tag = safe_tag(checkpoint_path)
            suffix = (
                f"_reval_{tag}_wedit{word_accuracy_edit_distance}"
                f"_cedit{character_accuracy_edit_distance}"
            )
            if word_accuracy_require_first_char_match:
                suffix = f"{suffix}_firstchar"
            if word_accuracy_require_second_char_match:
                suffix = f"{suffix}_secondchar"
            if word_accuracy_require_last_char_match:
                suffix = f"{suffix}_lastchar"

            run_eval(
                python_bin=args.python,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                decode_mode=args.decode_mode,
                word_accuracy_edit_distance=word_accuracy_edit_distance,
                character_accuracy_edit_distance=character_accuracy_edit_distance,
                word_accuracy_require_first_char_match=word_accuracy_require_first_char_match,
                word_accuracy_require_second_char_match=word_accuracy_require_second_char_match,
                word_accuracy_require_last_char_match=word_accuracy_require_last_char_match,
                max_batches=args.max_batches,
                output_suffix=suffix,
            )

            result_path = metrics_dir / f"eval_{args.split}{suffix}.json"
            result = load_json(result_path)
            metrics = result["metrics"]

            epoch = result.get("checkpoint_epoch")
            if epoch is None:
                # eval_recognition stores the checkpoint path but not epoch in result.
                # Read epoch lazily from torch-free metadata is not available, so use
                # the checkpoint name order if needed. The eval log still shows epoch.
                epoch = metrics.get("epoch", -1)

            # Prefer parsing common names; best/last still stay comparable by tag.
            if epoch == -1:
                match = re.search(r"epoch[_-]?(\d+)", checkpoint_path.stem)
                epoch = int(match.group(1)) if match else -1

            all_rows.append(
                {
                    "experiment": exp_name,
                    "config": str(config_path).replace("\\", "/"),
                    "checkpoint": str(checkpoint_path).replace("\\", "/"),
                    "checkpoint_tag": tag,
                    "epoch": int(epoch),
                    "split": args.split,
                    "decode_mode": args.decode_mode,
                    "loss": float(metrics["loss"]),
                    "ctc_loss": float(metrics.get("ctc_loss", metrics["loss"])),
                    "word_accuracy": float(metrics["word_accuracy"]),
                    "character_accuracy": float(metrics["character_accuracy"]),
                    "cer": float(metrics["cer"]),
                    "strict_character_accuracy": float(metrics.get("strict_character_accuracy", metrics["character_accuracy"])),
                    "strict_cer": float(metrics.get("strict_cer", metrics["cer"])),
                    "edit_distance": float(metrics["edit_distance"]),
                    "num_samples": int(metrics.get("num_samples", 0)),
                }
            )

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"recognition_revalidated_{args.split}.json"
    save_json(all_rows, out_path)
    print(f"[OK] saved {out_path}")

    if all_rows:
        plot_revalidated_curves(
            rows=all_rows,
            out_dir=Path(args.fig_dir),
            split=args.split,
        )

        unique_epochs = sorted({row["epoch"] for row in all_rows})
        if args.checkpoint_mode == "best-last" or len(unique_epochs) <= 2:
            print(
                "[WARN] Only best/last or very few checkpoints were evaluated. "
                "This is a unified post-training revalidation, not a full per-epoch curve. "
                "A full historical curve requires saved epoch checkpoints."
            )


if __name__ == "__main__":
    main()
