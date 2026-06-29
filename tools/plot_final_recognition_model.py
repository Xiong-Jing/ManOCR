import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict:
    data = load_json(path)
    return data.get("metrics", data)


def plot_train_val_curves(root: Path, out_dir: Path, exp_name: str, label: str) -> None:
    history_path = root / exp_name / "metrics.json"
    history = load_json(history_path)

    epochs = [item["epoch"] for item in history]
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["train_loss"] for item in history], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title(f"{label} Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = out_dir / f"fig_final_recognition_{exp_name}_loss.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [item["val_word_accuracy"] * 100 for item in history], label="Val WA")
    plt.plot(epochs, [item["val_character_accuracy"] * 100 for item in history], label="Val CA")
    plt.plot(epochs, [item["val_cer"] * 100 for item in history], label="Val CER")
    plt.xlabel("Epoch")
    plt.ylabel("Metric (%)")
    plt.title(f"{label} Validation Metrics")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path = out_dir / f"fig_final_recognition_{exp_name}_val_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    component_keys = [
        ("val_ctc_loss", "Val CTC"),
        ("val_align_loss", "Val Align"),
        ("val_ortho_loss", "Val Ortho"),
    ]

    available = [(key, name) for key, name in component_keys if key in history[0]]

    if available:
        plt.figure(figsize=(8, 5))

        for key, name in available:
            plt.plot(epochs, [item[key] for item in history], label=name)

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{label} Validation Loss Components")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        output_path = out_dir / f"fig_final_recognition_{exp_name}_loss_components.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[OK] saved {output_path}")


def plot_val_test_summary(root: Path, out_dir: Path, exp_name: str, label: str) -> None:
    val = load_metrics(root / exp_name / "eval_val.json")
    test = load_metrics(root / exp_name / "eval_test.json")

    names = ["WA", "CA", "CER"]
    val_values = [
        val["word_accuracy"] * 100,
        val["character_accuracy"] * 100,
        val["cer"] * 100,
    ]
    test_values = [
        test["word_accuracy"] * 100,
        test["character_accuracy"] * 100,
        test["cer"] * 100,
    ]

    x = list(range(len(names)))
    width = 0.36

    plt.figure(figsize=(8, 5))
    plt.bar([item - width / 2 for item in x], val_values, width=width, label="Val")
    plt.bar([item + width / 2 for item in x], test_values, width=width, label="Test")
    plt.xticks(x, names)
    plt.ylabel("Metric (%)")
    plt.title(f"{label} Final Val/Test Metrics")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    output_path = out_dir / f"fig_final_recognition_{exp_name}_val_test_metrics.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] saved {output_path}")

    summary = {
        "experiment": exp_name,
        "val": {
            "word_accuracy": val["word_accuracy"],
            "character_accuracy": val["character_accuracy"],
            "cer": val["cer"],
            "edit_distance": val["edit_distance"],
            "num_samples": val.get("num_samples", 0),
        },
        "test": {
            "word_accuracy": test["word_accuracy"],
            "character_accuracy": test["character_accuracy"],
            "cer": test["cer"],
            "edit_distance": test["edit_distance"],
            "num_samples": test.get("num_samples", 0),
        },
    }

    output_json = out_dir / f"final_recognition_{exp_name}_val_test_summary.json"
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] saved {output_json}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs/metrics/recognition")
    parser.add_argument("--out-dir", type=str, default="outputs/visualizations/paper_figures/main_models")
    parser.add_argument("--exp-name", type=str, default="svtr_official_dab_lortho")
    parser.add_argument("--label", type=str, default="SVTR + DAB + Lortho")
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)

    plot_train_val_curves(
        root=root,
        out_dir=out_dir,
        exp_name=args.exp_name,
        label=args.label,
    )
    plot_val_test_summary(
        root=root,
        out_dir=out_dir,
        exp_name=args.exp_name,
        label=args.label,
    )


if __name__ == "__main__":
    main()
