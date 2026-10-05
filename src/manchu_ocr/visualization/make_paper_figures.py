from pathlib import Path

from manchu_ocr.visualization.plot_curves import plot_history


def make_detection_figures(output_dir: str | Path = "outputs/visualizations/paper_figures") -> None:
    output_dir = Path(output_dir)
    experiments = [
        "dbnetpp_official_baseline",
        "dbnetpp_vsaa",
        "dbnetpp_asym_shrink",
        "dbnetpp_vsaa_asym_shrink",
    ]
    paths = [Path("outputs/metrics/detection") / exp / "metrics.json" for exp in experiments]
    labels = ["DBNet++", "+VSAA", "+AS", "+VSAA+AS"]

    plot_history(paths, "train_loss", output_dir / "fig_detection_train_loss.png", labels)
    plot_history(paths, "val_loss", output_dir / "fig_detection_val_loss.png", labels)
    plot_history(paths, "val_fmeasure", output_dir / "fig_detection_fmeasure.png", labels)
