import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt


def plot_history(
    metrics_paths: Sequence[str | Path],
    metric_key: str,
    output_path: str | Path,
    labels: Sequence[str] | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for idx, path in enumerate(metrics_paths):
        path = Path(path)
        history = json.loads(path.read_text(encoding="utf-8"))
        label = labels[idx] if labels is not None else path.parent.name
        plt.plot([x["epoch"] for x in history], [x[metric_key] for x in history], label=label)

    plt.xlabel("Epoch")
    plt.ylabel(metric_key)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
