import argparse
from pathlib import Path

from manchu_ocr.utils.config import load_yaml


def count_manifest(path: str | Path) -> int:
    path = Path(path)
    if not path.exists():
        print(f"[MISSING] {path}")
        return 0
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"[OK] {path}: {count} samples")
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/paths/remote_server.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    rec = cfg.get("recognition_data", {})
    if rec:
        print("[Recognition]")
        for key in ["train_list", "val_list", "test_list", "charset", "transition_matrix"]:
            path = Path(rec[key])
            print(f"[{'OK' if path.exists() else 'MISSING'}] {key}: {path}")
        for key in ["train_list", "val_list", "test_list"]:
            count_manifest(rec[key])

    det = cfg.get("detection_data", {})
    if det:
        print("[Detection]")
        for key in ["train_list", "val_list", "test_list"]:
            count_manifest(det[key])


if __name__ == "__main__":
    main()

