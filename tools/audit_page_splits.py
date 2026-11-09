"""Check detection, recognition and E2E manifests against explicit page IDs."""
import argparse
import json
from pathlib import Path

from manchu_ocr.data.page_splits import audit_page_splits, load_page_mapping
from manchu_ocr.utils.config import load_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paths/remote_server.yaml")
    parser.add_argument("--page-map", required=True)
    parser.add_argument("--output", default="outputs/metrics/page_split_audit.json")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    sections = {task: {split: cfg[task][f"{split}_list"] for split in ("train", "val", "test")
                       if cfg[task].get(f"{split}_list") and not (task == "e2e_data" and split == "train")}
                for task in ("detection_data", "recognition_data", "e2e_data") if task in cfg}
    report = audit_page_splits(sections, load_page_mapping(args.page_map))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
