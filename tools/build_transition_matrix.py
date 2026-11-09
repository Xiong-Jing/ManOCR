import argparse
from pathlib import Path

from manchu_ocr.data.orthography import build_transition_matrix, write_transition_artifact
from manchu_ocr.data.page_splits import read_manifest

from manchu_ocr.utils.config import load_yaml


def read_charset(path: Path) -> list[str]:
    return [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.rstrip("\n")]


def read_labels(path: Path) -> list[str]:
    return [label for _, label in read_manifest(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--charset", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--epsilon-t", type=float, default=1e-6)
    args = parser.parse_args()

    manifest = args.manifest
    charset_path = args.charset
    output = args.output

    if args.config is not None:
        cfg = load_yaml(args.config)
        rec_cfg = cfg["recognition_data"]
        manifest = manifest or rec_cfg["train_list"]
        charset_path = charset_path or rec_cfg["charset"]
        output = output or rec_cfg["transition_matrix"]

    if manifest is None or charset_path is None or output is None:
        raise ValueError("Either provide --config or provide --manifest, --charset, and --output.")

    charset = read_charset(Path(charset_path))
    labels = read_labels(Path(manifest))
    if not labels:
        raise ValueError("Training manifest is empty")
    matrix = build_transition_matrix(labels, charset, smoothing=args.smoothing,
                                     rho=args.rho, epsilon_t=args.epsilon_t)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_transition_artifact(output_path, matrix, manifest=manifest, charset_path=charset_path,
                              rho=args.rho, smoothing=args.smoothing, epsilon_t=args.epsilon_t)
    print(f"[OK] saved transition matrix {matrix.shape} to {output_path}")


if __name__ == "__main__":
    main()
