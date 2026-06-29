import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml


_ENV_DEFAULT_PATTERN = re.compile(r"\$\{([^}:]+):-([^}]+)\}")


def _expand_env_defaults(value: str) -> str:
    def repl(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)
        return os.environ.get(name, default)

    return _ENV_DEFAULT_PATTERN.sub(repl, value)


def _expand_path_like_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_path_like_values(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand_path_like_values(item) for item in value]

    if isinstance(value, str):
        value = _expand_env_defaults(value)
        return os.path.expanduser(os.path.expandvars(value))

    return value


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        path: YAML file path.

    Returns:
        Parsed dictionary.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return _expand_path_like_values(cfg)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """
    Save dictionary to YAML file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def get_by_dotted_key(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """
    Access nested config value by dotted key.

    Example:
        get_by_dotted_key(cfg, "detection_data.raw_images")
    """
    current = cfg

    for key in dotted_key.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]

    return current
