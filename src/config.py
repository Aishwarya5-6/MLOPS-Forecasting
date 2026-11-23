"""Load params.yaml once, so every module reads the same settings."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PARAMS_FILE = ROOT / "params.yaml"


def load_params(path: Path = PARAMS_FILE) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)
