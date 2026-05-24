#!/usr/bin/env python3
"""Inspect a GraphCast checkpoint with this fork's loader."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def params_dir() -> Path:
    if os.environ.get("PARAMS_DIR"):
        return Path(os.environ["PARAMS_DIR"]).expanduser()
    if os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "graphcast-small-lamse" / "params"
    return project_dir() / "params"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=params_dir() / "graphcast_small_lamse.000000.npz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(project_dir()))

    from forecast import generate_model

    model_config, task_config, params = generate_model.load_model(str(args.checkpoint))
    resolution = getattr(model_config, "resolution", None)
    if resolution is None and isinstance(model_config, dict):
        resolution = model_config.get("resolution")
    pressure_levels = getattr(task_config, "pressure_levels", None)
    if pressure_levels is None and isinstance(task_config, dict):
        pressure_levels = task_config.get("pressure_levels")

    print(f"checkpoint: {args.checkpoint}")
    print(f"model resolution: {resolution}")
    print(f"pressure levels: {len(pressure_levels) if pressure_levels is not None else 'unknown'}")
    print(f"parameter groups: {len(params)}")


if __name__ == "__main__":
    main()
