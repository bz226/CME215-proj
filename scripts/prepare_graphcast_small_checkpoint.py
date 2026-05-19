#!/usr/bin/env python3
"""Create train.py-compatible numeric checkpoint alias for GraphCast Small."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from download_graphcast_small import FILENAME, project_dir


def parse_args() -> argparse.Namespace:
    root = project_dir()
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=root / "params" / FILENAME)
    parser.add_argument("--target", type=Path, default=root / "params" / "graphcast_small_lamse.000000.npz")
    parser.add_argument("--mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    target = args.target
    if not source.exists():
        raise FileNotFoundError(
            f"Missing source checkpoint: {source}\n"
            "Run scripts/download_graphcast_small.py first, or pass --source."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not args.force:
            print(f"Already exists: {target}")
            return
        target.unlink()

    if args.mode == "symlink":
        os.symlink(source, target)
    else:
        shutil.copy2(source, target)
    print(f"Prepared training checkpoint: {target}")


if __name__ == "__main__":
    main()
