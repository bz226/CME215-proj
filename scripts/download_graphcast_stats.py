#!/usr/bin/env python3
"""Download GraphCast normalization statistics needed by training."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


FILES = (
    "diffs_stddev_by_level.nc",
    "mean_by_level.nc",
    "stddev_by_level.nc",
)

HF_DATASET_REPO = "shermansiu/dm_graphcast_datasets"
GCS_BASES = (
    "https://storage.googleapis.com/dm_graphcast/graphcast/stats",
    "https://storage.googleapis.com/dm_graphcast/stats",
)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def download_with_hub(filename: str, output_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cached = Path(
        hf_hub_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            filename=f"stats/{filename}",
        )
    )
    output = output_dir / filename
    shutil.copy2(cached, output)
    return output


def direct_url(base: str, filename: str) -> str:
    return f"{base}/{urllib.parse.quote(filename)}"


def download_direct(filename: str, output_dir: Path) -> Path:
    errors = []
    for base in GCS_BASES:
        url = direct_url(base, filename)
        output = output_dir / filename
        try:
            with urllib.request.urlopen(url, timeout=60) as response, output.open("wb") as f:
                shutil.copyfileobj(response, f)
            return output
        except Exception as exc:
            errors.append(f"{url}: {exc}")
            if output.exists():
                output.unlink()
    raise RuntimeError("Could not download stats file:\n" + "\n".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=project_dir() / "stats")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        output = args.output_dir / filename
        if output.exists() and not args.force:
            print(f"Already exists: {output}", flush=True)
            continue
        try:
            output = download_direct(filename, args.output_dir)
        except Exception as direct_exc:
            print(f"Direct GCS download failed for {filename}: {direct_exc}", file=sys.stderr, flush=True)
            print("Falling back to Hugging Face dataset mirror.", flush=True)
            output = download_with_hub(filename, args.output_dir)
        print(f"Stats file ready: {output}", flush=True)


if __name__ == "__main__":
    main()
