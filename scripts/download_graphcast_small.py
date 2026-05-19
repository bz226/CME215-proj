#!/usr/bin/env python3
"""Download the Hugging Face GraphCast Small checkpoint for this fork."""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path


REPO_ID = "shermansiu/dm_graphcast_small"
FILENAME = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 "
    "- mesh 2to5 - precipitation input and output.npz"
)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def direct_download_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{urllib.parse.quote(filename)}"


def download_with_hub(repo_id: str, filename: str, output_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    output = output_dir / filename
    if not output.exists():
        shutil.copy2(cached, output)
    return output


def download_direct(repo_id: str, filename: str, output_dir: Path) -> Path:
    output = output_dir / filename
    if output.exists():
        return output
    with urllib.request.urlopen(direct_download_url(repo_id, filename)) as response, output.open("wb") as f:
        shutil.copyfileobj(response, f)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--filename", default=FILENAME)
    parser.add_argument("--output-dir", type=Path, default=project_dir() / "params")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        output = download_with_hub(args.repo_id, args.filename, args.output_dir)
    except ModuleNotFoundError:
        print("huggingface_hub is not installed; falling back to direct HTTPS download.")
        output = download_direct(args.repo_id, args.filename, args.output_dir)
    except Exception as exc:
        print(f"huggingface_hub download failed: {exc}", file=sys.stderr)
        print("Falling back to direct HTTPS download.")
        output = download_direct(args.repo_id, args.filename, args.output_dir)
    print(f"Checkpoint ready: {output}")


if __name__ == "__main__":
    main()
