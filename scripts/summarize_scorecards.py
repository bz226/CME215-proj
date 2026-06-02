#!/usr/bin/env python3
"""Summarize scorecard zarr outputs into one CSV table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_name_path(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def lead_hours(value) -> float:
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return float(arr / np.timedelta64(1, "h"))
    return float(arr)


def scalar(value) -> float:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape {arr.shape}")
    return float(arr.reshape(()))


def iter_manifest(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score_path = row.get("score_path", "")
            if score_path:
                yield row["model"], Path(score_path)


def summarize_run(model: str, path: Path) -> list[dict[str, object]]:
    import xarray as xr

    ds = xr.open_zarr(path)
    rows: list[dict[str, object]] = []
    stat_values = list(ds.coords["stat"].values) if "stat" in ds.coords else [None]
    lead_values = list(ds.coords["lead_time"].values) if "lead_time" in ds.coords else [None]

    for variable in ds.data_vars:
        base = ds[variable]
        for stat in stat_values:
            da_stat = base.sel(stat=stat) if stat is not None and "stat" in base.dims else base
            for lead in lead_values:
                da = da_stat.sel(lead_time=lead) if lead is not None and "lead_time" in da_stat.dims else da_stat
                reduce_dims = [dim for dim in da.dims if dim == "idate"]
                count_da = da.count(dim=reduce_dims) if reduce_dims else da.count()
                mean_da = da.mean(dim=reduce_dims, skipna=True) if reduce_dims else da
                median_da = da.median(dim=reduce_dims, skipna=True) if reduce_dims else da

                kept_dims = list(mean_da.dims)
                if kept_dims:
                    for index in np.ndindex(*(mean_da.sizes[dim] for dim in kept_dims)):
                        selector = dict(zip(kept_dims, index))
                        row = {
                            "model": model,
                            "score_path": str(path),
                            "variable": variable,
                            "stat": "" if stat is None else str(stat),
                            "lead_hours": "" if lead is None else lead_hours(lead),
                            "mean": scalar(mean_da.isel(selector).values),
                            "median": scalar(median_da.isel(selector).values),
                            "count": int(scalar(count_da.isel(selector).values)),
                        }
                        for dim, idx in selector.items():
                            coord = mean_da.coords[dim].values[idx] if dim in mean_da.coords else idx
                            row[dim] = coord.item() if hasattr(coord, "item") else coord
                        rows.append(row)
                else:
                    rows.append(
                        {
                            "model": model,
                            "score_path": str(path),
                            "variable": variable,
                            "stat": "" if stat is None else str(stat),
                            "lead_hours": "" if lead is None else lead_hours(lead),
                            "mean": scalar(mean_da.values),
                            "median": scalar(median_da.values),
                            "count": int(scalar(count_da.values)),
                        }
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="TSV written by submit_scorecard_2022_all.sh")
    parser.add_argument("--score", action="append", default=[], type=parse_name_path, metavar="NAME=PATH")
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    runs = list(args.score)
    if args.manifest:
        runs.extend(iter_manifest(args.manifest))
    if not runs:
        raise SystemExit("Provide --manifest or at least one --score NAME=PATH")

    rows: list[dict[str, object]] = []
    for model, path in runs:
        rows.extend(summarize_run(model, path))

    fieldnames = [
        "model",
        "variable",
        "stat",
        "lead_hours",
        "level",
        "mean",
        "median",
        "count",
        "score_path",
    ]
    extra = sorted({key for row in rows for key in row if key not in fieldnames})
    fieldnames = fieldnames[:-1] + extra + ["score_path"]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
