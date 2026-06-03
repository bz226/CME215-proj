#!/usr/bin/env python3
"""Summarize scorecard zarr outputs into one CSV table."""

import argparse
import csv
from pathlib import Path
import warnings

import numpy as np


def parse_name_path(spec):
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {spec!r}")
    return name, Path(path)


def lead_hours(value):
    arr = np.asarray(value)
    if np.issubdtype(arr.dtype, np.timedelta64):
        return float(arr / np.timedelta64(1, "h"))
    return float(arr)


def scalar(value):
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape {arr.shape}")
    return float(arr.reshape(()))


def coord_value(da, dim, idx):
    coord = da.coords[dim].values[idx] if dim in da.coords else idx
    return coord.item() if hasattr(coord, "item") else coord


def reduce_over_idate(da):
    reduce_dims = [dim for dim in da.dims if dim == "idate"]
    kept_dims = [dim for dim in da.dims if dim not in reduce_dims]
    values = np.asarray(da.data, dtype=np.float64)

    if reduce_dims:
        axes = tuple(da.get_axis_num(dim) for dim in reduce_dims)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_values = np.nanmean(values, axis=axes)
            median_values = np.nanmedian(values, axis=axes)
        count_values = np.sum(np.isfinite(values), axis=axes)
    else:
        mean_values = values
        median_values = values
        count_values = np.isfinite(values).astype(np.int64)

    return kept_dims, mean_values, median_values, count_values


def iter_manifest(path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            score_path = row.get("score_path", "")
            if score_path:
                yield row["model"], Path(score_path)


def summarize_run(model, path):
    import xarray as xr

    # Scorecard zarrs are compact summary arrays. Loading them up front avoids
    # Dask's unsupported full-axis nanmedian path when reducing over idate.
    ds = xr.open_zarr(path).load()
    rows = []
    stat_values = list(ds.coords["stat"].values) if "stat" in ds.coords else [None]
    lead_values = list(ds.coords["lead_time"].values) if "lead_time" in ds.coords else [None]

    for variable in ds.data_vars:
        base = ds[variable]
        for stat in stat_values:
            da_stat = base.sel(stat=stat) if stat is not None and "stat" in base.dims else base
            for lead in lead_values:
                da = da_stat.sel(lead_time=lead) if lead is not None and "lead_time" in da_stat.dims else da_stat
                kept_dims, mean_values, median_values, count_values = reduce_over_idate(da)
                if kept_dims:
                    for index in np.ndindex(*(da.sizes[dim] for dim in kept_dims)):
                        selector = dict(zip(kept_dims, index))
                        row = {
                            "model": model,
                            "score_path": str(path),
                            "variable": variable,
                            "stat": "" if stat is None else str(stat),
                            "lead_hours": "" if lead is None else lead_hours(lead),
                            "mean": scalar(mean_values[index]),
                            "median": scalar(median_values[index]),
                            "count": int(scalar(count_values[index])),
                        }
                        for dim, idx in selector.items():
                            row[dim] = coord_value(da, dim, idx)
                        rows.append(row)
                else:
                    rows.append(
                        {
                            "model": model,
                            "score_path": str(path),
                            "variable": variable,
                            "stat": "" if stat is None else str(stat),
                            "lead_hours": "" if lead is None else lead_hours(lead),
                            "mean": scalar(mean_values),
                            "median": scalar(median_values),
                            "count": int(scalar(count_values)),
                        }
                    )
    return rows


def main():
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

    rows = []
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
