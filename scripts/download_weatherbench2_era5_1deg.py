#!/usr/bin/env python3
"""Stage WeatherBench 2 ERA5 data for GraphCast Small training.

The training loader expects local monthly zarr stores arranged as:

    ANALYSIS_PATH/YYYY/MM

This script mirrors only the variables needed by the GraphCast checkpoint,
subsets to the checkpoint pressure levels, and writes a 1 degree grid suitable
for the GraphCast Small checkpoint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import warnings
from pathlib import Path


DEFAULT_SOURCE = (
    "gs://weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
DEFAULT_CHECKPOINT = Path("params/graphcast_small_lamse.000000.npz")
DEFAULT_OUTPUT_DIR = Path("data/era5_1deg_weatherbench2")


def parse_date(value: str) -> dt.datetime:
    import dateparser

    parsed = dateparser.parse(
        value,
        [
            "%Y%m%d%H",
            "%Y%m%d%HZ",
            "%Y%m%dT%H",
            "%Y%m%dT%HZ",
            "%Y-%m-%dT%H",
            "%Y-%m-%d %H:%M",
        ],
    )
    if parsed is None:
        raise argparse.ArgumentTypeError(f"Could not parse date: {value}")
    return parsed.replace(tzinfo=None)


def month_floor(value: dt.datetime) -> dt.datetime:
    return dt.datetime(value.year, value.month, 1)


def month_after(value: dt.datetime) -> dt.datetime:
    if value.month == 12:
        return dt.datetime(value.year + 1, 1, 1)
    return dt.datetime(value.year, value.month + 1, 1)


def iter_months(start: dt.datetime, end: dt.datetime):
    current = month_floor(start)
    last = month_floor(end)
    while current <= last:
        yield current.year, current.month
        current = month_after(current)


def config_values(config: dict, key: str):
    value = config[key]
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return list(value)


def load_checkpoint_metadata(checkpoint_path: Path):
    import forecast.generate_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}\n"
            "Run scripts/download_graphcast_small.py and "
            "scripts/prepare_graphcast_small_checkpoint.py first."
        )
    _, task_config, _ = forecast.generate_model.load_model(str(checkpoint_path))
    return {
        "pressure_levels": sorted(config_values(task_config, "pressure_levels")),
        "input_variables": set(config_values(task_config, "input_variables")),
        "target_variables": set(config_values(task_config, "target_variables")),
        "forcing_variables": set(config_values(task_config, "forcing_variables")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-start", type=parse_date, default=parse_date("1 Jan 2016 00:00"))
    parser.add_argument("--train-end", type=parse_date, default=parse_date("31 Dec 2017 18:00"))
    parser.add_argument("--forecast-length", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def prepare_month_dataset(ds, metadata: dict, year: int, month: int):
    import numpy as np
    import xarray as xr

    six_hours = dt.timedelta(hours=6)
    one_hour = dt.timedelta(hours=1)
    date_start = dt.datetime(year, month, 1, 0)
    date_end = month_after(date_start) - six_hours

    required_vars = (
        metadata["input_variables"]
        | metadata["target_variables"]
        | metadata["forcing_variables"]
    )

    # These are created inside the loader, not read from ERA5.
    from forecast.forecast_variables import DERIVED_VARIABLES

    derived_vars = set(DERIVED_VARIABLES)
    needs_precip_6hr = "total_precipitation_6hr" in required_vars
    source_vars = required_vars - derived_vars - {"total_precipitation_6hr"}
    download_vars = sorted(source_vars.intersection(ds.data_vars))

    missing = sorted(source_vars - set(download_vars))
    if missing:
        raise ValueError(f"Source dataset is missing required variables: {missing}")
    if needs_precip_6hr and "total_precipitation" not in ds.data_vars:
        raise ValueError("Source dataset is missing total_precipitation")

    output = ds[download_vars].sel(time=slice(date_start, date_end)).isel(time=slice(None, None, 6))
    if "level" in output.coords:
        output = output.sel(level=metadata["pressure_levels"]).sortby("level")

    if output.latitude.data[1] - output.latitude.data[0] < 0:
        output = output.isel(latitude=slice(None, None, -1))
    if output.longitude.data[1] - output.longitude.data[0] < 0:
        output = output.isel(longitude=slice(None, None, -1))

    output = output.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))

    if needs_precip_6hr:
        output_start = output.time.data[0].astype("datetime64[h]").astype(dt.datetime)
        output_end = output.time.data[-1].astype("datetime64[h]").astype(dt.datetime)
        precip = ds["total_precipitation"].sel(time=slice(output_start - 5 * one_hour, output_end))
        if precip.latitude.data[1] - precip.latitude.data[0] < 0:
            precip = precip.isel(latitude=slice(None, None, -1))
        if precip.longitude.data[1] - precip.longitude.data[0] < 0:
            precip = precip.isel(longitude=slice(None, None, -1))
        precip = precip.isel(latitude=slice(None, None, 4), longitude=slice(None, None, 4))

        expected_hours = output.time.size * 6
        if precip.time.size != expected_hours:
            raise ValueError(
                f"Expected {expected_hours} hourly precipitation fields for "
                f"{year}-{month:02d}, got {precip.time.size}"
            )

        precip_data = precip.data.reshape(
            (output.time.size, 6, output.latitude.size, output.longitude.size)
        ).sum(axis=1)
        output["total_precipitation_6hr"] = xr.DataArray(
            precip_data.astype(np.float32),
            dims=("time", "latitude", "longitude"),
            coords={
                "time": output.time,
                "latitude": output.latitude,
                "longitude": output.longitude,
            },
        )

    return output


def main() -> None:
    import dask
    import dask.config
    import dask.distributed
    import xarray as xr

    import forecast.encabulator

    args = build_parser().parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = project_dir / checkpoint
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_dir / output_dir

    required_start = args.train_start - dt.timedelta(hours=6)
    required_end = args.train_end + dt.timedelta(hours=6 * args.forecast_length)

    metadata = load_checkpoint_metadata(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {args.source}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output ANALYSIS_PATH: {output_dir}")
    print(f"Required data window: {required_start} through {required_end}")
    print(f"Pressure levels: {metadata['pressure_levels']}")

    dask.config.set(
        {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
        }
    )

    storage_options = {"session_kwargs": {"trust_env": True}}
    with warnings.catch_warnings(action="ignore"):
        client = dask.distributed.Client(processes=False, threads_per_worker=args.threads)
        ds = xr.open_dataset(
            args.source,
            engine="zarr",
            storage_options=storage_options,
            chunks={"time": 1, "latitude": -1, "longitude": -1, "level": -1},
        )

    try:
        for year, month in iter_months(required_start, required_end):
            out_store = output_dir / f"{year}" / f"{month:02d}"
            complete_marker = out_store.parent / f".{month:02d}.complete"
            if complete_marker.exists() and not args.overwrite:
                print(f"Skipping complete store: {out_store}")
                continue

            out_store.parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing {year}-{month:02d} to {out_store}")
            month_ds = prepare_month_dataset(ds, metadata, year, month)

            compressed_vars = (
                set(month_ds.data_vars)
                - {"geopotential_at_surface", "land_sea_mask", "total_precipitation_6hr"}
                - metadata["forcing_variables"]
            )
            encoding = {
                var: {"compressor": forecast.encabulator.LayerQuantizer(nbits=16)}
                for var in compressed_vars
            }

            delayed = month_ds.to_zarr(
                out_store,
                mode="w",
                encoding=encoding,
                compute=False,
            )
            delayed = dask.optimize(delayed)[0]
            client.compute(delayed, sync=True)
            complete_marker.touch()
            print(f"Finished {year}-{month:02d}")
    except Exception as exc:
        print(f"Data staging failed: {exc}", file=sys.stderr)
        raise
    finally:
        client.close()

    print("\nTraining data ready.")
    print(f"Use ANALYSIS_PATH={output_dir}")


if __name__ == "__main__":
    main()
