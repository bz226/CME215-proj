#!/usr/bin/env python3
"""Save and plot selected GraphCast prediction errors."""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import shutil

if os.environ.get("JAX_PLATFORMS", "") in ("", "cuda"):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import numpy as np


FIELD_ALIASES = {
    "2t": ("2m_temperature", None),
    "10u": ("10m_u_component_of_wind", None),
    "10v": ("10m_v_component_of_wind", None),
    "10m_wind_speed": ("10m_wind_speed", None),
    "msl": ("mean_sea_level_pressure", None),
    "tp": ("total_precipitation_6hr", None),
}

LEVEL_ALIASES = {
    "z": "geopotential",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
}


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


def parse_field(spec: str) -> tuple[str, int | None, str]:
    """Parse a plotting field such as z500, t850, 2t, or geopotential:500."""

    spec = spec.strip()
    if not spec:
        raise argparse.ArgumentTypeError("Empty field spec")
    if spec in FIELD_ALIASES:
        var, level = FIELD_ALIASES[spec]
        return var, level, spec
    if ":" in spec:
        var, level_text = spec.split(":", 1)
        return var, int(level_text), f"{var}_{level_text}"
    match = re.fullmatch(r"([ztuvq])(\d+)", spec)
    if match:
        var = LEVEL_ALIASES[match.group(1)]
        level = int(match.group(2))
        return var, level, spec
    return spec, None, spec.replace("/", "_")


def unwrap_dataset(in_ds: xr.Dataset) -> xr.Dataset:
    """Move a JAX-backed xarray Dataset to normal host-backed arrays."""

    from graphcast import xarray_jax

    return xr.Dataset(
        {
            var: (in_ds[var].dims, np.asarray(xarray_jax.unwrap_data(in_ds[var])))
            for var in in_ds.data_vars
        },
        coords=in_ds.coords,
    )


def as_numpy_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Compute dask arrays and unwrap JAX arrays if present."""

    try:
        return unwrap_dataset(ds)
    except Exception:
        return ds.compute()


def select_field(ds: xr.Dataset, var: str, level: int | None, lead_hours: int) -> xr.DataArray:
    lead = np.timedelta64(int(lead_hours), "h")
    da = ds[var].sel(time=lead)
    if "batch" in da.dims:
        da = da.isel(batch=0, drop=True)
    if level is not None:
        da = da.sel(level=level)
    elif "level" in da.dims:
        raise ValueError(f"Field {var!r} has a level dimension; use e.g. {var}:500")
    return da.reset_coords(drop=True)


def add_derived(ds: xr.Dataset) -> xr.Dataset:
    out = ds.copy()
    if {"10m_u_component_of_wind", "10m_v_component_of_wind"}.issubset(out.data_vars):
        out["10m_wind_speed"] = (
            out["10m_u_component_of_wind"] ** 2 + out["10m_v_component_of_wind"] ** 2
        ) ** 0.5
    if {"u_component_of_wind", "v_component_of_wind"}.issubset(out.data_vars):
        out["wind_speed"] = (
            out["u_component_of_wind"] ** 2 + out["v_component_of_wind"] ** 2
        ) ** 0.5
    return out


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_triplet(
    target: xr.DataArray,
    prediction: xr.DataArray,
    error: xr.DataArray,
    title: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = target["lon"].data
    lat = target["lat"].data
    target_values = np.asarray(target)
    prediction_values = np.asarray(prediction)
    error_values = np.asarray(error)

    combined = np.concatenate([target_values.ravel(), prediction_values.ravel()])
    vmin, vmax = np.nanpercentile(combined, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    err_abs = np.nanpercentile(np.abs(error_values), 99)
    if not np.isfinite(err_abs) or err_abs == 0:
        err_abs = np.nanmax(np.abs(error_values))
    if not np.isfinite(err_abs) or err_abs == 0:
        err_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    panels = [
        ("Target", target_values, "viridis", vmin, vmax),
        ("Prediction", prediction_values, "viridis", vmin, vmax),
        ("Prediction - Target", error_values, "RdBu_r", -err_abs, err_abs),
    ]
    for ax, (panel_title, values, cmap, panel_vmin, panel_vmax) in zip(axes, panels):
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap=cmap, vmin=panel_vmin, vmax=panel_vmax)
        ax.set_title(panel_title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.86)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--apath", required=True, help="Analysis/verification database path")
    parser.add_argument("--norm-factors", default="stats")
    parser.add_argument("--init-date", type=parse_date, required=True)
    parser.add_argument("--forecast-length", type=int, default=240, help="Maximum lead in hours")
    parser.add_argument("--lead-hours", type=int, nargs="+", default=[6, 120, 240])
    parser.add_argument(
        "--fields",
        nargs="+",
        default=["z500", "t850", "2t", "10m_wind_speed", "msl"],
        help="Fields such as z500, t850, 2t, 10m_wind_speed, or geopotential:500",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/prediction_error"))
    parser.add_argument("--output-zarr", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    import dask
    import jax
    import xarray as xr

    import forecast.generate_model
    import trainer.dataloader
    if args.forecast_length % 6 != 0:
        raise ValueError("--forecast-length must be a multiple of 6 hours")
    if any(lead % 6 != 0 for lead in args.lead_hours):
        raise ValueError("--lead-hours values must be multiples of 6")
    if max(args.lead_hours) > args.forecast_length:
        raise ValueError("All --lead-hours must be <= --forecast-length")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_zarr = args.output_zarr or args.out_dir / "prediction_error_fields.zarr"

    print("devices", jax.devices())
    print(f"Loading checkpoint: {args.model_checkpoint}")
    model_config, task_config, params = forecast.generate_model.load_model(args.model_checkpoint)
    model_latitude, model_longitude = forecast.generate_model.get_model_coords(model_config)
    input_variables = list(task_config["input_variables"])
    target_variables = list(task_config["target_variables"])

    norm_path = args.norm_factors
    diffs_stddev_by_level = xr.load_dataset(f"{norm_path}/diffs_stddev_by_level.nc").compute()
    mean_by_level = xr.load_dataset(f"{norm_path}/mean_by_level.nc").compute()
    stddev_by_level = xr.load_dataset(f"{norm_path}/stddev_by_level.nc").compute()

    predictor = forecast.generate_model.build_predictor_params(
        model_config,
        task_config,
        use_float16=True,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    print(f"Opening data: {args.apath}")
    dbase, _ = trainer.dataloader.open_databases(args.apath, None)
    forecast_steps = args.forecast_length // 6
    print(f"Building forecast init={args.init_date} steps={forecast_steps}")
    inputs, forcings, targets = trainer.dataloader.build_forecast(
        args.init_date,
        forecast_steps,
        task_config,
        model_latitude,
        model_longitude,
        input_variables,
        target_variables,
        dbase,
        dbase,
    )
    inputs, forcings, targets = dask.compute(inputs, forcings, targets)

    print("Running model")
    predictions = predictor(inputs=inputs, targets=targets, forcings=forcings, params=params)
    predictions = add_derived(as_numpy_dataset(predictions))
    targets = add_derived(as_numpy_dataset(targets))

    data_vars = {}
    attrs = {
        "model_checkpoint": args.model_checkpoint,
        "init_date": args.init_date.isoformat(),
        "forecast_length_hours": args.forecast_length,
    }
    for field_spec in args.fields:
        var, level, label = parse_field(field_spec)
        if var not in predictions:
            print(f"Skipping {field_spec}: {var} not in predictions")
            continue
        if var not in targets:
            print(f"Skipping {field_spec}: {var} not in targets")
            continue
        for lead in args.lead_hours:
            prediction_field = select_field(predictions, var, level, lead)
            target_field = select_field(targets, var, level, lead)
            error_field = prediction_field - target_field
            suffix = safe_name(f"{label}_{lead:03d}h")
            data_vars[f"prediction_{suffix}"] = prediction_field
            data_vars[f"target_{suffix}"] = target_field
            data_vars[f"error_{suffix}"] = error_field
            if not args.no_plots:
                title = f"{label} lead={lead}h init={args.init_date:%Y-%m-%d %H:%M}"
                plot_path = args.out_dir / f"{suffix}_error.png"
                plot_triplet(target_field, prediction_field, error_field, title, plot_path)
                print(f"Wrote {plot_path}")

    output = xr.Dataset(data_vars, attrs=attrs)
    if output_zarr.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_zarr} exists; pass --overwrite")
        shutil.rmtree(output_zarr)
    print(f"Writing {output_zarr}")
    output.to_zarr(output_zarr)
    print("complete")


if __name__ == "__main__":
    main()
