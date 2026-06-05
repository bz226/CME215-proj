#!/usr/bin/env python3
"""Plot Figure-5-style storm-centered wind and pressure panels.

The plot shows shaded 10 m wind speed with mean sea-level-pressure contours for
the analysis target and one or more GraphCast checkpoint forecasts.
"""

import argparse
import datetime as dt
import gc
import os
from pathlib import Path
import shutil

if os.environ.get("JAX_PLATFORMS", "") in ("", "cuda"):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import numpy as np

from compare_four_predictions import parse_extra_checkpoint
from hurricane_performance_check import pressure_to_hpa, read_track, track_lookup
from plot_prediction_error import add_derived, as_numpy_dataset, materialize_dataarray, parse_date, select_field


MODEL_LABELS = {
    "prefinetuned": "Pre-finetuned",
    "control": "Pre-finetuned",
    "mse5000": "MSE-5000",
    "amse": "AMSE-5000",
    "amse5000": "AMSE-5000",
    "amse25000": "AMSE-25000",
    "lamse": "LAMSE-0.1",
    "lamse0p1": "LAMSE-0.1",
    "lamse0p1_lmax32": "LAMSE-0.1-LMAX32",
    "lamse0p1_lmax32_5000": "LAMSE-0.1-LMAX32",
    "lamse0p5": "LAMSE-0.5",
    "lamse0p5_lmax127": "LAMSE-0.5-LMAX127",
    "lamse0p5_lmax127_5000": "LAMSE-0.5-LMAX127",
}


def default_params_dir():
    if os.environ.get("PARAMS_DIR"):
        return Path(os.environ["PARAMS_DIR"]).expanduser()
    if os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "graphcast-small-lamse" / "params"
    return Path("params")


def default_checkpoints():
    params_dir = default_params_dir()
    return {
        "prefinetuned": params_dir / "graphcast_small_lamse.000000.npz",
        "amse5000": params_dir / "graphcast_small_amse.005000.npz",
        "lamse0p5_lmax127": params_dir / "graphcast_small_lamse_lam0p5_lmax127.005000.npz",
    }


def model_label(name):
    return MODEL_LABELS.get(name, name.replace("_", " "))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-file", required=True, type=Path, help="HURDAT2 text file or CSV best-track file")
    parser.add_argument("--storm-name", default="IAN")
    parser.add_argument("--storm-year", type=int, default=2022)
    parser.add_argument("--apath", required=True, help="Analysis/verification database path")
    parser.add_argument("--norm-factors", default="stats")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Checkpoint panel to include after the analysis panel. May be repeated. "
            "Defaults to prefinetuned, AMSE-5000, and LAMSE-0.5-LMAX127."
        ),
    )
    parser.add_argument(
        "--init-date",
        type=parse_date,
        default=dt.datetime(2022, 9, 23, 12),
        help="Forecast initialization. Default gives a 5-day forecast valid at 28 Sep 2022 12 UTC.",
    )
    parser.add_argument("--lead-hours", type=int, default=120)
    parser.add_argument("--forecast-length", type=int, default=120)
    parser.add_argument("--analysis-label", default="ERA5 analysis")
    parser.add_argument("--lat-min", type=float, default=None)
    parser.add_argument("--lat-max", type=float, default=None)
    parser.add_argument("--lon-min", type=float, default=None)
    parser.add_argument("--lon-max", type=float, default=None)
    parser.add_argument("--radius-deg", type=float, default=3.0, help="Crop half-width if explicit bounds are omitted")
    parser.add_argument("--wind-vmin", type=float, default=18.0)
    parser.add_argument("--wind-vmax", type=float, default=40.0)
    parser.add_argument(
        "--pressure-contours",
        type=float,
        nargs="+",
        default=[940, 950, 960, 970, 980, 990, 1000, 1010, 1020],
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/hurricane_check/ian2022_figure5"))
    parser.add_argument("--out-file", default="ian2022_figure5.png")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-coastlines", action="store_true")
    return parser


def checkpoint_dict(args):
    if args.checkpoint:
        checkpoints = {}
        for spec in args.checkpoint:
            name, path = parse_extra_checkpoint(spec)
            checkpoints[name] = path
    else:
        checkpoints = default_checkpoints()
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    return checkpoints


def signed_lon(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def format_lat(value):
    suffix = "N" if value >= 0 else "S"
    return f"{abs(value):.0f}{suffix}"


def format_lon(value):
    suffix = "E" if value >= 0 else "W"
    return f"{abs(value):.0f}{suffix}"


def field_at_lead(ds, var, lead_hours):
    return materialize_dataarray(select_field(ds, var, None, lead_hours))


def selected_wind_pressure(ds, lead_hours):
    wind = field_at_lead(ds, "10m_wind_speed", lead_hours)
    pressure = field_at_lead(ds, "mean_sea_level_pressure", lead_hours)
    return wind, pressure


def subset_panel(wind_da, pressure_da, bounds):
    lat_min, lat_max, lon_min, lon_max = bounds
    lat = np.asarray(wind_da["lat"].data, dtype=np.float64)
    lon = signed_lon(wind_da["lon"].data)
    lat_mask = (lat >= min(lat_min, lat_max)) & (lat <= max(lat_min, lat_max))
    lon_mask = (lon >= min(lon_min, lon_max)) & (lon <= max(lon_min, lon_max))
    if not np.any(lat_mask):
        raise ValueError(f"No latitude points in crop [{lat_min}, {lat_max}]")
    if not np.any(lon_mask):
        raise ValueError(f"No longitude points in crop [{lon_min}, {lon_max}]")
    wind = np.asarray(wind_da, dtype=np.float64)[np.ix_(lat_mask, lon_mask)]
    pressure = pressure_to_hpa(np.asarray(pressure_da, dtype=np.float64)[np.ix_(lat_mask, lon_mask)])
    return lon[lon_mask], lat[lat_mask], wind, pressure


def choose_bounds(args, track_row):
    explicit = [args.lat_min, args.lat_max, args.lon_min, args.lon_max]
    if all(value is not None for value in explicit):
        return args.lat_min, args.lat_max, args.lon_min, args.lon_max
    if any(value is not None for value in explicit):
        raise ValueError("Set all of --lat-min --lat-max --lon-min --lon-max, or omit all of them")
    center_lat = float(track_row["lat"])
    center_lon = float(track_row["lon"])
    radius = float(args.radius_deg)
    return center_lat - radius, center_lat + radius, center_lon - radius, center_lon + radius


def run_checkpoint(checkpoint, predictor, inputs, target_template, forcings):
    import forecast.generate_model

    print(f"Loading checkpoint: {checkpoint}")
    _, _, params = forecast.generate_model.load_model(str(checkpoint))
    print(f"Running model: {checkpoint}")
    raw_prediction = predictor(inputs=inputs, targets=target_template, forcings=forcings, params=params)
    prediction = add_derived(as_numpy_dataset(raw_prediction))
    del raw_prediction, params
    gc.collect()
    return prediction


def plot_figure5(panels, bounds, args, valid_time, track_row):
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "plot_hurricane_figure5.py needs matplotlib for PNG output. "
            "Install it with: python3 -m pip install matplotlib==3.8.3"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    use_cartopy = False
    ccrs = None
    if not args.no_coastlines:
        try:
            import cartopy.crs as ccrs  # type: ignore

            use_cartopy = True
        except ImportError:
            print("cartopy is not available; plotting without coastlines")

    subplot_kw = {"projection": ccrs.PlateCarree()} if use_cartopy else {}
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(5.3, max(3.0 * len(panels), 3.5)),
        constrained_layout=True,
        subplot_kw=subplot_kw,
    )
    if len(panels) == 1:
        axes = [axes]

    mesh = None
    transform = ccrs.PlateCarree() if use_cartopy else None
    lat_min, lat_max, lon_min, lon_max = bounds
    contour_levels = sorted(set(float(level) for level in args.pressure_contours))
    for ax, panel in zip(axes, panels):
        lon, lat, wind, pressure = subset_panel(panel["wind"], panel["pressure"], bounds)
        plot_kwargs = {}
        contour_kwargs = {}
        if transform is not None:
            plot_kwargs["transform"] = transform
            contour_kwargs["transform"] = transform
        mesh = ax.pcolormesh(
            lon,
            lat,
            wind,
            shading="auto",
            cmap="PuBuGn",
            vmin=args.wind_vmin,
            vmax=args.wind_vmax,
            **plot_kwargs,
        )
        pressure_min = float(np.nanmin(pressure))
        pressure_max = float(np.nanmax(pressure))
        levels = [level for level in contour_levels if pressure_min <= level <= pressure_max]
        if levels:
            contour = ax.contour(lon, lat, pressure, levels=levels, colors="black", linewidths=0.9, **contour_kwargs)
            ax.clabel(contour, inline=True, fontsize=8, fmt="%.0f")
        if use_cartopy:
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=transform)
            ax.coastlines(resolution="50m", linewidth=0.7)
        else:
            ax.set_xlim(lon_min, lon_max)
            ax.set_ylim(lat_min, lat_max)
        ax.set_title(panel["label"], loc="left", fontsize=12)
        ax.grid(True, color="0.65", linewidth=0.5, alpha=0.8)
        ax.set_yticks(np.linspace(np.ceil(lat_min), np.floor(lat_max), 3))
        ax.set_yticklabels([format_lat(value) for value in ax.get_yticks()])
        ax.set_xticks(np.linspace(np.ceil(lon_min), np.floor(lon_max), 3))
        ax.set_xticklabels([format_lon(value) for value in ax.get_xticks()])

    if mesh is not None:
        colorbar = fig.colorbar(mesh, ax=axes, shrink=0.96, pad=0.04, extend="both")
        colorbar.set_label("10m wind speed (m/s)")

    if args.title:
        title = args.title
    else:
        title = (
            f"{args.storm_name.upper()} {valid_time:%Y-%m-%d %H:%M} UTC; "
            f"{args.lead_hours}h forecast panels"
        )
    fig.suptitle(title, fontsize=12)
    caption = (
        f"Best-track center: {track_row['lat']:.2f}, {track_row['lon']:.2f}; "
        "shading is 10 m wind speed, contours are MSLP in hPa."
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / args.out_file
    try:
        fig.savefig(path, dpi=args.dpi)
    except Exception:
        if use_cartopy:
            plt.close(fig)
            print("Saving with cartopy coastlines failed; retrying without coastlines")
            args.no_coastlines = True
            return plot_figure5(panels, bounds, args, valid_time, track_row)
        raise
    plt.close(fig)
    return path


def main():
    args = build_parser().parse_args()
    if args.forecast_length % 6 != 0:
        raise ValueError("--forecast-length must be a multiple of 6 hours")
    if args.lead_hours % 6 != 0:
        raise ValueError("--lead-hours must be a multiple of 6")
    if args.lead_hours > args.forecast_length:
        raise ValueError("--lead-hours must be <= --forecast-length")

    import dask
    import jax
    import pandas as pd
    import xarray as xr

    import forecast.generate_model
    import trainer.dataloader

    checkpoints = checkpoint_dict(args)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("devices", jax.devices())
    valid_time = args.init_date + dt.timedelta(hours=int(args.lead_hours))
    track = read_track(args.track_file, args.storm_name, args.storm_year)
    track = track[pd.to_datetime(track["time"]).dt.minute == 0].copy()
    track_row = track_lookup(track, valid_time)
    if track_row is None:
        raise ValueError(f"No best-track row for valid time {valid_time:%Y-%m-%d %H:%M}")
    bounds = choose_bounds(args, track_row)

    first_checkpoint = next(iter(checkpoints.values()))
    print(f"Loading reference checkpoint: {first_checkpoint}")
    model_config, task_config, _ = forecast.generate_model.load_model(str(first_checkpoint))
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
    print(f"Building forecast init={args.init_date:%Y-%m-%d %H:%M} steps={forecast_steps}")
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
    inputs, forcings, target_template = dask.compute(inputs, forcings, targets)
    target_fields = add_derived(as_numpy_dataset(target_template))
    target_wind, target_pressure = selected_wind_pressure(target_fields, args.lead_hours)

    panels = [{"label": args.analysis_label, "wind": target_wind, "pressure": target_pressure}]
    for name, checkpoint in checkpoints.items():
        prediction = run_checkpoint(checkpoint, predictor, inputs, target_template, forcings)
        wind, pressure = selected_wind_pressure(prediction, args.lead_hours)
        panels.append({"label": model_label(name), "wind": wind, "pressure": pressure})
        del prediction
        gc.collect()

    plot_path = plot_figure5(panels, bounds, args, valid_time, track_row)
    print(f"Wrote {plot_path}")
    print("complete")


if __name__ == "__main__":
    main()
