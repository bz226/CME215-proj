#!/usr/bin/env python3
"""Paper-style tropical cyclone intensity and track diagnostics.

The script runs one or more GraphCast checkpoints over a date range and scores
storm-centered maximum 10 m wind speed, minimum sea-level pressure, and center
position against a supplied best-track file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
from pathlib import Path
import re
import shutil

if os.environ.get("JAX_PLATFORMS", "") in ("", "cuda"):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import numpy as np

from compare_four_predictions import label_for_model, parse_extra_checkpoint
from plot_prediction_error import add_derived, as_numpy_dataset, materialize_dataarray, parse_date, select_field


KT_TO_MPS = 0.514444
EARTH_RADIUS_KM = 6371.0


def default_params_dir() -> Path:
    if os.environ.get("PARAMS_DIR"):
        return Path(os.environ["PARAMS_DIR"]).expanduser()
    if os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "graphcast-small-lamse" / "params"
    return Path("params")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-file", required=True, type=Path, help="HURDAT2 text file or CSV best-track file")
    parser.add_argument("--storm-name", default="IAN", help="Storm name in the track file")
    parser.add_argument("--storm-year", type=int, default=2022)
    parser.add_argument("--apath", required=True, help="Analysis/verification database path")
    parser.add_argument("--norm-factors", default="stats")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Checkpoint to score. May be repeated. Defaults to prefinetuned, AMSE-5000, and LAMSE-0.1-5000.",
    )
    parser.add_argument("--init-start", type=parse_date, required=True)
    parser.add_argument("--init-end", type=parse_date, required=True)
    parser.add_argument("--init-interval-hours", type=int, default=24)
    parser.add_argument("--forecast-length", type=int, default=240)
    parser.add_argument("--lead-hours", type=int, nargs="+", default=list(range(6, 241, 6)))
    parser.add_argument(
        "--truth-source",
        choices=("best_track", "analysis"),
        default="best_track",
        help=(
            "best_track compares model intensity to the supplied track wind/pressure. "
            "analysis compares model intensity to ERA5/analysis fields in the same search window."
        ),
    )
    parser.add_argument("--min-best-track-wind-kt", type=float, default=34.0)
    parser.add_argument("--search-radius-deg", type=float, default=6.0)
    parser.add_argument("--wind-radius-deg", type=float, default=3.0)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/hurricane_check/ian2022"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser


def default_checkpoints() -> dict[str, Path]:
    params_dir = default_params_dir()
    return {
        "prefinetuned": params_dir / "graphcast_small_lamse.000000.npz",
        "amse": params_dir / "graphcast_small_amse.005000.npz",
        "lamse": params_dir / "graphcast_small_lamse_lam0p1_lmax32.005000.npz",
    }


def checkpoint_dict(args: argparse.Namespace) -> dict[str, Path]:
    if not args.checkpoint:
        checkpoints = default_checkpoints()
    else:
        checkpoints = {}
    for spec in args.checkpoint:
        name, path = parse_extra_checkpoint(spec)
        if name in checkpoints:
            raise ValueError(f"Duplicate checkpoint name: {name}")
        checkpoints[name] = path
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")
    return checkpoints


def parse_latlon_token(value: str) -> float:
    token = value.strip().upper()
    if token.endswith("N"):
        return float(token[:-1])
    if token.endswith("S"):
        return -float(token[:-1])
    if token.endswith("E"):
        return float(token[:-1])
    if token.endswith("W"):
        return -float(token[:-1])
    return float(token)


def read_hurdat2(path: Path, storm_name: str, storm_year: int):
    import pandas as pd

    rows = []
    active = False
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            parts = [part.strip() for part in line.split(",")]
            if not parts or not parts[0]:
                continue
            if re.fullmatch(r"[A-Z]{2}\d{6}", parts[0]):
                basin_id = parts[0]
                name = parts[1].upper() if len(parts) > 1 else ""
                year = int(basin_id[-4:])
                active = name == storm_name.upper() and year == storm_year
                continue
            if not active:
                continue
            if len(parts) < 8:
                continue
            when = dt.datetime.strptime(parts[0] + parts[1].zfill(4), "%Y%m%d%H%M")
            vmax_kt = np.nan if parts[6] in ("", "-999") else float(parts[6])
            mslp_hpa = np.nan if parts[7] in ("", "-999") else float(parts[7])
            rows.append(
                {
                    "time": when,
                    "lat": parse_latlon_token(parts[4]),
                    "lon": parse_latlon_token(parts[5]),
                    "vmax_kt": vmax_kt,
                    "vmax_mps": vmax_kt * KT_TO_MPS if np.isfinite(vmax_kt) else np.nan,
                    "mslp_hpa": mslp_hpa,
                    "status": parts[3],
                }
            )
    if not rows:
        raise ValueError(f"No HURDAT2 rows found for {storm_name} {storm_year} in {path}")
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def read_track_csv(path: Path, storm_name: str, storm_year: int):
    import pandas as pd

    df = pd.read_csv(path)
    columns = {col.lower(): col for col in df.columns}
    if "storm_name" in columns:
        df = df[df[columns["storm_name"]].str.upper() == storm_name.upper()]
    if "name" in columns:
        df = df[df[columns["name"]].str.upper() == storm_name.upper()]
    if "year" in columns:
        df = df[df[columns["year"]].astype(int) == storm_year]

    time_col = next((columns[name] for name in ("time", "datetime", "date_time", "valid_time") if name in columns), None)
    lat_col = next((columns[name] for name in ("lat", "latitude") if name in columns), None)
    lon_col = next((columns[name] for name in ("lon", "longitude") if name in columns), None)
    wind_col = next((columns[name] for name in ("vmax_kt", "max_wind_kt", "wind_kt", "vmax") if name in columns), None)
    wind_mps_col = next((columns[name] for name in ("vmax_mps", "max_wind_mps", "wind_mps") if name in columns), None)
    pressure_col = next((columns[name] for name in ("mslp_hpa", "min_pressure_hpa", "pressure_hpa", "mslp") if name in columns), None)
    if time_col is None or lat_col is None or lon_col is None:
        raise ValueError("CSV track file must include time/datetime, lat/latitude, and lon/longitude columns")
    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[time_col]).dt.to_pydatetime(),
            "lat": df[lat_col].astype(float),
            "lon": df[lon_col].astype(float),
        }
    )
    if wind_mps_col is not None:
        out["vmax_mps"] = df[wind_mps_col].astype(float)
        out["vmax_kt"] = out["vmax_mps"] / KT_TO_MPS
    elif wind_col is not None:
        out["vmax_kt"] = df[wind_col].astype(float)
        out["vmax_mps"] = out["vmax_kt"] * KT_TO_MPS
    else:
        out["vmax_kt"] = np.nan
        out["vmax_mps"] = np.nan
    out["mslp_hpa"] = df[pressure_col].astype(float) if pressure_col is not None else np.nan
    out["status"] = ""
    return out.sort_values("time").reset_index(drop=True)


def read_track(path: Path, storm_name: str, storm_year: int):
    if path.suffix.lower() == ".csv":
        return read_track_csv(path, storm_name, storm_year)
    return read_hurdat2(path, storm_name, storm_year)


def init_dates(start: dt.datetime, end: dt.datetime, interval_hours: int) -> list[dt.datetime]:
    if interval_hours <= 0:
        raise ValueError("--init-interval-hours must be positive")
    dates = []
    current = start
    step = dt.timedelta(hours=interval_hours)
    while current <= end:
        dates.append(current)
        current += step
    return dates


def normalize_lon_to_360(lon: float | np.ndarray) -> float | np.ndarray:
    return np.mod(lon, 360.0)


def lon_delta_deg(lon: np.ndarray, center_lon: float) -> np.ndarray:
    lon360 = normalize_lon_to_360(lon)
    center360 = normalize_lon_to_360(center_lon)
    return ((lon360 - center360 + 180.0) % 360.0) - 180.0


def great_circle_km(lat1: float | np.ndarray, lon1: float | np.ndarray, lat2: float, lon2: float) -> np.ndarray:
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dphi = np.deg2rad(np.asarray(lat2) - np.asarray(lat1))
    dlambda = np.deg2rad(lon_delta_deg(np.asarray(lon2), np.asarray(lon1)))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(1.0 - a, 0.0)))


def pressure_to_hpa(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanmedian(finite) > 2000.0:
        return arr / 100.0
    return arr


def field_at_lead(ds, var: str, lead_hours: int):
    return materialize_dataarray(select_field(ds, var, None, lead_hours))


def storm_metrics_from_fields(
    wind_da,
    mslp_da,
    search_lat: float,
    search_lon: float,
    search_radius_deg: float,
    wind_radius_deg: float,
) -> dict[str, float]:
    lat = np.asarray(mslp_da["lat"].data, dtype=np.float64)
    lon = np.asarray(mslp_da["lon"].data, dtype=np.float64)
    pressure = pressure_to_hpa(np.asarray(mslp_da, dtype=np.float64))
    wind = np.asarray(wind_da, dtype=np.float64)

    lat2d = np.broadcast_to(lat[:, None], pressure.shape)
    lon2d = np.broadcast_to(lon[None, :], pressure.shape)
    box = (
        (np.abs(lat2d - search_lat) <= search_radius_deg)
        & (np.abs(lon_delta_deg(lon2d, search_lon)) <= search_radius_deg)
        & np.isfinite(pressure)
    )
    if not np.any(box):
        raise ValueError("No finite sea-level-pressure points in storm search window")
    masked_pressure = np.where(box, pressure, np.inf)
    center_flat = int(np.argmin(masked_pressure))
    center_i, center_j = np.unravel_index(center_flat, pressure.shape)
    center_lat = float(lat[center_i])
    center_lon = float(lon[center_j])
    min_mslp_hpa = float(pressure[center_i, center_j])

    distance_deg = great_circle_km(lat2d, lon2d, center_lat, center_lon) / 111.195
    wind_mask = (distance_deg <= wind_radius_deg) & np.isfinite(wind)
    if not np.any(wind_mask):
        max_wind_mps = np.nan
    else:
        max_wind_mps = float(np.nanmax(np.where(wind_mask, wind, np.nan)))

    return {
        "center_lat": center_lat,
        "center_lon": center_lon if center_lon <= 180.0 else center_lon - 360.0,
        "min_mslp_hpa": min_mslp_hpa,
        "max_wind_mps": max_wind_mps,
    }


def track_lookup(track, valid_time: dt.datetime):
    import pandas as pd

    matches = track[pd.to_datetime(track["time"]) == pd.Timestamp(valid_time)]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def run_checkpoint(checkpoint: Path, predictor, inputs, targets, forcings):
    import forecast.generate_model

    print(f"Loading checkpoint: {checkpoint}")
    _, _, params = forecast.generate_model.load_model(str(checkpoint))
    print(f"Running model: {checkpoint}")
    raw_prediction = predictor(inputs=inputs, targets=targets, forcings=forcings, params=params)
    prediction = add_derived(as_numpy_dataset(raw_prediction))
    del raw_prediction, params
    gc.collect()
    return prediction


def summarize(rows):
    import pandas as pd

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    grouped = df.groupby(["model", "model_label", "lead_hours"], as_index=False)
    return grouped.agg(
        count=("wind_error_mps", "count"),
        mean_wind_error_mps=("wind_error_mps", "mean"),
        mean_pressure_error_hpa=("pressure_error_hpa", "mean"),
        mean_abs_position_error_km=("position_error_km", "mean"),
        mean_abs_wind_error_mps=("wind_error_mps", lambda x: np.nanmean(np.abs(x))),
        mean_abs_pressure_error_hpa=("pressure_error_hpa", lambda x: np.nanmean(np.abs(x))),
    )


def plot_summary(summary, path: Path, title: str) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "hurricane_performance_check.py needs matplotlib for PNG output. "
            "Install it with: python3 -m pip install matplotlib==3.8.3"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    models = list(dict.fromkeys(summary["model"]))
    for model in models:
        part = summary[summary["model"] == model].sort_values("lead_hours")
        label = part["model_label"].iloc[0]
        x_days = part["lead_hours"] / 24.0
        axes[0].plot(x_days, part["mean_wind_error_mps"], marker="o", linewidth=1.6, markersize=3.5, label=label)
        axes[1].plot(x_days, part["mean_pressure_error_hpa"], marker="o", linewidth=1.6, markersize=3.5, label=label)
        axes[2].plot(
            x_days,
            part["mean_abs_position_error_km"],
            marker="o",
            linewidth=1.6,
            markersize=3.5,
            label=label,
        )
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Wind speed error [m/s]")
    axes[1].set_ylabel("Pressure error [hPa]")
    axes[2].set_ylabel("Position error [km]")
    axes[2].set_xlabel("Forecast lead time [days]")
    axes[0].set_title("Mean maximum 10 m wind speed error")
    axes[1].set_title("Mean minimum central pressure error")
    axes[2].set_title("Mean absolute position error")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.forecast_length % 6 != 0:
        raise ValueError("--forecast-length must be a multiple of 6 hours")
    if any(lead % 6 != 0 for lead in args.lead_hours):
        raise ValueError("--lead-hours values must be multiples of 6")
    if max(args.lead_hours) > args.forecast_length:
        raise ValueError("All --lead-hours must be <= --forecast-length")

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
    track = read_track(args.track_file, args.storm_name, args.storm_year)
    track = track[pd.to_datetime(track["time"]).dt.minute == 0].copy()
    if args.min_best_track_wind_kt is not None:
        track = track[track["vmax_kt"].fillna(0.0) >= args.min_best_track_wind_kt].copy()
    if track.empty:
        raise ValueError("No best-track rows remain after filtering")
    track_path = args.out_dir / "track_used.csv"
    track.to_csv(track_path, index=False)
    print(f"Wrote {track_path}")

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
    rows = []
    dates = init_dates(args.init_start, args.init_end, args.init_interval_hours)
    for init_date in dates:
        print(f"Building forecast init={init_date:%Y-%m-%d %H:%M} steps={forecast_steps}")
        inputs, forcings, targets = trainer.dataloader.build_forecast(
            init_date,
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
        predictions = {
            name: run_checkpoint(path, predictor, inputs, target_template, forcings)
            for name, path in checkpoints.items()
        }

        for lead in args.lead_hours:
            valid_time = init_date + dt.timedelta(hours=int(lead))
            truth = track_lookup(track, valid_time)
            if truth is None:
                continue
            try:
                target_wind = field_at_lead(target_fields, "10m_wind_speed", lead)
                target_mslp = field_at_lead(target_fields, "mean_sea_level_pressure", lead)
            except Exception as exc:
                print(f"Skipping truth fields for init={init_date} lead={lead}: {exc}")
                continue

            if args.truth_source == "analysis":
                truth_metrics = storm_metrics_from_fields(
                    target_wind,
                    target_mslp,
                    float(truth["lat"]),
                    float(truth["lon"]),
                    args.search_radius_deg,
                    args.wind_radius_deg,
                )
                truth_wind_mps = truth_metrics["max_wind_mps"]
                truth_pressure_hpa = truth_metrics["min_mslp_hpa"]
                truth_center_lat = truth_metrics["center_lat"]
                truth_center_lon = truth_metrics["center_lon"]
            else:
                truth_wind_mps = float(truth["vmax_mps"])
                truth_pressure_hpa = float(truth["mslp_hpa"])
                truth_center_lat = float(truth["lat"])
                truth_center_lon = float(truth["lon"])

            for model, prediction in predictions.items():
                try:
                    pred_wind = field_at_lead(prediction, "10m_wind_speed", lead)
                    pred_mslp = field_at_lead(prediction, "mean_sea_level_pressure", lead)
                    metrics = storm_metrics_from_fields(
                        pred_wind,
                        pred_mslp,
                        truth_center_lat,
                        truth_center_lon,
                        args.search_radius_deg,
                        args.wind_radius_deg,
                    )
                except Exception as exc:
                    print(f"Skipping {model} init={init_date} lead={lead}: {exc}")
                    continue
                rows.append(
                    {
                        "model": model,
                        "model_label": label_for_model(model),
                        "init_time": init_date.isoformat(),
                        "valid_time": valid_time.isoformat(),
                        "lead_hours": lead,
                        "truth_source": args.truth_source,
                        "truth_lat": truth_center_lat,
                        "truth_lon": truth_center_lon,
                        "truth_vmax_mps": truth_wind_mps,
                        "truth_mslp_hpa": truth_pressure_hpa,
                        "best_track_status": truth.get("status", ""),
                        "pred_lat": metrics["center_lat"],
                        "pred_lon": metrics["center_lon"],
                        "pred_max_wind_mps": metrics["max_wind_mps"],
                        "pred_min_mslp_hpa": metrics["min_mslp_hpa"],
                        "wind_error_mps": metrics["max_wind_mps"] - truth_wind_mps,
                        "pressure_error_hpa": metrics["min_mslp_hpa"] - truth_pressure_hpa,
                        "position_error_km": float(
                            great_circle_km(metrics["center_lat"], metrics["center_lon"], truth_center_lat, truth_center_lon)
                        ),
                    }
                )

        del target_fields, target_template, inputs, forcings, predictions
        gc.collect()

    metrics_path = args.out_dir / "hurricane_metrics.csv"
    summary_path = args.out_dir / "hurricane_mean_errors.csv"
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    print(f"Wrote {metrics_path}")
    summary = summarize(rows)
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    if not args.no_plots and not summary.empty:
        plot_path = args.out_dir / "hurricane_error_summary.png"
        plot_summary(
            summary,
            plot_path,
            f"{args.storm_name.upper()} {args.storm_year} paper-style hurricane diagnostics",
        )
        print(f"Wrote {plot_path}")
    print("complete")


if __name__ == "__main__":
    main()
