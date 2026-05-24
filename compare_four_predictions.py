#!/usr/bin/env python3
"""Compare ground truth, pre-finetuned, AMSE, and LAMSE predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import os
from pathlib import Path
import shutil

if os.environ.get("JAX_PLATFORMS", "") in ("", "cuda"):
    os.environ["JAX_PLATFORMS"] = "cuda,cpu"

import numpy as np

from plot_prediction_error import (
    add_derived,
    as_numpy_dataset,
    materialize_dataarray,
    parse_date,
    parse_field,
    safe_name,
    select_field,
)


MODEL_LABELS = {
    "prefinetuned": "Pre-finetuned",
    "amse": "AMSE-5000",
    "lamse": "LAMSE-5000",
}


def default_params_dir() -> Path:
    if os.environ.get("PARAMS_DIR"):
        return Path(os.environ["PARAMS_DIR"]).expanduser()
    if os.environ.get("SCRATCH"):
        return Path(os.environ["SCRATCH"]) / "graphcast-small-lamse" / "params"
    return Path("params")


def parse_args() -> argparse.Namespace:
    params_dir = default_params_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefinetuned-checkpoint", type=Path, default=params_dir / "graphcast_small_lamse.000000.npz")
    parser.add_argument("--amse-checkpoint", type=Path, default=params_dir / "graphcast_small_amse.005000.npz")
    parser.add_argument(
        "--lamse-checkpoint",
        type=Path,
        default=params_dir / "graphcast_small_lamse_lam0p1_lmax32.005000.npz",
    )
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
    parser.add_argument("--out-dir", type=Path, default=Path("runs/prediction_compare/20220101"))
    parser.add_argument("--output-zarr", type=Path, default=None)
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--spectral-csv", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--no-spectral",
        action="store_true",
        help="Skip AMSE-style spherical harmonic amplitude/coherence diagnostics.",
    )
    return parser.parse_args()


def weighted_stats(prediction, target, model: str, field: str, var: str, level: int | None, lead_hours: int) -> dict:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    err = pred - truth

    lat = np.asarray(target["lat"].data, dtype=np.float64)
    weights = np.cos(np.deg2rad(lat))
    weights = np.maximum(weights, 0.0)
    weights_2d = np.broadcast_to(weights[:, None], err.shape)
    valid = np.isfinite(err) & np.isfinite(pred) & np.isfinite(truth) & np.isfinite(weights_2d)
    if not np.any(valid):
        raise ValueError(f"No finite points for {field} {lead_hours}h {model}")
    norm = np.sum(weights_2d[valid])

    def wmean(values: np.ndarray) -> float:
        return float(np.sum(values[valid] * weights_2d[valid]) / norm)

    bias = wmean(err)
    pred_mean = wmean(pred)
    target_mean = wmean(truth)
    pred_anom = pred - pred_mean
    target_anom = truth - target_mean
    pred_var = wmean(pred_anom**2)
    target_var = wmean(target_anom**2)
    covariance = wmean(pred_anom * target_anom)
    correlation = covariance / np.sqrt(pred_var * target_var) if pred_var > 0 and target_var > 0 else np.nan

    return {
        "model": model,
        "field": field,
        "variable": var,
        "level": "" if level is None else level,
        "lead_hours": lead_hours,
        "bias": bias,
        "rmse": float(np.sqrt(wmean(err**2))),
        "mae": wmean(np.abs(err)),
        "error_std": float(np.sqrt(wmean((err - bias) ** 2))),
        "correlation": float(correlation),
    }


def plot_values(target, predictions: dict[str, object], title: str, path: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "compare_four_predictions.py needs matplotlib for PNG output. "
            "Install it in the active Sherlock venv with: python3 -m pip install matplotlib==3.8.3"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = target["lon"].data
    lat = target["lat"].data
    panels = [("Ground truth", np.asarray(target))]
    panels.extend((MODEL_LABELS[name], np.asarray(predictions[name])) for name in ("prefinetuned", "amse", "lamse"))

    combined = np.concatenate([values.ravel() for _, values in panels])
    vmin, vmax = np.nanpercentile(combined, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = None, None

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.5), constrained_layout=True)
    for ax, (panel_title, values) in zip(axes, panels):
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.84)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_errors(target, predictions: dict[str, object], title: str, path: Path) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "compare_four_predictions.py needs matplotlib for PNG output. "
            "Install it in the active Sherlock venv with: python3 -m pip install matplotlib==3.8.3"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lon = target["lon"].data
    lat = target["lat"].data
    errors = [(MODEL_LABELS[name], np.asarray(predictions[name]) - np.asarray(target)) for name in ("prefinetuned", "amse", "lamse")]

    all_errors = np.concatenate([values.ravel() for _, values in errors])
    err_abs = np.nanpercentile(np.abs(all_errors), 99)
    if not np.isfinite(err_abs) or err_abs == 0:
        err_abs = np.nanmax(np.abs(all_errors))
    if not np.isfinite(err_abs) or err_abs == 0:
        err_abs = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, (panel_title, values) in zip(axes, errors):
        mesh = ax.pcolormesh(lon, lat, values, shading="auto", cmap="RdBu_r", vmin=-err_abs, vmax=err_abs)
        ax.set_title(f"{panel_title} - truth")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        fig.colorbar(mesh, ax=ax, shrink=0.84)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def spectral_rows(target, predictions: dict[str, object], field: str, var: str, level: int | None, lead_hours: int) -> list[dict]:
    import jax.numpy as jnp

    import trainer.spectrum

    leg_coefs = trainer.spectrum.generate_spectral_coefs(np.asarray(target["lat"].data))
    target_values = jnp.asarray(np.asarray(target), dtype=jnp.float32)
    target_spec = trainer.spectrum.sht_eval(leg_coefs, target_values)
    target_psd = np.asarray(trainer.spectrum.power_spectral_density(target_spec), dtype=np.float64)

    rows = []
    eps = 1e-16
    for model, prediction in predictions.items():
        pred_values = jnp.asarray(np.asarray(prediction), dtype=jnp.float32)
        pred_spec = trainer.spectrum.sht_eval(leg_coefs, pred_values)
        pred_psd = np.asarray(trainer.spectrum.power_spectral_density(pred_spec), dtype=np.float64)
        cross = np.asarray(trainer.spectrum.cross_spectral_density(pred_spec, target_spec), dtype=np.float64)
        geo_mean = np.sqrt(np.maximum(pred_psd, 0.0) * np.maximum(target_psd, 0.0) + eps)
        coherence = np.clip(cross / geo_mean, -1.0, 1.0)
        amplitude_ratio = np.sqrt((pred_psd + eps) / (target_psd + eps))
        amplitude_error = (np.sqrt(np.maximum(pred_psd, 0.0)) - np.sqrt(np.maximum(target_psd, 0.0))) ** 2
        decorrelation_error = 2.0 * np.maximum(pred_psd, target_psd) * (1.0 - coherence)
        for wavenumber in range(target_psd.size):
            rows.append(
                {
                    "model": model,
                    "field": field,
                    "variable": var,
                    "level": "" if level is None else level,
                    "lead_hours": lead_hours,
                    "total_wavenumber": wavenumber,
                    "target_psd": target_psd[wavenumber],
                    "prediction_psd": pred_psd[wavenumber],
                    "amplitude_ratio": amplitude_ratio[wavenumber],
                    "coherence": coherence[wavenumber],
                    "amplitude_error": amplitude_error[wavenumber],
                    "decorrelation_error": decorrelation_error[wavenumber],
                }
            )
    return rows


def plot_spectra(rows: list[dict], field: str, lead_hours: int, path: Path) -> None:
    if not rows:
        return
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "compare_four_predictions.py needs matplotlib for PNG output. "
            "Install it in the active Sherlock venv with: python3 -m pip install matplotlib==3.8.3"
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), constrained_layout=True, sharex=True)
    row_models = ("prefinetuned", "amse", "lamse")
    target_wavenumber = np.asarray([r["total_wavenumber"] for r in rows if r["model"] == row_models[0]])
    target_psd = np.asarray([r["target_psd"] for r in rows if r["model"] == row_models[0]])
    axes[0].semilogy(target_wavenumber, target_psd, color="black", linewidth=2, label="Ground truth")
    for model in row_models:
        model_rows = [r for r in rows if r["model"] == model]
        wavenumber = np.asarray([r["total_wavenumber"] for r in model_rows])
        axes[0].semilogy(wavenumber, [r["prediction_psd"] for r in model_rows], label=MODEL_LABELS[model])
        axes[1].plot(wavenumber, [r["amplitude_ratio"] for r in model_rows], label=MODEL_LABELS[model])
        axes[2].plot(wavenumber, [r["coherence"] for r in model_rows], label=MODEL_LABELS[model])
    axes[0].set_ylabel("PSD")
    axes[1].set_ylabel("Amplitude ratio")
    axes[2].set_ylabel("Coherence")
    axes[2].set_xlabel("total wavenumber")
    axes[1].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    axes[2].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle(f"{field} lead={lead_hours}h AMSE-style spectral diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_prediction(checkpoint: Path, predictor, inputs, targets, forcings, field_specs: list[tuple], lead_hours: list[int]):
    import forecast.generate_model

    print(f"Loading checkpoint: {checkpoint}")
    _, _, params = forecast.generate_model.load_model(str(checkpoint))
    print(f"Running model: {checkpoint}")
    raw_prediction = predictor(inputs=inputs, targets=targets, forcings=forcings, params=params)
    prediction = add_derived(as_numpy_dataset(raw_prediction))
    selected = {}
    for var, level, label in field_specs:
        if var not in prediction:
            print(f"Skipping {label}: {var} not in prediction")
            continue
        for lead in lead_hours:
            selected[(label, lead)] = materialize_dataarray(select_field(prediction, var, level, lead))
    del raw_prediction, prediction, params
    gc.collect()
    return selected


def main() -> None:
    args = parse_args()
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_zarr = args.output_zarr or args.out_dir / "comparison_fields.zarr"
    metrics_csv = args.metrics_csv or args.out_dir / "comparison_metrics.csv"
    spectral_csv = args.spectral_csv or args.out_dir / "spectral_metrics.csv"

    checkpoints = {
        "prefinetuned": args.prefinetuned_checkpoint,
        "amse": args.amse_checkpoint,
        "lamse": args.lamse_checkpoint,
    }
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} checkpoint: {path}")

    print("devices", jax.devices())
    print(f"Loading reference checkpoint: {args.prefinetuned_checkpoint}")
    model_config, task_config, _ = forecast.generate_model.load_model(str(args.prefinetuned_checkpoint))
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
    inputs, forcings, target_template = dask.compute(inputs, forcings, targets)
    targets = add_derived(as_numpy_dataset(target_template))

    field_specs = [parse_field(field) for field in args.fields]
    target_fields = {}
    for var, level, label in field_specs:
        if var not in targets:
            print(f"Skipping {label}: {var} not in targets")
            continue
        for lead in args.lead_hours:
            target_fields[(label, lead)] = materialize_dataarray(select_field(targets, var, level, lead))

    predictions = {
        name: run_prediction(path, predictor, inputs, target_template, forcings, field_specs, args.lead_hours)
        for name, path in checkpoints.items()
    }

    data_vars = {}
    metric_rows = []
    spectral_metric_rows = []
    attrs = {
        "init_date": args.init_date.isoformat(),
        "forecast_length_hours": args.forecast_length,
        "prefinetuned_checkpoint": str(args.prefinetuned_checkpoint),
        "amse_checkpoint": str(args.amse_checkpoint),
        "lamse_checkpoint": str(args.lamse_checkpoint),
    }

    for var, level, label in field_specs:
        for lead in args.lead_hours:
            key = (label, lead)
            if key not in target_fields:
                continue
            if any(key not in predictions[model] for model in MODEL_LABELS):
                print(f"Skipping plots for {label} {lead}h because at least one model is missing it")
                continue
            suffix = safe_name(f"{label}_{lead:03d}h")
            target_field = target_fields[key]
            model_fields = {model: predictions[model][key] for model in MODEL_LABELS}
            data_vars[f"groundtruth_{suffix}"] = target_field
            for model, da in model_fields.items():
                data_vars[f"{model}_{suffix}"] = da
                data_vars[f"{model}_error_{suffix}"] = materialize_dataarray(da - target_field)
                metric_rows.append(weighted_stats(da, target_field, model, label, var, level, lead))
            if not args.no_plots:
                title = f"{label} lead={lead}h init={args.init_date:%Y-%m-%d %H:%M}"
                plot_values(target_field, model_fields, title, args.out_dir / f"{suffix}_values.png")
                plot_errors(target_field, model_fields, title, args.out_dir / f"{suffix}_errors.png")
                print(f"Wrote plots for {suffix}")
            if not args.no_spectral:
                try:
                    rows = spectral_rows(target_field, model_fields, label, var, level, lead)
                except Exception as exc:
                    print(f"Skipping spectral diagnostics for {suffix}: {exc}")
                    args.no_spectral = True
                    rows = []
                spectral_metric_rows.extend(rows)
                if rows and not args.no_plots:
                    plot_spectra(rows, label, lead, args.out_dir / f"{suffix}_spectra.png")

    output = xr.Dataset(data_vars, attrs=attrs)
    if output_zarr.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_zarr} exists; pass --overwrite")
        shutil.rmtree(output_zarr)
    print(f"Writing {output_zarr}")
    output.to_zarr(output_zarr)

    print(f"Writing {metrics_csv}")
    pd.DataFrame(metric_rows).to_csv(metrics_csv, index=False)
    if spectral_metric_rows:
        print(f"Writing {spectral_csv}")
        pd.DataFrame(spectral_metric_rows).to_csv(spectral_csv, index=False)
    print("complete")


if __name__ == "__main__":
    main()
