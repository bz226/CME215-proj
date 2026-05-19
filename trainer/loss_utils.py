# Copyright 2024 Crown in Right of Canada
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

from collections import namedtuple
import argparse
import typing


LossParameter = namedtuple("LossParameter", ("name", "varname", "dtype", "default", "help"))

Parameters = (
    LossParameter("--error-weights", "error_weight_file", str, None, "File containing non-default variable and level weights"),
    LossParameter("--wind-speed", "wind_speed", bool, False, "Add wind speed variable to loss function"),
    LossParameter("--time-bias", "time_bias", bool, False, "Add time-averaged term to loss function"),
    LossParameter("--mean-bias", "mean_bias", bool, False, "Add global mean bias term to loss function"),
    LossParameter("--spectral-amse", "spectral_amse", bool, False, "Compute loss in spectral AMSE space"),
    LossParameter("--lamse", "lamse", bool, False, "Compute hybrid AMSE/LAMSE loss"),
    LossParameter("--lamse-lambda", "lamse_lambda", float, 0.1, "Hybrid LAMSE weight"),
    LossParameter("--lamse-lmax", "lamse_lmax", int, None, "Inclusive maximum degree for LAMSE"),
    LossParameter("--mae", "mae", bool, False, "Compute loss with mean absolute error rather than MSE"),
)

config_dict = {p.varname: p.default for p in Parameters}


if typing.TYPE_CHECKING:
    import graphcast.graphcast
    import xarray


def add_error_args(arggroup: argparse._ArgumentGroup | argparse.ArgumentParser):
    """Add loss-related command-line arguments."""

    for p in Parameters:
        if p.dtype is bool:
            if p.default is False:
                arggroup.add_argument(p.name, action="store_true", dest=p.varname, help=p.help)
            elif p.default is True:
                arggroup.add_argument(p.name, action="store_false", dest=p.varname, help=p.help)
            else:
                raise ValueError(f"Invalid default {p.default} for boolean parameter {p.name}")
        else:
            arggroup.add_argument(p.name, type=p.dtype, dest=p.varname, default=p.default, help=p.help)


def parse_arguments(args: argparse.Namespace):
    """Populate module-level loss configuration from parsed arguments."""

    parsed = vars(args)
    for p in Parameters:
        if p.varname in parsed:
            config_dict[p.varname] = parsed[p.varname]


def normalize(ds, base, mean_by_level, stddev_by_level, diffs_stddev_by_level):
    import xarray as xr

    delta = xr.Dataset()
    for var in ds.data_vars:
        if var in base.data_vars:
            delta[var] = (ds[var] - base[var]) / diffs_stddev_by_level[var]
        else:
            delta[var] = (ds[var] - mean_by_level[var]) / stddev_by_level[var]
    return delta


def derived_variables(ds, compute_wind_speed):
    ds_out = ds.copy()
    if compute_wind_speed:
        ds_out["wind_speed"] = (
            ds_out["u_component_of_wind"] ** 2 + ds_out["v_component_of_wind"] ** 2
        ) ** 0.5
        ds_out["10m_wind_speed"] = (
            ds_out["10m_u_component_of_wind"] ** 2
            + ds_out["10m_v_component_of_wind"] ** 2
        ) ** 0.5
    return ds_out


def losses_over_time(forecast, targets, analysis, per_variable_weights, norm_fn):
    from graphcast import xarray_jax
    from graphcast import losses

    norm_fc = norm_fn(forecast, analysis)
    norm_tg = norm_fn(targets, analysis)
    persist = 0 * norm_fc.isel(time=0)
    forecast_losses = [
        xarray_jax.unwrap_data(
            losses.weighted_mse_per_level(
                norm_fc.isel(time=lead), norm_tg.isel(time=lead), per_variable_weights
            )[0]
        )
        for lead in range(forecast.time.size)
    ]
    persist_losses = [
        xarray_jax.unwrap_data(
            losses.weighted_mse_per_level(persist, norm_tg.isel(time=lead), per_variable_weights)[0]
        )
        for lead in range(forecast.time.size)
    ]
    return forecast_losses, persist_losses


def make_loss_new(
    model_config: "graphcast.graphcast.ModelConfig",
    task_config: "graphcast.graphcast.TaskConfig",
    diffs_stddev_by_level: typing.Optional["xarray.Dataset"],
    mean_by_level: typing.Optional["xarray.Dataset"],
    stddev_by_level: typing.Optional["xarray.Dataset"],
    silent: bool = False,
):
    """Build the configured differentiable custom loss function."""

    if (
        diffs_stddev_by_level is None
        and mean_by_level is None
        and stddev_by_level is None
        and all(config_dict[p.varname] == p.default for p in Parameters)
    ):
        if not silent:
            print("Using built-in (default) loss function")
        return None

    import numpy as np
    import xarray as xr

    import forecast.generate_model
    import graphcast.losses

    model_latitude, model_longitude = forecast.generate_model.get_model_coords(model_config)
    latitude_weights = graphcast.losses.normalized_latitude_weights(model_latitude.rename(latitude="lat"))
    latitude_weights = latitude_weights / latitude_weights.mean()

    input_variables = list(task_config["input_variables"])
    target_variables = list(task_config["target_variables"])
    levels = np.array(task_config["pressure_levels"])

    if config_dict["error_weight_file"] is not None:
        errfile_path = config_dict["error_weight_file"]
        if not silent:
            print(f"Loading level and variable weights from {errfile_path}")
        with open(errfile_path, "rb") as errfile:
            import pickle

            per_variable_weights, level_weights = pickle.load(errfile)
        assert np.all(np.isin(levels, level_weights.level.data))
        level_weights = level_weights.sel(level=levels)
        level_weights = level_weights / level_weights.sum()
    else:
        if not silent:
            print("Using default level and variable weights")
        per_variable_weights = {
            "2m_temperature": 1.0,
            "10m_u_component_of_wind": 0.1,
            "10m_v_component_of_wind": 0.1,
            "mean_sea_level_pressure": 0.1,
            "total_precipitation_6hr": 0.1,
        }
        level_weights = xr.DataArray(levels, dims=("level"), coords={"level": levels})
        level_weights = level_weights / level_weights.sum()

    if diffs_stddev_by_level is None:
        diffs_stddev_path = "stats/diffs_stddev_by_level.nc"
        if not silent:
            print(f"Loading 6h difference standard deviation from {diffs_stddev_path}")
        diffs_stddev_by_level = xr.load_dataset(diffs_stddev_path).compute()

    if stddev_by_level is None:
        stddev_path = "stats/stddev_by_level.nc"
        if not silent:
            print(f"Loading total standard deviation from {stddev_path}")
        stddev_by_level = xr.load_dataset(stddev_path).compute()

    norms_by_level = xr.merge(
        [
            diffs_stddev_by_level[v] if v in input_variables else stddev_by_level[v]
            for v in target_variables
        ]
    )

    compute_wind_speed = config_dict["wind_speed"]
    time_bias = config_dict["time_bias"]
    mean_bias = config_dict["mean_bias"]
    spectral_amse = config_dict["spectral_amse"]
    lamse = config_dict["lamse"]
    mae_error = config_dict["mae"]

    if lamse and spectral_amse:
        raise ValueError("Use --lamse or --spectral-amse, not both.")
    if (spectral_amse or lamse) and mae_error:
        raise ValueError("Cannot use AMSE/LAMSE and MAE error calculations simultaneously.")

    if compute_wind_speed:
        if not silent:
            print("Computing loss function with wind speed added")
        norms_by_level = derived_variables(norms_by_level, compute_wind_speed)

    if time_bias and not silent:
        print("Computing loss function with additional time bias term")
    if mean_bias and not silent:
        print("Computing loss function with additional global mean loss term")

    if spectral_amse or lamse:
        if time_bias or mean_bias:
            raise ValueError("Spectral AMSE/LAMSE is not compatible with time-bias or mean-bias terms")

        import trainer.spectrum

        if not silent:
            print("Generating AMSE spectral coefficients")
        leg_coef = trainer.spectrum.generate_spectral_coefs(model_latitude)
        if not silent:
            print("... done")
        sht_forward = lambda f: trainer.spectrum.sht_eval(leg_coef, f)

        lamse_precompute = None
        lamse_config = None
        if lamse:
            from config import LAMSEConfig
            import lamse_loss

            lamse_config = LAMSEConfig(
                lambda_lamse=config_dict["lamse_lambda"],
                L=config_dict["lamse_lmax"],
            )
            if not silent:
                print("Building LAMSE HEALPix/needlet geometry")
            lamse_precompute = lamse_loss.build_lamse_precompute(
                model_latitude.data,
                model_longitude.data,
                lamse_config,
            )

        def my_loss(prediction, targets):
            prediction = derived_variables(prediction, compute_wind_speed)
            targets = derived_variables(targets, compute_wind_speed)
            first_var = next(iter(prediction.data_vars))
            targets = targets.transpose(*prediction[first_var].dims)
            amse_loss, amse_diag = spectral_adj_loss(
                prediction,
                targets,
                norms_by_level,
                sht_forward,
                level_weights,
                per_variable_weights,
            )
            if not lamse or lamse_config.lambda_lamse == 0.0:
                return amse_loss, amse_diag

            import lamse_loss as lamse_module

            local_loss, local_diag = lamse_module.lamse_dataset_loss(
                prediction,
                targets,
                norms_by_level,
                lamse_precompute,
                level_weights,
                per_variable_weights,
            )
            lam = lamse_config.lambda_lamse
            hybrid = (1.0 - lam) * amse_loss + lam * local_loss
            hybrid_diag = (1.0 - lam) * amse_diag + lam * local_diag
            hybrid_diag["amse_total"] = amse_loss
            hybrid_diag["lamse_total"] = local_loss
            hybrid_diag["hybrid_total"] = hybrid
            return hybrid, hybrid_diag

    elif mae_error:
        if time_bias or mean_bias:
            raise ValueError("MAE loss is not compatible with time-bias or mean-bias terms")

        def my_loss(prediction, targets):
            prediction = derived_variables(prediction, compute_wind_speed)
            targets = derived_variables(targets, compute_wind_speed)
            return mae_loss(
                prediction,
                targets,
                per_variable_weights,
                level_weights,
                norms_by_level,
                latitude_weights,
            )

    else:

        def my_loss(prediction, targets):
            prediction = derived_variables(prediction, compute_wind_speed)
            targets = derived_variables(targets, compute_wind_speed)
            return spatial_loss(
                prediction,
                targets,
                per_variable_weights,
                level_weights,
                norms_by_level,
                latitude_weights,
                time_bias,
                mean_bias,
            )

    return my_loss


def spectral_adj_loss(prediction, analysis, norm_factors, sht_forward, level_weights, per_variable_weights):
    import numpy as np
    import xarray as xr

    from trainer.spectrum import cross_spectral_density_ds
    from trainer.spectrum import power_spectral_density_ds
    from trainer.spectrum import sht_ds

    pred_spec = sht_ds(prediction / norm_factors, sht_forward)
    targ_spec = sht_ds(analysis / norm_factors, sht_forward)
    pred_psd = power_spectral_density_ds(pred_spec)
    targ_psd = power_spectral_density_ds(targ_spec)
    cross_psd = cross_spectral_density_ds(pred_spec, targ_spec)

    max_psd = np.maximum(pred_psd, targ_psd)
    geo_mean_psd = (1e-16 + pred_psd * targ_psd) ** 0.5
    corr_coef = np.minimum(1, cross_psd / geo_mean_psd)
    raw_mse = (pred_psd + targ_psd - 2 * geo_mean_psd) + 2 * max_psd * (1 - corr_coef)

    dc_vars = {
        var: np.abs(
            pred_spec[var].isel(total_wavenumber=0, zonal_wavenumber=0)
            - targ_spec[var].isel(total_wavenumber=0, zonal_wavenumber=0)
        )
        ** 2
        for var in raw_mse.data_vars
    }
    dc_dset = xr.Dataset(dc_vars)

    mse_by_var = (raw_mse * level_weights).sum(dim="level").mean(dim=("time", "batch"))
    dc_by_var = (dc_dset * level_weights).sum(dim="level").mean(dim=("time", "batch"))
    loss_by_var = dc_by_var + mse_by_var.isel(total_wavenumber=slice(1, None)).sum(
        dim=("total_wavenumber")
    )
    loss = sum(per_variable_weights.get(v, 1.0) * loss_by_var[v] for v in mse_by_var.data_vars)
    return loss, loss_by_var


def spatial_loss(
    prediction,
    targets,
    per_variable_weights,
    level_weights,
    norms_by_level,
    latitude_weights,
    compute_time_bias,
    compute_mean_bias,
):
    prediction = prediction[list(targets.data_vars)]
    diffs = targets - prediction
    adj_level_weights = level_weights / norms_by_level**2
    mse = (((diffs**2).mean(dim="lon") * latitude_weights).mean(dim="lat") * adj_level_weights).sum(
        dim="level"
    ).mean(dim=("time", "batch"))
    if compute_time_bias and diffs.time.size > 1:
        mse_time_bias = (
            (((diffs.mean(dim="time") ** 2).mean(dim="lon")) * latitude_weights).mean(dim="lat")
            * adj_level_weights
        ).sum(dim="level").mean(dim="batch")
        mse = mse + 0.1 * mse_time_bias
    if compute_mean_bias:
        mse_mean_bias = (
            ((diffs.mean(dim="lon") * latitude_weights).mean(dim="lat") ** 2) * adj_level_weights
        ).sum(dim="level").mean(dim=("time", "batch"))
        mse = mse + 0.1 * mse_mean_bias
    total = sum(mse[i] * per_variable_weights.get(i, 1.0) for i in mse.data_vars)
    return total, mse


def mae_loss(prediction, targets, per_variable_weights, level_weights, norms_by_level, latitude_weights):
    import numpy as np

    prediction = prediction[list(targets.data_vars)]
    diffs = targets - prediction
    adj_level_weights = level_weights / norms_by_level
    mae = ((np.abs(diffs).mean(dim="lon") * latitude_weights).mean(dim="lat") * adj_level_weights).sum(
        dim="level"
    ).mean(dim=("time", "batch"))
    total = sum(mae[i] * per_variable_weights.get(i, 1.0) for i in mae.data_vars)
    return total, mae


def make_loss(norms_by_level, per_variable_weights, level_weights, latitude_weights):
    """Backward-compatible custom weighted MSE builder."""

    def my_loss(prediction, targets):
        prediction = prediction[list(targets.data_vars)]
        diffs = (targets - prediction) / norms_by_level
        mse = (diffs**2 * latitude_weights * level_weights).sum(dim=("level")).mean(
            dim=("lat", "lon", "time", "batch")
        )
        total = sum(mse[i] * per_variable_weights.get(i, 1.0) for i in mse.data_vars)
        return total, mse

    return my_loss
