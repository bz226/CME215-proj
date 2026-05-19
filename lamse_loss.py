"""Localized adjusted mean squared error loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from graphcast import xarray_jax
from config import LAMSEConfig
from needlet_construction import NeedletGeometry, build_needlet_geometry, healpix_centers
from needlet_transform import (
    LatLonToPoints,
    build_latlon_to_points,
    needlet_coefficients,
)


@dataclass(frozen=True)
class LAMSEPrecompute:
    """Static objects captured by the LAMSE closure."""

    geometry: NeedletGeometry
    input_interpolator: LatLonToPoints


def build_lamse_precompute(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    config: LAMSEConfig,
) -> LAMSEPrecompute:
    """Build reusable LAMSE geometry and lat-lon interpolation tables."""

    geometry = build_needlet_geometry(config, latitudes)
    input_theta, input_phi = healpix_centers(geometry.input_nside)
    input_interpolator = build_latlon_to_points(
        latitudes,
        longitudes,
        input_theta,
        input_phi,
    )
    return LAMSEPrecompute(geometry=geometry, input_interpolator=input_interpolator)


def localized_psd(beta: jnp.ndarray, local_weights: jnp.ndarray) -> jnp.ndarray:
    """Compute localized PSD from needlet coefficients beta[..., j, k]."""

    return jnp.einsum("jpk,...jk->...jp", local_weights, jnp.square(beta))


def localized_coherence(
    beta_x: jnp.ndarray,
    beta_y: jnp.ndarray,
    psd_x: jnp.ndarray,
    psd_y: jnp.ndarray,
    local_weights: jnp.ndarray,
    delta: float,
) -> jnp.ndarray:
    """Compute localized coherence from needlet coefficients."""

    cross = jnp.einsum("jpk,...jk->...jp", local_weights, beta_x * beta_y)
    coh = cross / jnp.sqrt(psd_x * psd_y + delta)
    return jnp.minimum(coh, 1.0)


def lamse_from_coefficients(
    beta_x: jnp.ndarray,
    beta_y: jnp.ndarray,
    geometry: NeedletGeometry,
) -> jnp.ndarray:
    """LAMSE for one field from needlet coefficients."""

    beta_x = jnp.real(beta_x)
    beta_y = jnp.real(beta_y)
    psd_x = localized_psd(beta_x, geometry.local_weights)
    psd_y = localized_psd(beta_y, geometry.local_weights)
    coh = localized_coherence(
        beta_x,
        beta_y,
        psd_x,
        psd_y,
        geometry.local_weights,
        geometry.config.delta,
    )
    amp = jnp.square(jnp.sqrt(jnp.maximum(psd_x, 0.0)) - jnp.sqrt(jnp.maximum(psd_y, 0.0)))
    decorrelation = 2.0 * jnp.maximum(psd_x, psd_y) * (1.0 - coh)
    return jnp.sum((amp + decorrelation) * geometry.coarse_area[None, :])


def lamse_one_field(
    prediction: jnp.ndarray,
    target: jnp.ndarray,
    precompute: LAMSEPrecompute,
) -> jnp.ndarray:
    """LAMSE for one normalized 2D lat-lon field."""

    beta_prediction = needlet_coefficients(
        prediction,
        precompute.input_interpolator,
        precompute.geometry,
    )
    beta_target = needlet_coefficients(
        target,
        precompute.input_interpolator,
        precompute.geometry,
    )
    return lamse_from_coefficients(beta_prediction, beta_target, precompute.geometry)


def lamse_dataset_loss(
    prediction: xr.Dataset,
    target: xr.Dataset,
    norm_factors: xr.Dataset,
    precompute: LAMSEPrecompute,
    level_weights: xr.DataArray,
    per_variable_weights: Mapping[str, float],
) -> tuple[jnp.ndarray, xr.Dataset]:
    """LAMSE over a GraphCast prediction/target Dataset."""

    prediction = prediction[list(target.data_vars)]
    per_var = {}
    for name in target.data_vars:
        pred_da = prediction[name] / norm_factors[name]
        targ_da = target[name] / norm_factors[name]
        targ_da = targ_da.transpose(*pred_da.dims)
        per_var[name] = _lamse_dataarray(pred_da, targ_da, precompute, level_weights)

    diagnostics = xarray_jax.Dataset({name: ((), value) for name, value in per_var.items()})
    total = sum(per_variable_weights.get(name, 1.0) * value for name, value in per_var.items())
    return total, diagnostics


def _lamse_dataarray(
    prediction: xr.DataArray,
    target: xr.DataArray,
    precompute: LAMSEPrecompute,
    level_weights: xr.DataArray,
) -> jnp.ndarray:
    lat_name = "lat" if "lat" in prediction.dims else "latitude"
    lon_name = "lon" if "lon" in prediction.dims else "longitude"
    leading_dims = [d for d in prediction.dims if d not in (lat_name, lon_name)]
    ordered = prediction.transpose(*leading_dims, lat_name, lon_name)
    ordered_target = target.transpose(*leading_dims, lat_name, lon_name)

    pred_arr = xarray_jax.unwrap_data(ordered)
    target_arr = xarray_jax.unwrap_data(ordered_target)
    leading_shape = pred_arr.shape[:-2]
    flat_pred = pred_arr.reshape((-1,) + pred_arr.shape[-2:])
    flat_target = target_arr.reshape((-1,) + target_arr.shape[-2:])
    flat_loss = jax.vmap(lambda x, y: lamse_one_field(x, y, precompute))(flat_pred, flat_target)
    loss = flat_loss.reshape(leading_shape)

    if "level" in leading_dims:
        level_axis = leading_dims.index("level")
        weights = jnp.asarray(level_weights.sel(level=prediction.level).data, dtype=loss.dtype)
        weight_shape = [1] * loss.ndim
        weight_shape[level_axis] = weights.shape[0]
        loss = jnp.sum(loss * weights.reshape(weight_shape), axis=level_axis)
        leading_dims = [d for d in leading_dims if d != "level"]

    if loss.ndim == 0:
        return loss
    return jnp.mean(loss)
