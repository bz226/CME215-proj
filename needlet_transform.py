"""Forward needlet transform built on s2fft."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from needlet_construction import NeedletGeometry


@dataclass(frozen=True)
class LatLonToPoints:
    """Bilinear interpolation indices from a regular lat-lon grid to points."""

    lat0: jnp.ndarray
    lat1: jnp.ndarray
    lat_w: jnp.ndarray
    lon0: jnp.ndarray
    lon1: jnp.ndarray
    lon_w: jnp.ndarray


def build_latlon_to_points(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    point_theta: np.ndarray,
    point_phi: np.ndarray,
) -> LatLonToPoints:
    """Precompute regular-grid bilinear interpolation to spherical points."""

    latitudes = np.asarray(latitudes, dtype=np.float64)
    longitudes = np.asarray(longitudes, dtype=np.float64)
    point_lat = 90.0 - np.rad2deg(np.asarray(point_theta, dtype=np.float64))
    point_lon = np.mod(np.rad2deg(np.asarray(point_phi, dtype=np.float64)), 360.0)

    lat_descending = latitudes[0] > latitudes[-1]
    work_lat = latitudes[::-1] if lat_descending else latitudes
    point_lat = np.clip(point_lat, work_lat[0], work_lat[-1])
    lat0 = np.searchsorted(work_lat, point_lat, side="right") - 1
    lat0 = np.clip(lat0, 0, work_lat.size - 2)
    lat1 = lat0 + 1
    lat_w = (point_lat - work_lat[lat0]) / (work_lat[lat1] - work_lat[lat0])
    if lat_descending:
        lat0, lat1 = work_lat.size - 1 - lat0, work_lat.size - 1 - lat1

    lon0_ref = float(longitudes[0])
    lon_step = float(np.median(np.diff(np.unwrap(np.deg2rad(longitudes))) * 180.0 / np.pi))
    if lon_step <= 0:
        raise ValueError("Longitude coordinate must be ascending and regular.")
    lon_pos = np.mod(point_lon - lon0_ref, 360.0) / lon_step
    lon0 = np.floor(lon_pos).astype(np.int64) % longitudes.size
    lon1 = (lon0 + 1) % longitudes.size
    lon_w = lon_pos - np.floor(lon_pos)

    return LatLonToPoints(
        lat0=jnp.asarray(lat0, dtype=jnp.int32),
        lat1=jnp.asarray(lat1, dtype=jnp.int32),
        lat_w=jnp.asarray(lat_w, dtype=jnp.float32),
        lon0=jnp.asarray(lon0, dtype=jnp.int32),
        lon1=jnp.asarray(lon1, dtype=jnp.int32),
        lon_w=jnp.asarray(lon_w, dtype=jnp.float32),
    )


def interpolate_latlon_to_points(field: jnp.ndarray, interp: LatLonToPoints) -> jnp.ndarray:
    """Bilinearly interpolate `field[..., lat, lon]` to point samples."""

    f00 = field[..., interp.lat0, interp.lon0]
    f01 = field[..., interp.lat0, interp.lon1]
    f10 = field[..., interp.lat1, interp.lon0]
    f11 = field[..., interp.lat1, interp.lon1]
    top = f00 * (1.0 - interp.lon_w) + f01 * interp.lon_w
    bottom = f10 * (1.0 - interp.lon_w) + f11 * interp.lon_w
    return top * (1.0 - interp.lat_w) + bottom * interp.lat_w


def spherical_harmonic_coefficients(
    field: jnp.ndarray,
    interp: LatLonToPoints,
    geometry: NeedletGeometry,
) -> jnp.ndarray:
    """Compute s2fft spherical harmonic coefficients from a lat-lon field."""

    try:
        from s2fft.transforms import spherical
    except ImportError as exc:
        raise ImportError("LAMSE needlet transforms require `s2fft`.") from exc

    healpix_map = interpolate_latlon_to_points(field, interp)
    return spherical.forward(
        healpix_map,
        L=geometry.bandlimit,
        nside=geometry.input_nside,
        sampling="healpix",
        method=geometry.config.s2fft_method,
        reality=False,
    )


def needlet_coefficients(
    field: jnp.ndarray,
    interp: LatLonToPoints,
    geometry: NeedletGeometry,
) -> jnp.ndarray:
    """Return beta[j, k] needlet coefficients for one 2D field."""

    try:
        from s2fft.transforms import spherical
    except ImportError as exc:
        raise ImportError("LAMSE needlet transforms require `s2fft`.") from exc

    flm = spherical_harmonic_coefficients(field, interp, geometry)
    filters = geometry.filters.astype(flm.real.dtype).astype(flm.dtype)
    filtered = filters[:, :, None] * flm[None, :, :]

    def inverse_one_scale(scale_flm):
        values = spherical.inverse(
            scale_flm,
            L=geometry.bandlimit,
            nside=geometry.output_nside,
            sampling="healpix",
            method=geometry.config.s2fft_method,
            reality=False,
        )
        return jnp.real(values) * geometry.sqrt_center_area

    return jax.vmap(inverse_one_scale)(filtered)


def inverse_needlet_synthesis(
    flm: jnp.ndarray,
    geometry: NeedletGeometry,
) -> jnp.ndarray:
    """Synthesize the finite needlet filter bank on the output HEALPix grid."""

    try:
        from s2fft.transforms import spherical
    except ImportError as exc:
        raise ImportError("LAMSE needlet transforms require `s2fft`.") from exc

    filters = geometry.filters.astype(flm.real.dtype).astype(flm.dtype)
    filtered = filters[:, :, None] * flm[None, :, :]

    def inverse_one_scale(scale_flm):
        return spherical.inverse(
            scale_flm,
            L=geometry.bandlimit,
            nside=geometry.output_nside,
            sampling="healpix",
            method=geometry.config.s2fft_method,
            reality=False,
        )

    return jnp.real(jnp.sum(jax.vmap(inverse_one_scale)(filtered), axis=0))
