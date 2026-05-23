"""Needlet windows, HEALPix centers, and localization weights."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from config import LAMSEConfig


@dataclass(frozen=True)
class NeedletGeometry:
    """Static geometry needed by the localized needlet loss."""

    config: LAMSEConfig
    max_degree: int
    bandlimit: int
    filters: jnp.ndarray
    ell: jnp.ndarray
    input_nside: int
    output_nside: int
    centers_theta: jnp.ndarray
    centers_phi: jnp.ndarray
    sqrt_center_area: jnp.ndarray
    local_weights: jnp.ndarray
    coarse_area: jnp.ndarray


def smooth_phi(t: jnp.ndarray, B: float = 2.0) -> jnp.ndarray:
    """C-infinity monotone cutoff equal to 1 on [0, 1/B] and 0 on [1, inf)."""

    t = jnp.asarray(t)
    a = 1.0 / B
    u = (t - a) / (1.0 - a)
    u = jnp.clip(u, 0.0, 1.0)

    def bump(x):
        return jnp.where(x > 0, jnp.exp(-1.0 / x), 0.0)

    left = bump(u)
    right = bump(1.0 - u)
    transition = left / (left + right + jnp.finfo(t.dtype).tiny)
    return jnp.where(t <= a, 1.0, jnp.where(t >= 1.0, 0.0, 1.0 - transition))


def raw_needlet_window(t: jnp.ndarray, B: float = 2.0) -> jnp.ndarray:
    """Marinucci-style needlet window b(t)."""

    value = smooth_phi(t / B, B) - smooth_phi(t, B)
    return jnp.sqrt(jnp.maximum(value, 0.0))


def needlet_filters(max_degree: int, config: LAMSEConfig) -> jnp.ndarray:
    """Return finite scale filters normalized to partition unity for ell >= 1."""

    ell = jnp.arange(max_degree + 1, dtype=jnp.float32)
    scale_exponents = _scale_exponents(max_degree, config)
    raw = jnp.stack(
        [raw_needlet_window(ell / (config.B**j), config.B) for j in scale_exponents],
        axis=0,
    )
    norm = jnp.sqrt(jnp.sum(raw**2, axis=0, keepdims=True))
    filters = jnp.where(norm > 0, raw / jnp.maximum(norm, 1e-30), 0.0)
    return filters.at[:, 0].set(0.0)


def build_needlet_geometry(config: LAMSEConfig, latitudes: Sequence[float]) -> NeedletGeometry:
    """Build static HEALPix and localization geometry.

    The implementation uses one common HEALPix output grid at the maximum scale
    resolution so the inverse transforms can be vectorized over scale. This
    over-resolves coarser scales while still satisfying the resolution
    requirement for every scale.
    """

    max_degree = int(config.L) if config.L is not None else int(len(latitudes) - 1)
    bandlimit = max_degree + 1
    filters = needlet_filters(max_degree, config)
    max_healpix_nside = _max_nside_for_bandlimit(bandlimit)
    input_nside = config.input_nside or _nside_for_degree(max_degree)
    if input_nside > max_healpix_nside:
        raise ValueError(
            f"LAMSE input_nside={input_nside} is too high for s2fft bandlimit "
            f"L={bandlimit}; HEALPix requires L >= 2*nside. Increase "
            f"--lamse-lmax to at least {2 * input_nside - 1} or lower input_nside."
        )

    auto_output_nside = max(
        _nside_for_degree(config.B ** (j + 1))
        for j in _scale_exponents(max_degree, config)
    )
    if config.output_nside is None:
        output_nside = min(auto_output_nside, max_healpix_nside)
    else:
        output_nside = config.output_nside
        if output_nside > max_healpix_nside:
            raise ValueError(
                f"LAMSE output_nside={output_nside} is too high for s2fft "
                f"bandlimit L={bandlimit}; HEALPix requires L >= 2*nside. "
                f"Increase --lamse-lmax to at least {2 * output_nside - 1} "
                "or lower output_nside."
            )

    theta, phi = healpix_centers(output_nside)
    coarse_theta, coarse_phi, coarse_area = coarse_grid(config.coarse_lat, config.coarse_lon)
    local_weights = localization_weights(
        theta,
        phi,
        coarse_theta,
        coarse_phi,
        _scale_exponents(max_degree, config),
        config,
    )
    center_area = 4.0 * math.pi / theta.size
    return NeedletGeometry(
        config=config,
        max_degree=max_degree,
        bandlimit=bandlimit,
        filters=filters,
        ell=jnp.arange(max_degree + 1),
        input_nside=input_nside,
        output_nside=output_nside,
        centers_theta=jnp.asarray(theta, dtype=jnp.float32),
        centers_phi=jnp.asarray(phi, dtype=jnp.float32),
        sqrt_center_area=jnp.asarray(math.sqrt(center_area), dtype=jnp.float32),
        local_weights=jnp.asarray(local_weights, dtype=jnp.float32),
        coarse_area=jnp.asarray(coarse_area, dtype=jnp.float32),
    )


def healpix_centers(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Return HEALPix RING pixel centers as theta, phi in radians."""

    try:
        import healpy as hp
    except ImportError as exc:
        raise ImportError("LAMSE HEALPix geometry requires `healpy`.") from exc

    pix = np.arange(hp.nside2npix(nside))
    theta, phi = hp.pix2ang(nside, pix, nest=False)
    return theta.astype(np.float32), phi.astype(np.float32)


def coarse_grid(nlat: int, nlon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Equal-area coarse sphere cells."""

    z_edges = np.linspace(-1.0, 1.0, nlat + 1, dtype=np.float64)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])
    theta_lat = np.arccos(z)
    phi_lon = (np.arange(nlon, dtype=np.float64) + 0.5) * (2.0 * np.pi / nlon)
    theta, phi = np.meshgrid(theta_lat, phi_lon, indexing="ij")
    area = np.full(theta.size, 4.0 * np.pi / (nlat * nlon), dtype=np.float32)
    return theta.ravel().astype(np.float32), phi.ravel().astype(np.float32), area


def localization_weights(
    center_theta: np.ndarray,
    center_phi: np.ndarray,
    coarse_theta: np.ndarray,
    coarse_phi: np.ndarray,
    scale_exponents: Sequence[int],
    config: LAMSEConfig,
) -> np.ndarray:
    """Gaussian weights from needlet centers to coarse cells."""

    center_xyz = _spherical_to_cartesian(center_theta, center_phi)
    coarse_xyz = _spherical_to_cartesian(coarse_theta, coarse_phi)
    cos_d = np.clip(coarse_xyz @ center_xyz.T, -1.0, 1.0)
    d = np.arccos(cos_d)
    weights = []
    for j in scale_exponents:
        radius = config.locality_c * (config.B ** (-j))
        weights.append(np.exp(-(d**2) / (2.0 * radius**2)))
    return np.stack(weights, axis=0).astype(np.float32)


def _scale_exponents(max_degree: int, config: LAMSEConfig) -> tuple[int, ...]:
    # A finite Marinucci bank must include enough dyadic scales to cover the
    # requested bandlimit. Treat config.J as a lower bound; otherwise high
    # degrees would have zero window weight and violate partition of unity.
    if max_degree <= 1:
        count = config.J
    else:
        count = max(config.J, int(math.ceil(math.log(max_degree, config.B))) + 1)
    return tuple(range(count))


def _nside_for_degree(degree: float) -> int:
    required = max(1, int(math.ceil((float(degree) + 1.0) / 3.0)))
    nside = 1
    while nside < required:
        nside *= 2
    return nside


def _max_nside_for_bandlimit(bandlimit: int) -> int:
    """Largest power-of-two HEALPix nside supported by s2fft for this L."""

    if bandlimit < 2:
        raise ValueError("HEALPix s2fft transforms require bandlimit L >= 2.")
    limit = bandlimit // 2
    nside = 1
    while nside * 2 <= limit:
        nside *= 2
    return nside


def _spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    sin_theta = np.sin(theta)
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
        axis=-1,
    )
