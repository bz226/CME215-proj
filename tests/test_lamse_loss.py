import pytest

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"JAX runtime unavailable: {exc}", allow_module_level=True)

pytest.importorskip("healpy")
pytest.importorskip("s2fft")

import numpy as np
import xarray as xr

from config import small_test_config
from lamse_loss import build_lamse_precompute, lamse_one_field


def _blob(lat, lon, center_lat=10.0, center_lon=120.0):
    lat2 = (lat[:, None] - center_lat) ** 2
    dlon = ((lon[None, :] - center_lon + 180.0) % 360.0) - 180.0
    return jnp.exp(-(lat2 + dlon**2) / (2.0 * 8.0**2))


def test_self_loss_is_zero():
    config = small_test_config()
    lat = np.linspace(-90.0, 90.0, 2 * config.L + 1, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 4 * config.L, endpoint=False, dtype=np.float32)
    precompute = build_lamse_precompute(lat, lon, config)
    field = _blob(jnp.asarray(lat), jnp.asarray(lon))
    value = lamse_one_field(field, field, precompute)
    assert float(value) < 1e-8


def test_gradient_is_finite():
    config = small_test_config()
    lat = np.linspace(-90.0, 90.0, 2 * config.L + 1, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 4 * config.L, endpoint=False, dtype=np.float32)
    precompute = build_lamse_precompute(lat, lon, config)
    x = _blob(jnp.asarray(lat), jnp.asarray(lon))
    y = _blob(jnp.asarray(lat), jnp.asarray(lon), center_lon=140.0)

    grad = jax.grad(lambda z: lamse_one_field(z, y, precompute))(x)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert float(jnp.linalg.norm(grad)) > 0.0


def test_xarray_signature_smoke():
    # Keeps the expected GraphCast-like dimensions visible in tests. The full
    # dataset aggregation path is covered by training integration tests.
    lat = np.linspace(-90.0, 90.0, 9, dtype=np.float32)
    lon = np.linspace(0.0, 360.0, 16, endpoint=False, dtype=np.float32)
    data = np.zeros((1, 1, 2, lat.size, lon.size), dtype=np.float32)
    ds = xr.Dataset(
        {"geopotential": (("batch", "time", "level", "lat", "lon"), data)},
        coords={"batch": [0], "time": [0], "level": [500, 850], "lat": lat, "lon": lon},
    )
    assert ds.geopotential.dims == ("batch", "time", "level", "lat", "lon")
