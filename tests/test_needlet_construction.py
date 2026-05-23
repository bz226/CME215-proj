import pytest

try:
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - environment guard
    pytest.skip(f"JAX runtime unavailable: {exc}", allow_module_level=True)

from config import LAMSEConfig, small_test_config
from needlet_construction import build_needlet_geometry, needlet_filters


def test_window_partition_of_unity():
    config = small_test_config()
    filters = needlet_filters(config.L, config)
    partition = jnp.sum(filters[:, 1:] ** 2, axis=0)
    assert float(jnp.max(jnp.abs(partition - 1.0))) < 1e-6


def test_healpix_geometry_metadata():
    pytest.importorskip("healpy")
    config = small_test_config()
    latitudes = jnp.linspace(-90.0, 90.0, 2 * config.L + 1)
    geometry = build_needlet_geometry(config, latitudes)
    assert geometry.centers_theta.ndim == 1
    assert geometry.centers_phi.shape == geometry.centers_theta.shape
    assert geometry.bandlimit >= 2 * geometry.input_nside
    assert geometry.bandlimit >= 2 * geometry.output_nside
    assert geometry.local_weights.shape[0] == geometry.filters.shape[0]
    assert geometry.coarse_area.ndim == 1


def test_auto_output_nside_respects_s2fft_healpix_limit():
    pytest.importorskip("healpy")
    config = LAMSEConfig(J=6, L=16, coarse_lat=4, coarse_lon=8)
    latitudes = jnp.linspace(-90.0, 90.0, 2 * config.L + 1)
    geometry = build_needlet_geometry(config, latitudes)
    assert geometry.output_nside == 8
    assert geometry.bandlimit >= 2 * geometry.output_nside


def test_explicit_output_nside_error_is_clear():
    pytest.importorskip("healpy")
    config = LAMSEConfig(J=6, L=16, output_nside=16)
    latitudes = jnp.linspace(-90.0, 90.0, 2 * config.L + 1)
    with pytest.raises(ValueError, match="HEALPix requires L >= 2\\*nside"):
        build_needlet_geometry(config, latitudes)
