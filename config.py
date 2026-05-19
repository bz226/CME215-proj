"""Configuration for localized AMSE experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LAMSEConfig:
    """Configuration for the hybrid AMSE/LAMSE loss."""

    B: float = 2.0
    J: int = 6
    L: int | None = None
    lambda_lamse: float = 0.1
    locality_c: float = 3.0
    coarse_lat: int = 32
    coarse_lon: int = 64
    delta: float = 1e-12
    s2fft_method: str = "jax"
    input_nside: int | None = None
    output_nside: int | None = None


def small_test_config() -> LAMSEConfig:
    """Small CPU/GPU test configuration."""

    return LAMSEConfig(J=4, L=8, coarse_lat=4, coarse_lon=8)
