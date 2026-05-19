# Helper functions to compute the spherical harmonic transform, using Jax-on-GPU.


def generate_spectral_coefs(lat_deg):
    """Return weighted associated-Legendre coefficients for the AMSE SHT."""

    import jax
    import jax.numpy as jnp
    import numpy as np

    lat_deg = np.asanyarray(lat_deg)
    nlat = lat_deg.size
    lat_rad = jnp.array(lat_deg) * jnp.pi / 180
    cart_z = jnp.sin(lat_rad)

    lat_mid = jnp.concatenate(
        (
            jnp.array([-jnp.pi / 2]),
            0.5 * (lat_rad[1:] + lat_rad[:-1]),
            jnp.array([jnp.pi / 2]),
        )
    )
    lat_weights = jnp.sin(lat_mid[1:]) - jnp.sin(lat_mid[:-1])
    lat_weights = lat_weights / jnp.sum(lat_weights)

    if jnp.array([0]).devices().issubset(jax.devices("cpu")):
        raise ValueError("Spherical harmonic coefficient generation currently requires a GPU")

    nspec = nlat - 1
    leg_coef = jax.scipy.special.lpmn_values(nspec, nspec, cart_z, True)

    orthonorm_factor_squared = 1 / jnp.sum(leg_coef[0, 0, :] ** 2 * lat_weights)
    orthonorm_factor = orthonorm_factor_squared**0.5
    leg_coef_weighted = (
        leg_coef[:, :, :] * (orthonorm_factor * lat_weights[None, None, :])
    ).astype(jnp.float32)
    return np.array(leg_coef_weighted)


def sht_eval(weighted_coef, arr):
    """Evaluate the complete AMSE spherical harmonic transform."""

    import jax
    import jax.numpy as jnp

    means = jnp.mean(arr, axis=(-1, -2))
    arr = arr - means[..., None, None]
    farr = jnp.fft.rfft(arr, norm="forward", axis=-1)
    sht = jnp.einsum("ijk,...ki->...ij", weighted_coef, farr)
    return jax.lax.dynamic_update_slice(
        sht,
        means.reshape(sht.shape[:-2] + (1, 1)) + sht[..., 0, 0][..., None, None],
        (0,) * sht.ndim,
    )


def power_spectral_density(fspec):
    """Power spectral density by total wavenumber."""

    import jax.numpy as jnp

    return jnp.abs(fspec[..., 0, :]) ** 2 + 2 * jnp.sum(
        jnp.abs(fspec[..., 1:, :]) ** 2, axis=-2
    )


def cross_spectral_density(speca, specb):
    """Real cross spectral density by total wavenumber."""

    import jax.numpy as jnp

    areal = speca.real
    breal = specb.real
    aimag = speca.imag
    bimag = specb.imag
    return areal[..., 0, :] * breal[..., 0, :] + 2 * jnp.sum(
        areal[..., 1:, :] * breal[..., 1:, :]
        + aimag[..., 1:, :] * bimag[..., 1:, :],
        axis=-2,
    )


def sht_ds(ds_in, sht_forward):
    """Apply an AMSE SHT to every variable in an xarray Dataset."""

    from graphcast import xarray_jax
    from jax import tree_util
    import numpy as np

    exploded = xarray_jax.unwrap_vars(ds_in)
    spec_dict = tree_util.tree_map(sht_forward, exploded)
    coords = ds_in[set(ds_in.coords) - {"latitude", "longitude", "lat", "lon"}].coords
    rewrapped = xarray_jax.Dataset(
        {
            v: (
                ds_in[v].dims[:-2] + ("zonal_wavenumber", "total_wavenumber"),
                spec_dict[v],
            )
            for v in spec_dict.keys()
        },
        coords=coords,
    )
    return rewrapped.assign_coords(total_wavenumber=("total_wavenumber", np.arange(ds_in.lat.size)))


def power_spectral_density_ds(spec):
    """Compute power spectra for all variables in a spectral Dataset."""

    from graphcast import xarray_jax
    from jax import tree_util

    exploded = xarray_jax.unwrap_vars(spec)
    power_spec_dict = tree_util.tree_map(power_spectral_density, exploded)
    return xarray_jax.Dataset(
        {
            v: (spec[v].dims[:-2] + ("total_wavenumber",), power_spec_dict[v])
            for v in power_spec_dict.keys()
        },
        coords=spec.coords,
    )


def cross_spectral_density_ds(pspec, tspec):
    """Compute cross spectra for all variables in two spectral Datasets."""

    from graphcast import xarray_jax
    from jax import tree_util

    pexplode = xarray_jax.unwrap_vars(pspec)
    aexplode = xarray_jax.unwrap_vars(tspec)
    cross_spec_dict = tree_util.tree_map(cross_spectral_density, pexplode, aexplode)
    return xarray_jax.Dataset(
        {
            v: (pspec[v].dims[:-2] + ("total_wavenumber",), cross_spec_dict[v])
            for v in cross_spec_dict.keys()
        },
        coords=pspec.coords,
    )
