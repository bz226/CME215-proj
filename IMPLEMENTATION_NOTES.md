# GraphCast Small LAMSE Implementation Notes

## Fork Scope

This folder is a dedicated fork of `../graphcast-amse` for the lower-resolution GraphCast Small proposal route. The intended code delta is deliberately small:

- keep the GraphCast architecture and training loop from `graphcast-amse`;
- use the GraphCast Small checkpoint through `params/graphcast_small_lamse.000000.npz`;
- keep AMSE as the lambda-zero baseline;
- add LAMSE only as a configurable hybrid loss.

The lower model resolution is not hard-coded in the training code. It is read from the checkpoint's `model_config`, so the main lower-resolution changes are checkpoint/data/stat paths and spectral-loss geometry derived from that model grid.

## Summary

Implemented the LAMSE hybrid-loss scaffolding in this `code/graphcast-small-lamse` tree, using `code/graphcast-amse` as the AMSE reference.

Added:

- `config.py`
- `needlet_construction.py`
- `needlet_transform.py`
- `lamse_loss.py`
- `trainer/spectrum.py`
- LAMSE/AMSE flags and configurable loss construction in `trainer/loss_utils.py`
- `forecast.generate_model.get_model_coords`
- pytest coverage under `tests/`

Training can now select:

```bash
python train.py --spectral-amse
python train.py --lamse --lamse-lambda 0.1
```

## Deviations and Caveats

- `code/graphcast-amse` AMSE uses a custom JAX/GPU spherical harmonic transform. The LAMSE branch follows the instruction and calls `s2fft` for spherical harmonic forward/inverse transforms.
- The LAMSE implementation first bilinearly samples the GraphCast lat-lon field onto a HEALPix grid, then uses `s2fft` with `sampling="healpix"`. This is a differentiable approximation introduced because GraphCast's regular lat-lon grid is not one of the exact `s2fft` sampling-theorem grids.
- Needlet coefficients are evaluated on one common HEALPix output grid at the maximum required scale resolution so that inverse transforms can be vectorized over scales. This over-resolves coarse scales rather than using a different `Nside` per scale.
- `LAMSEConfig.J` is treated as a lower bound. If the requested maximum degree requires more dyadic scales to preserve partition of unity, the filter bank is expanded.
- HEALPix needlets are approximate and are not exact Gauss-Legendre needlets.

## Validation Performed

Successful:

- Python syntax compilation for the changed modules.
- CLI config parsing unit test.

Command:

```bash
cd code/graphcast-small-lamse
PYTHONPATH=. pytest tests/test_loss_cli_config.py -q
```

Result in this local environment:

```text
1 passed
```

The remaining numerical tests require a working JAX runtime, `s2fft`, and `healpy`.

## Local Environment Blockers

The local Python environment cannot import JAX:

```text
RuntimeError: This version of jaxlib was built using AVX instructions,
which your CPU and/or operating system do not support.
```

Installing `s2fft` and `healpy` was attempted. `healpy` had a wheel, but `s2fft` attempted a source build and failed because the local macOS compiler/linker setup could not compile a simple C++ test program:

```text
ld: unsupported tapi file type '!tapi-tbd' ... libSystem.tbd
```

Because of these environment issues, I could not run:

- Needlet tight-frame test.
- LAMSE self-loss test.
- Longitude-shift sensitivity ratio.
- Gradient sanity.
- Forward/backward performance.
- 100-batch GraphCast integration checks.

## Performance

No reliable wall-clock or memory numbers were produced locally because JAX and `s2fft` could not run.

Expected bottlenecks:

- Lat-lon to HEALPix interpolation at every field/level/time slice.
- `s2fft` forward transform on the input HEALPix grid.
- Per-scale inverse transforms to the common output HEALPix grid.

The HEALPix `s2fft` path requires `L >= 2*nside`. Automatic input/output `nside` choices are capped to this limit. If wall-clock exceeds 3x AMSE in the target GPU environment, first reduce `--lamse-lmax`, `coarse_lat/coarse_lon`, or the output HEALPix `Nside`, then profile the `s2fft` forward/inverse calls.

## Next Validation Steps

Run in the intended GPU environment with `s2fft` and `healpy` installed:

1. `PYTHONPATH=. pytest tests/test_needlet_construction.py tests/test_lamse_loss.py -q`
2. Lambda-zero AMSE equivalence.
3. Synthetic longitude-shift ratio; record the exact LAMSE/AMSE value here.
4. One forward/backward timing comparison against AMSE.
5. 100 batches with `--lamse --lamse-lambda 0.0`.
6. 100 batches with `--lamse --lamse-lambda 0.01 --lamse-lmax 32`.
