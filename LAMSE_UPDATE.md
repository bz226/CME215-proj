# GraphCast Small LAMSE Update

This fork starts from `../graphcast-amse` and keeps its training infrastructure as the AMSE reference. It adds a local spectral LAMSE branch while targeting the proposal's lower-resolution GraphCast Small checkpoint.

## Checkpoint

Use the proposal checkpoint:

```text
Hugging Face repo: shermansiu/dm_graphcast_small
file: GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz
```

Prepare the local trainable alias:

```bash
python3 scripts/download_graphcast_small.py
python3 scripts/prepare_graphcast_small_checkpoint.py
```

Training uses:

```text
params/graphcast_small_lamse.000000.npz
```

The `000000` token is required because `train.py` parses the initial batch number from the checkpoint filename.

## Loss Commands

AMSE baseline:

```bash
python train.py --spectral-amse ...
```

Hybrid local spectral loss:

```bash
python train.py --lamse --lamse-lambda 0.1 ...
```

Lambda-zero equivalence gate:

```bash
python train.py --lamse --lamse-lambda 0.0 --batch-number 100 ...
```

The lambda-zero run should match AMSE behavior before any longer LAMSE training is attempted.

## What Changed

- Added LAMSE modules: `config.py`, `needlet_construction.py`, `needlet_transform.py`, `lamse_loss.py`.
- Updated `trainer/loss_utils.py` with `--lamse`, `--lamse-lambda`, and `--lamse-lmax`.
- Added `healpy` and `s2fft` to `environment.yml`.
- Updated `forecast/models.py` so `era5_100` points at `params/graphcast_small_lamse.000000.npz`.
- Added checkpoint helper scripts under `scripts/`.

The GraphCast architecture, data-loader contract, autoregressive training loop, and optimizer are intentionally inherited from `graphcast-amse`.
