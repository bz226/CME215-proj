# GraphCast Small LAMSE Fork

This folder is a focused fork of `../graphcast-amse` for the proposal route:

`GraphCast Small 1 degree checkpoint -> AMSE/LAMSE fine-tuning -> displacement-sensitive evaluation`

The intent is to stay close to `graphcast-amse`. The architecture, checkpoint format, data-loader contract, autoregressive training loop, optimizer, and AMSE baseline are retained. The meaningful code difference is the optional hybrid local spectral loss:

`hybrid = (1 - lambda) * AMSE + lambda * LAMSE`

At `--lamse-lambda 0.0`, the loss returns the AMSE branch directly.

## Main Differences From `graphcast-amse`

- `forecast/models.py` points `era5_100` at `params/graphcast_small_lamse.000000.npz`.
- `trainer/loss_utils.py` adds `--lamse`, `--lamse-lambda`, and `--lamse-lmax`.
- `config.py`, `needlet_construction.py`, `needlet_transform.py`, and `lamse_loss.py` implement the local needlet/LAMSE path.
- `environment.yml` adds `healpy` and `s2fft`.
- `tests/` contains synthetic LAMSE and CLI tests copied from the active LAMSE implementation.

## Checkpoint

Download the proposal checkpoint:

```bash
python3 scripts/download_graphcast_small.py
python3 scripts/prepare_graphcast_small_checkpoint.py
python3 scripts/inspect_graphcast_checkpoint.py
```

The prepared path is:

```text
params/graphcast_small_lamse.000000.npz
```

The `000000` token is intentional. `train.py` parses the initial batch number from the token immediately before the file extension.

## First Validation Runs

On Stanford Sherlock, create the recommended module/venv environment with:

```bash
bash scripts/setup_sherlock_env.sh
source "${SCRATCH}/venvs/graphcast-small-lamse/activate_graphcast_small_lamse.sh"
```

Run local parser tests first:

```bash
PYTHONPATH=. pytest tests/test_loss_cli_config.py -q
```

Then, in an environment with working JAX, `s2fft`, and `healpy`:

```bash
PYTHONPATH=. pytest tests/test_needlet_construction.py tests/test_lamse_loss.py -q
```

## 100-Batch Gates

From this folder, after preparing the checkpoint and a compatible 1 degree ERA5/WeatherBench data path:

```bash
PYTHONPATH=. python train.py \
  --model-checkpoint params/graphcast_small_lamse.000000.npz \
  --apath /path/to/era5_or_weatherbench_1deg \
  --norm-factors stats \
  --start-date "1 Jan 2016 00:00" \
  --end-date "31 Dec 2017 18:00" \
  --forecast-length 1 \
  --batch-size 1 \
  --batch-number 100 \
  --checkpoint-every 100 \
  --learning-rate 1e-6 \
  --lamse \
  --lamse-lambda 0.0
```

Only after lambda-zero matches AMSE behavior, repeat with:

```bash
--lamse-lambda 0.1
```

Do not run longer fine-tuning until the 100-batch lambda-zero and lambda-0.1 gates pass.
