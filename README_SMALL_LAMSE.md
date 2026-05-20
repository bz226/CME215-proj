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
sh_dev
bash scripts/setup_sherlock_env.sh
source .venv-sherlock/activate_graphcast_small_lamse.sh
```

The setup follows Sherlock's Python guidance: use modules plus `python3 -m venv`, install packages inside the activated venv, and run package installation from a compute node. By default, it creates the environment locally at `.venv-sherlock/` inside this project directory. Sherlock's available Python modules include `3.12.1`, which is the default used by `setup_sherlock_env.sh`; avoid `3.9.0` for this JAX/GraphCast stack and avoid `3.14.2` because scientific wheels may lag behind it.

The activation helper sets `JAX_PLATFORMS=cuda`, `JAX_ENABLE_X64=True`, and conservative thread limits. Run JAX smoke tests from a GPU allocation; CPU-only device discovery can fail on constrained Sherlock allocations.

If setting up from a CPU-only allocation, skip the CUDA smoke check:

```bash
RUN_JAX_SMOKE=0 bash scripts/setup_sherlock_env.sh
```

or run the smoke check against CPU explicitly:

```bash
SMOKE_JAX_PLATFORMS=cpu bash scripts/setup_sherlock_env.sh
```

Use the generated activation helper unchanged for GPU training jobs; it still sets `JAX_PLATFORMS=cuda`.

Sherlock Slurm wrappers are included:

```bash
# Build/update the local .venv-sherlock environment as a batch job.
sbatch scripts/sbatch_setup_env.sh

# Stage WeatherBench 2 ERA5 data on CPU/network resources.
DATA_DIR=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
  sbatch scripts/sbatch_fetch_training_data.sh

# Open an interactive GPU allocation for debugging.
bash scripts/srun_gpu_shell.sh

# Submit the 100-batch LAMSE/AMSE gate.
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 LAMSE_LAMBDA=0.0 \
  sbatch scripts/sbatch_lamse_100_batches.sh
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

Stage the data before requesting a GPU:

```bash
DATA_DIR=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
  sbatch scripts/sbatch_fetch_training_data.sh
```

The staged layout is `DATA_DIR/YYYY/MM`, matching the loader's `ANALYSIS_PATH/*/*` monthly zarr expectation. The default staging window covers the current 100-batch gate dates plus one 6-hour input/target buffer.

For batch submission on Sherlock:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 LAMSE_LAMBDA=0.0 \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

Use `LAMSE_LAMBDA=0.1` only after the lambda-zero gate is stable.

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
