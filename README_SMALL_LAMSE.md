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

The Sherlock wrappers pass explicit `--model-checkpoint` paths under `PARAMS_DIR`, so training does not need to store large `.npz` files in the project home directory.

## Checkpoint

On Sherlock, keep model checkpoints in scratch rather than the project home directory. The scripts default to:

```text
$SCRATCH/graphcast-small-lamse/params
```

Override with `PARAMS_DIR=/some/scratch/path` if needed. Local runs without `SCRATCH` still fall back to `params/`.

Download the proposal checkpoint:

```bash
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
python3 scripts/download_graphcast_small.py
python3 scripts/prepare_graphcast_small_checkpoint.py
python3 scripts/inspect_graphcast_checkpoint.py
python3 scripts/download_graphcast_stats.py
```

The prepared path is:

```text
$PARAMS_DIR/graphcast_small_lamse.000000.npz
```

Training also requires the GraphCast normalization files in `stats/`:

```text
stats/diffs_stddev_by_level.nc
stats/mean_by_level.nc
stats/stddev_by_level.nc
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

The activation helper sets `JAX_PLATFORMS=cuda,cpu`, `JAX_ENABLE_X64=True`, and conservative thread limits. By default, JAX is installed with pip-bundled CUDA/cuDNN (`jax[cuda12]`) to avoid local CUDA/cuDNN mismatches across Sherlock GPU nodes. Run JAX smoke tests from a GPU allocation; CPU-only device discovery can fail on constrained Sherlock allocations.

If setting up from a CPU-only allocation, skip the CUDA smoke check:

```bash
RUN_JAX_SMOKE=0 bash scripts/setup_sherlock_env.sh
```

or run the smoke check against CPU explicitly:

```bash
SMOKE_JAX_PLATFORMS=cpu bash scripts/setup_sherlock_env.sh
```

Use the generated activation helper unchanged for GPU training jobs; it still exposes both CUDA and CPU backends because `train.py` keeps canonical parameters on the CPU while evaluating gradients on GPU.

Sherlock Slurm wrappers are included:

```bash
# Build/update the local .venv-sherlock environment as a batch job.
sbatch scripts/sbatch_setup_env.sh

# Stage WeatherBench 2 ERA5 data on CPU/network resources.
DATA_DIR=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
  sbatch scripts/sbatch_fetch_training_data.sh

# Open an interactive GPU allocation for debugging.
bash scripts/srun_gpu_shell.sh

# Submit the pure AMSE 100-batch gate.
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 LOSS_MODE=amse \
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

For pure AMSE batch submission on Sherlock:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 LOSS_MODE=amse \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

For a true AMSE-only training run, use the dedicated wrapper. It always passes
`--spectral-amse`, never passes `--lamse`, and saves AMSE checkpoints under
`$PARAMS_DIR/graphcast_small_amse.<batch>.npz`:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
BATCH_NUMBER=1000 \
  sbatch scripts/sbatch_amse_training.sh
```

The `sbatch` wrappers write project-local logs to `logs/*.log`. The training wrapper also writes per-example losses to `runs/<loss_mode>_<lambda>_job_<jobid>.csv` unless `CSV_PATH` is set.

The Sherlock wrappers submit to the `serc` partition by default. The training `sbatch` script requests `GPU_MEM:80GB` and `256G` host RAM by default. If Sherlock rejects that feature for your account or partition, list available GPU features with:

```bash
sh_node_feat -p gpu | grep GPU_
```

Then test the lambda-zero LAMSE path separately. This should match pure AMSE behavior while exercising the `--lamse --lamse-lambda 0.0` route:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LOSS_MODE=lamse LAMSE_LAMBDA=0.0 \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

Use `LAMSE_LAMBDA=0.1` only after the pure-AMSE and lambda-zero gates are stable.

For the first nonzero LAMSE gate, use the intended final lambda with a reduced needlet bandlimit to keep host RAM bounded. The Sherlock wrappers default `LAMSE_LMAX` to `32` for `LOSS_MODE=lamse` when it is not set:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LOSS_MODE=lamse LAMSE_LAMBDA=0.1 LAMSE_LMAX=32 \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

```bash
PYTHONPATH=. python train.py \
  --model-checkpoint $PARAMS_DIR/graphcast_small_lamse.000000.npz \
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
--lamse-lambda 0.1 --lamse-lmax 32
```

Do not run longer fine-tuning until the 100-batch lambda-zero and lambda-0.1 gates pass. After those logs look stable, launch the 5000-batch LAMSE run:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LAMSE_LAMBDA=0.1 LAMSE_LMAX=32 \
  bash scripts/submit_final_lamse.sh
```

To continue from an existing 1000-batch LAMSE checkpoint:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LAMSE_LAMBDA=0.1 LAMSE_LMAX=32 START_BATCH=1000 \
  bash scripts/submit_final_lamse.sh
```

## Multi-Model Prediction Comparison

After AMSE-5000 and LAMSE-5000 checkpoints exist, compare ground truth,
pre-finetuned, AMSE, and one or more LAMSE predictions for the same
initialization. Extra checkpoints can be included with
`--extra-checkpoint NAME=PATH`.

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

python3 compare_four_predictions.py \
  --prefinetuned-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --amse-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --lamse-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --extra-checkpoint "amse25000=$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  --extra-checkpoint "lamse0p3=$PARAMS_DIR/graphcast_small_lamse_lam0p3_lmax32.005000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --reference-model amse \
  --out-dir runs/prediction_compare/20220101_lam0p3 \
  --overwrite
```

This writes multi-panel value maps, model-minus-truth error maps, selected
fields in Zarr, weighted scalar metrics, direct pairwise delta/improvement
metrics versus AMSE, and AMSE-style spectral diagnostics when the GPU
spherical harmonic path is available. The direct `*_delta_vs_amse.png` plots
are useful when AMSE and LAMSE error maps look visually identical. Omit any
`--extra-checkpoint` line until that checkpoint exists and has been verified.

For the larger-lambda branch, run a short gate first:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LOSS_MODE=lamse LAMSE_LAMBDA=0.3 LAMSE_LMAX=32 FIRST_STEP_TIMEOUT=3600 \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

If stable, launch the 5000-batch run:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LAMSE_LAMBDA=0.3 LAMSE_LMAX=32 FIRST_STEP_TIMEOUT=3600 \
  bash scripts/submit_final_lamse.sh
```

For the sharper high-L experiment, submit the fixed lambda-0.5, LMAX-127
5000-batch wrapper:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
  bash scripts/submit_final_lamse_lam0p5_lmax127.sh
```

This wrapper uses the automatic `input_nside=64`: `LMAX=127` gives s2fft
bandlimit `128`, which satisfies the HEALPix `L >= 2*nside` constraint. It
defaults to `FIRST_STEP_TIMEOUT=21600`, `WATCHDOG_TIMEOUT=1200`, and
`TIME=48:00:00` because job `26581985` hit the previous three-hour first-step
watchdog while JAX was still compiling. Resume with:

```bash
START_BATCH=1000 \
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
  bash scripts/submit_final_lamse_lam0p5_lmax127.sh
```

The first LAMSE-0.3 compile can be slow on Sherlock. If the log ends with
`Timeout (0:10:00)!` inside JAX/XLA compilation before any `Received ... err=`
training lines, rerun with a larger `FIRST_STEP_TIMEOUT`; that is a watchdog
timeout, not evidence that the loss became nonfinite.

## Full-Year 2022 Aggregate Evaluation

Run the paper-style aggregate scorecard over all 2022 initializations after the
checkpoints have been verified. The default wrapper uses 12-hourly
initializations, 10-day rollouts, and spectrum leads at 6h, 120h, and 240h:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

bash scripts/submit_scorecard_2022_all.sh
```

This submits separate Slurm jobs for:

- `control`: `$PARAMS_DIR/graphcast_small_lamse.000000.npz`
- `amse5000`: `$PARAMS_DIR/graphcast_small_amse.005000.npz`
- `amse25000`: `$PARAMS_DIR/graphcast_small_amse.025000.npz`
- `lamse0p1_lmax32_5000`: `$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz`
- `lamse0p5_lmax127_5000`: `$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz`

Outputs are written under `runs/eval/2022_full_i12/` by default:

- `<model>_score_2022.zarr`
- `<model>_spec_2022.zarr`
- `scorecard_manifest.tsv`

If the 12-hourly run is too expensive, start with daily initializations:

```bash
INIT_INTERVAL=24 bash scripts/submit_scorecard_2022_all.sh
```

After all jobs finish, summarize the deterministic scorecards into one CSV:

```bash
python3 scripts/summarize_scorecards.py \
  --manifest runs/eval/2022_full_i12/scorecard_manifest.tsv \
  --out-csv runs/eval/2022_full_i12/scorecard_summary.csv
```

Use `stat=std` rows as the deterministic error summary; lower values are
better. The spectral Zarrs remain the source for amplitude/coherence analysis.

Leave `CLIMATO_PATH` unset for no-climatology evaluation (`--cpath none`). If a
real climatology zarr is available, pass it as `CLIMATO_PATH=/path/to/climatology.zarr`.
Do not use the shell variable `CPATH` for this; Sherlock modules set `CPATH` to
compiler include directories.

## Hurricane Diagnostics

The paper's hurricane check compares forecast maximum surface wind speed error,
minimum central pressure error, and center-position error by lead time. This
fork has a similar targeted diagnostic in `hurricane_performance_check.py`.
It uses a supplied best-track file, finds each forecast storm center by minimum
MSLP in a storm-centered search window, then scores max 10 m wind, min MSLP,
and center distance.

This is structurally similar to the paper but not identical: our current setup
uses the 1-degree GraphCast Small model and ERA5/WeatherBench2 fields, while the
paper's example used higher-resolution products. Best-track wind is also an
observed intensity estimate, not a grid-scale 1-degree model wind, so expect a
negative wind bias.

Example Hurricane Ian 2022 smoke check:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

python3 hurricane_performance_check.py \
  --track-file "$SCRATCH/graphcast-small-lamse/tracks/hurdat2_atlantic.txt" \
  --storm-name IAN \
  --storm-year 2022 \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --checkpoint "prefinetuned=$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --checkpoint "amse5000=$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --checkpoint "lamse0p1=$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --init-start "19 Sep 2022 00:00" \
  --init-end "27 Sep 2022 00:00" \
  --init-interval-hours 24 \
  --forecast-length 240 \
  --lead-hours 6 12 18 24 36 48 72 96 120 144 168 192 216 240 \
  --truth-source best_track \
  --out-dir runs/hurricane_check/ian2022 \
  --overwrite
```

Outputs:

- `runs/hurricane_check/ian2022/hurricane_metrics.csv`
- `runs/hurricane_check/ian2022/hurricane_mean_errors.csv`
- `runs/hurricane_check/ian2022/hurricane_error_summary.png`

For a model-grid-consistent sanity check, switch to `--truth-source analysis`;
this still uses best-track position as the initial search guide, but compares
wind and pressure against the staged analysis fields instead of best-track
intensity.
