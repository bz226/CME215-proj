# GraphCast Small LAMSE Progress

Updated: 2026-05-24

## Working Rule

After each user prompt in this job, update this file with:

- current state;
- completed actions;
- blockers or caveats;
- the next concrete commands or decisions.

## Current State

- AMSE baseline continuation `START_BATCH=1000 bash scripts/submit_final_amse.sh` completed cleanly to batch 5000 on Sherlock.
- The AMSE training CSV covered 4000 rows, batch 1000 through 4999, with no nonfinite losses. Mean loss dropped modestly from about `0.719` in batches 1000-1499 to about `0.690` in batches 4500-4999.
- Treat the AMSE batch-5000 checkpoint as the current AMSE baseline. The old path may be `params/graphcast_small_amse.005000.npz`; future scripts expect it under `${PARAMS_DIR:-$SCRATCH/graphcast-small-lamse/params}/graphcast_small_amse.005000.npz` unless `CHECKPOINT=...` is set explicitly.
- The 2022 evaluation data staging needed two fixes:
  - use WeatherBench2 source `gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`;
  - select precipitation by the actual output time range for the partial January 2023 buffer.
- January 2022 AMSE-5000 scorecard/spectral smoke evaluation completed:
  - `runs/eval/amse5000_score_jan2022.zarr`
  - `runs/eval/amse5000_spec_jan2022.zarr`
- The Dask `Future.__del__` messages after `... complete` are shutdown cleanup noise, not a failed run.
- Long-run LAMSE training should be run too, using a gated path: lambda-zero and short nonzero LAMSE checks first, then the 5000-batch run.
- The 5000-batch LAMSE wrapper path is now present, executable, and passes `bash -n`.
- The nonzero `LAMSE_LAMBDA=0.1`, `LAMSE_LMAX=32` 5000-batch LAMSE run is user-reported finished. Artifact/log verification is still the next step.
- Large GraphCast checkpoints should now live under scratch by default: `$SCRATCH/graphcast-small-lamse/params`. The scripts also accept `PARAMS_DIR=...` for an explicit alternate scratch path.
- Fine-tuning defaults use initialization dates `1 Jan 2016 00:00` through `31 Dec 2017 18:00`; the staged data can include adjacent 2015-12-31 and 2018-01-01 buffer times needed for inputs/targets, but the effective training init years are 2016-2017.
- Current evaluation data are held out in 2022. January 2022 has completed as the smoke evaluation; full-year 2022 remains the intended main evaluation.
- If shell `set -u` is active, define `PARAMS_DIR` before referencing `$PARAMS_DIR`: `export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"`.
- LAMSE-5000 should use the same `plot_prediction_error.py` workflow as AMSE-5000, with checkpoint `$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz` and output directory `runs/prediction_error/lamse5000_20220101`.
- A four-way comparison script now exists for ground truth, pre-finetuned, AMSE-5000, and LAMSE-5000 predictions.

## Local Code Patches Made

- `plot_prediction_error.py`
  - new utility to run one selected rollout, save prediction/target/error fields to Zarr, and write target/prediction/error PNG maps;
  - supports field aliases such as `z500`, `t850`, `2t`, `10m_wind_speed`, and `msl`;
  - supports selected leads such as `6`, `120`, and `240` hours;
  - now emits a clear install hint if `matplotlib` is missing;
  - materializes saved fields as plain NumPy-backed xarray arrays before `to_zarr`, avoiding `JaxArrayWrapper` serialization errors.
- `compare_four_predictions.py`
  - new utility to compare ground truth, pre-finetuned GraphCast Small, AMSE-5000, and LAMSE-5000 for the same init date, fields, and lead times;
  - writes four-panel value maps, three-panel model-minus-truth error maps, a Zarr bundle of selected fields, and a CSV of weighted bias/RMSE/MAE/error-std/correlation;
  - when GPU spherical harmonic diagnostics are available, also writes AMSE-style spectral amplitude ratio, coherence, amplitude-error, and decorrelation-error by total wavenumber, following the paper's amplitude/correlation decomposition.
- `scripts/requirements_sherlock.txt`
  - added `matplotlib==3.8.3` for `plot_prediction_error.py` PNG output.
- `scripts/run_lamse_training.sh`, `scripts/sbatch_lamse_training.sh`, `scripts/submit_final_lamse.sh`
  - new long-run LAMSE wrappers analogous to the AMSE final-run scripts;
  - checkpoint names include lambda and bandlimit, e.g. `$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz`;
  - default `LAMSE_LAMBDA=0.1`, `LAMSE_LMAX=32`, `FINAL_BATCH=5000`, `CHECKPOINT_EVERY=500`.
  - verified executable and syntax-clean locally with `bash -n`.
- `README_SMALL_LAMSE.md`
  - documented the post-gate 5000-batch LAMSE submission path and aligned the nonzero gate example with `LAMSE_LAMBDA=0.1`.
- Checkpoint path handling
  - training, final-run, 100-batch, and data-staging wrappers now default checkpoint files to `$SCRATCH/graphcast-small-lamse/params` when `SCRATCH` is set;
  - `download_graphcast_small.py` now downloads and caches Hugging Face artifacts under the selected params directory instead of the home HF cache;
  - `prepare_graphcast_small_checkpoint.py`, `inspect_graphcast_checkpoint.py`, and `download_weatherbench2_era5_1deg.py` now respect `PARAMS_DIR`.
  - AMSE/LAMSE `.000000.npz` starting checkpoints are symlink aliases to avoid duplicate full checkpoint copies; trained later checkpoints are still written as real `.npz` files.
- `scripts/download_weatherbench2_era5_1deg.py`
  - default WeatherBench2 source updated to the 2023-01-10 source;
  - precipitation slicing fixed for partial terminal months.
- `scripts/sbatch_fetch_training_data.sh`
  - added `SOURCE=...` override and logging.
- `build_scorecard.py`
  - forces `JAX_PLATFORMS=cuda,cpu` when needed;
  - supports `--cpath none` for no-climatology spectral/scorecard smoke tests;
  - skips climatology-only fields `p_act`, `a_act`, and `acc` when no climatology is supplied;
  - falls back to a single global mask if `hurr_scorecard_mask` is unavailable.
- `build_crpscard.py`
  - forces `JAX_PLATFORMS=cuda,cpu` when needed.

## Immediate Next Steps

1. Verify the finished LAMSE-5000 run artifacts:

```bash
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

ls -lh \
  $PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz \
  $PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz.opt

tail -n 80 logs/gc-lamse-final-*.log

python3 - <<'PY'
import glob
import pandas as pd

paths = sorted(glob.glob("runs/lamse_lam0p1_lmax32_final_*_to_005000.csv"))
print(paths)
for p in paths:
    df = pd.read_csv(p)
    print("\n==", p)
    print("rows", len(df))
    print(df.tail())
    for col in df.select_dtypes("number").columns:
        print(col, "first500", df[col].head(500).mean(), "last500", df[col].tail(500).mean())
PY
```

2. Run the four-way January 2022 visual comparison:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

python3 compare_four_predictions.py \
  --prefinetuned-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --amse-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --lamse-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --out-dir runs/prediction_compare/20220101 \
  --overwrite
```

This writes:

- `runs/prediction_compare/20220101/comparison_fields.zarr`
- `runs/prediction_compare/20220101/comparison_metrics.csv`
- `runs/prediction_compare/20220101/spectral_metrics.csv` if the AMSE-style spherical harmonic diagnostics succeed
- `*_values.png`, `*_errors.png`, and, when available, `*_spectra.png` for each selected field/lead.

3. Run the January 2022 LAMSE-5000 scorecard/spectral smoke evaluation:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
mkdir -p runs/eval

python3 build_scorecard.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --apath $SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022 \
  --norm-factors stats \
  --cpath none \
  --start-date "1 Jan 2022 00:00" \
  --end-date "31 Jan 2022 18:00" \
  --forecast-length 240 \
  --init-interval 24 \
  --spectrum-leads 6 120 240 \
  --to-path runs/eval/lamse5000_score_jan2022.zarr \
  --spectrum-output runs/eval/lamse5000_spec_jan2022.zarr
```

4. Save and plot example LAMSE-5000 prediction-error maps:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

ls -lh "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz"

python3 plot_prediction_error.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --apath $SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022 \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --out-dir runs/prediction_error/lamse5000_20220101 \
  --overwrite
```

This writes:

- `runs/prediction_error/lamse5000_20220101/prediction_error_fields.zarr`
- one PNG per selected LAMSE field and lead.

5. Inspect the AMSE-5000 and LAMSE-5000 January outputs side by side:

```bash
python3 - <<'PY'
import xarray as xr

for p in [
    "runs/eval/amse5000_score_jan2022.zarr",
    "runs/eval/amse5000_spec_jan2022.zarr",
    "runs/eval/lamse5000_score_jan2022.zarr",
    "runs/eval/lamse5000_spec_jan2022.zarr",
]:
    ds = xr.open_zarr(p)
    print("\n==", p)
    print(ds)
PY
```

6. Save and plot example AMSE-5000 prediction-error maps if not already done:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

python3 plot_prediction_error.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --apath $SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022 \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --out-dir runs/prediction_error/amse5000_20220101 \
  --overwrite
```

This writes:

- `runs/prediction_error/amse5000_20220101/prediction_error_fields.zarr`
- one PNG per selected field and lead.

7. Run the same January scorecard/spectral smoke test for the control checkpoint:

```bash
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
mkdir -p runs/eval

python3 build_scorecard.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --apath $SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022 \
  --norm-factors stats \
  --cpath none \
  --start-date "1 Jan 2022 00:00" \
  --end-date "31 Jan 2022 18:00" \
  --forecast-length 240 \
  --init-interval 24 \
  --spectrum-leads 6 120 240 \
  --to-path runs/eval/control_score_jan2022.zarr \
  --spectrum-output runs/eval/control_spec_jan2022.zarr
```

8. Compare AMSE-5000 vs LAMSE-5000 vs control:

- direct scorecard `std` by lead time;
- spectral amplitude ratio and coherence at `6h`, `120h`, and `240h`;
- focus first on `z`, `t`, `2t`, `10u`, `10v`, and `msl`.

9. If January comparison is sane, repeat for:

- `$PARAMS_DIR/graphcast_small_amse.001000.npz` as a mid-training AMSE reference;
- full-year 2022 with `--init-interval 12`;
- full-year 2022 for LAMSE-5000 with `--init-interval 12`.

## Caveats

- `build_scorecard.py` writes summary metrics, not full forecast maps; use `plot_prediction_error.py` for selected full-field forecast/error maps.
- If Sherlock reports `ModuleNotFoundError: No module named 'matplotlib'`, install it in the active venv with `python3 -m pip install matplotlib==3.8.3` or rebuild from the updated requirements file.
- If an older `plot_prediction_error.py` run already wrote PNGs but failed at `prediction_error_fields.zarr` with `JaxArrayWrapper`, rerun after applying the materialization patch and pass `--overwrite`.
- If `compare_four_predictions.py` reports that spectral diagnostics were skipped, the four-way value/error PNGs and scalar metrics are still valid; use `build_scorecard.py --spectrum-leads ...` for the production spectral scorecard.
- The LAMSE-5000 completion is currently user-reported in this file; confirm with the final checkpoint, `.opt`, CSV tail, and log `Exiting` lines before treating it as verified.
- `bash: PARAMS_DIR: unbound variable` means `set -u` is active and `$PARAMS_DIR` was referenced before export; run the `export PARAMS_DIR=...` line first or use `PARAMS_DIR=/scratch/users/zbao7/graphcast-small-lamse/params`.
- If an existing resume checkpoint only exists in the project `params/` directory, copy it once into `$SCRATCH/graphcast-small-lamse/params` or submit with `CHECKPOINT=/old/path/...` for that resume.
- A real climatology zarr is still needed for ACC/activity metrics matching the paper exactly.
- Full paper-style lagged-ensemble CRPS/eRMSE/SER still needs `build_crpscard.py` cleanup for this WeatherBench2 path.
