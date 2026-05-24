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
- Treat `params/graphcast_small_amse.005000.npz` as the current AMSE baseline checkpoint.
- The 2022 evaluation data staging needed two fixes:
  - use WeatherBench2 source `gs://weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr`;
  - select precipitation by the actual output time range for the partial January 2023 buffer.
- January 2022 AMSE-5000 scorecard/spectral smoke evaluation completed:
  - `runs/eval/amse5000_score_jan2022.zarr`
  - `runs/eval/amse5000_spec_jan2022.zarr`
- The Dask `Future.__del__` messages after `... complete` are shutdown cleanup noise, not a failed run.

## Local Code Patches Made

- `plot_prediction_error.py`
  - new utility to run one selected rollout, save prediction/target/error fields to Zarr, and write target/prediction/error PNG maps;
  - supports field aliases such as `z500`, `t850`, `2t`, `10m_wind_speed`, and `msl`;
  - supports selected leads such as `6`, `120`, and `240` hours.
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

1. Inspect the AMSE-5000 January outputs:

```bash
python3 - <<'PY'
import xarray as xr

for p in [
    "runs/eval/amse5000_score_jan2022.zarr",
    "runs/eval/amse5000_spec_jan2022.zarr",
]:
    ds = xr.open_zarr(p)
    print("\n==", p)
    print(ds)
PY
```

2. Save and plot example AMSE-5000 prediction-error maps:

```bash
export JAX_PLATFORMS=cuda,cpu

python3 plot_prediction_error.py \
  --model-checkpoint params/graphcast_small_amse.005000.npz \
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

3. Run the same January scorecard/spectral smoke test for the control checkpoint:

```bash
export JAX_PLATFORMS=cuda,cpu
mkdir -p runs/eval

python3 build_scorecard.py \
  --model-checkpoint params/graphcast_small_lamse.000000.npz \
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

4. Compare AMSE-5000 vs control:

- direct scorecard `std` by lead time;
- spectral amplitude ratio and coherence at `6h`, `120h`, and `240h`;
- focus first on `z`, `t`, `2t`, `10u`, `10v`, and `msl`.

5. If January comparison is sane, repeat for:

- `params/graphcast_small_amse.001000.npz` as a mid-training AMSE reference;
- full-year 2022 with `--init-interval 12`;
- later, LAMSE lambda-zero and nonzero checkpoints once those training gates exist.

## Caveats

- `build_scorecard.py` writes summary metrics, not full forecast maps; use `plot_prediction_error.py` for selected full-field forecast/error maps.
- A real climatology zarr is still needed for ACC/activity metrics matching the paper exactly.
- Full paper-style lagged-ensemble CRPS/eRMSE/SER still needs `build_crpscard.py` cleanup for this WeatherBench2 path.
