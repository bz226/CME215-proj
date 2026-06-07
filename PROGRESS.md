# GraphCast Small LAMSE Progress

Updated: 2026-06-07

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
- The AMSE batch-25000 run is user-reported finished. Treat `$PARAMS_DIR/graphcast_small_amse.025000.npz` as a candidate long-AMSE checkpoint only after verifying the checkpoint, `.opt`, CSV tail, finite losses, and clean log exit.
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
- Current AMSE/LAMSE fine-tuning is single-step training: `scripts/run_amse_training.sh` and `scripts/run_lamse_training.sh` default `FORECAST_LENGTH=1`, and `train.py --forecast-length` is measured in 6-hour model steps. Thus each training example backpropagates through a 6h forecast. The January 2022 evaluation/plotting is multi-step rollout: `--forecast-length 240` in the evaluation scripts means 240 forecast hours, or 40 autoregressive 6h steps.
- Current evaluation data are held out in 2022. January 2022 has completed as the smoke evaluation; full-year 2022 remains the intended main evaluation.
- Paper-aligned evaluation should prioritize aggregate verification over calendar year 2022. The paper uses single-event maps as illustrative case studies, but its main statistical and spectral conclusions are aggregated over 2022 forecast initializations. Treat the current `2022-01-01 00:00` plots as qualitative diagnostics only, not as evidence for final AMSE/LAMSE ranking.
- Course-project evaluation recommendation: January-only forecasts are acceptable as smoke tests and qualitative examples, but not strong enough as the main result because one initialization/month can be cherry-picked and does not establish model ranking. Use the full-year 2022 aggregate scorecard/spectral plots as the primary result, then add a Hurricane Ian 2022 tracking/intensity figure as a compelling case study if time permits.
- Current full-year `10m_wind_speed` aggregate improvement plot versus AMSE-5000 shows the pre-finetuned checkpoint has substantially lower mean `std` error at medium/long leads, while AMSE-25000 and both LAMSE variants are near or below AMSE-5000. This means AMSE/LAMSE fine-tuning is not improving this aggregate 10m wind metric; check other variables and raw error/ranking CSVs before making a final model-ranking claim.
- The uploaded full-year 2022 aggregate plots for `2t`, `t850`, `msl`, and `10m_wind_speed` show the same broad deterministic-error pattern: pre-finetuned has lower weighted `std` error than AMSE-5000 through most medium/long leads, while AMSE-25000 is only marginally different from AMSE-5000 and LAMSE variants are mostly slightly worse. This makes the main aggregate finding negative for AMSE/LAMSE on standard error metrics, unless spectral diagnostics or a targeted case study show a different tradeoff.
- The Hurricane Ian 2022 diagnostic plot shows a targeted counterpoint to the full-year aggregate: pre-finetuned has the weakest long-lead storm intensity, with increasingly negative 10 m wind error and positive minimum-pressure error, while LAMSE-0.1/0.5 reduce long-lead intensity bias and LAMSE-0.5 gives the lowest day-10 position error in this case. Treat this as a case-study benefit for localized cyclone structure, not a replacement for the full-year aggregate result.
- The Figure-5-style Hurricane Ian map is not very meaningful as a visual analogue to the paper because the current setup uses 1-degree GraphCast Small/ERA5 fields. The storm core is only a few grid cells wide, the wind shading is blocky, and small center/pressure differences can be dominated by grid resolution. In the report, mention this explicitly and use the map only as a qualitative limitation/case-study figure, not as quantitative evidence.
- Current progress is enough for a 5-page one-month course-project report if framed as a constrained replication/extension rather than a successful performance improvement. The strongest story is: implemented AMSE/LAMSE fine-tuning for GraphCast Small, ran multiple checkpoints, evaluated on held-out 2022 full-year aggregates, found that AMSE/LAMSE worsened standard deterministic error versus pre-finetuned, but saw a localized Hurricane Ian case-study tradeoff. Report caveats should emphasize 1-degree resolution, single-step fine-tuning, limited training data/budget, and incomplete parity with the paper's high-resolution/full methodology.
- `.gitignore` now ignores generated `*.png` plot artifacts so evaluation figures are not accidentally staged.
- `.gitignore` now also ignores the temporary top-level `plots/` directory for local result storage.
- The top-level `plots/` folder now exists locally and `git status --ignored plots` reports it as ignored.
- Drafted `course_report.tex` as a compact LaTeX course report using the current plots. It follows the requested structure: introduction/background, existing-method difficulty, problem statement, soft physics constraints, results/discussion, open questions, and future work. Local validation checked word count, figure paths, citation labels, and rough brace balance; local `pdflatex`/`latexmk` are not installed, so PDF compilation still needs a LaTeX environment.
- Created the Overleaf upload bundle at `plots/course_report_overleaf_bundle.zip`. It contains `course_report.tex` and the five referenced PNGs under `plots/`, preserving the paths used by `\includegraphics`; the zip is about 728 KB and is ignored by Git through the top-level `plots/` rule.
- Revised `course_report.tex` against the style/content of the supplied `graphcast_lamse_project_draft (1).tex` and `cme_215_proj.pdf`: added the abstract, notation macros, MSE amplitude argument, PSD/coherence/AMSE definitions, and retained the final full-year negative-result narrative. Refreshed `plots/course_report_overleaf_bundle.zip`; `unzip -t` passes and the bundle is about 729 KB.
- MSE fine-tuning is now the next useful control experiment. It should use the same 2016-2017 single-step training setup as AMSE/LAMSE, but with neither `--spectral-amse` nor `--lamse`, producing `$PARAMS_DIR/graphcast_small_mse.005000.npz` for held-out 2022 comparison.
- MSE aggregate comparison graphs are now wired through `scripts/plot_mse_scorecard_comparison.sh`. After the MSE scorecard finishes, run it to build `scorecard_summary_with_mse.csv` and `plots_with_mse/*_error_by_lead.png` / `*_improvement_vs_amse5000.png`.
- Report-style plots with MSE are now wired through `scripts/build_report_plots_with_mse.sh`. Its default aggregate step rewrites the top-level `plots/*_std_mean_*.png` files used by the report with MSE-5000 replacing AMSE-25000. Optional `RUN_SINGLE_CASE=1` / `RUN_HURRICANE=1` modes regenerate the January showcase/spectral and Hurricane Ian diagnostics with the same replacement; set `INCLUDE_AMSE25000=1` only to add the long-AMSE ablation back.
- Latest MSE-inclusive Hurricane Ian plot appears to be the `best_track`-truth version: all models have large negative maximum-wind errors and positive pressure errors, consistent with comparing 1-degree grid-cell storm intensity to observed best-track intensity. Interpret the plot mainly by relative ranking, not absolute intensity bias.
- Hurricane Ian tracking/intensity plotting has been made less error-prone: `hurricane_performance_check.py` now defaults to `--truth-source analysis`, prints a warning for `best_track`, and writes the truth source into the plot title. For the report, rerun Ian diagnostics with analysis truth; keep best-track truth only as a limitation/sensitivity figure because 1-degree storm-core intensity is not comparable to observed best-track intensity.
- The MSE-inclusive report wrapper now also writes a combined full-year deterministic-error overview at `plots/deterministic_error_summary.png`, using the default report model set with MSE-5000 replacing AMSE-25000. Plot titles should avoid wording such as "paper-style"; the Hurricane Ian title is now plain `IAN 2022 hurricane diagnostics (...)`.
- Plot label cleanup: `compare_four_predictions.py` now formats LAMSE ids like `lamse0p1_lmax32` and `lamse0p5_lmax127` as report-ready `LAMSE-0.1-LMAX32` and `LAMSE-0.5-LMAX127`; this also fixes labels in hurricane diagnostics because they use the shared label helper.
- Git hygiene note: `course_report.tex`, `plots.zip`, and `.DS_Store` should not be included in pushes; they are local/generated artifacts and are ignored.
- If shell `set -u` is active, define `PARAMS_DIR` before referencing `$PARAMS_DIR`: `export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"`.
- LAMSE-5000 should use the same `plot_prediction_error.py` workflow as AMSE-5000, with checkpoint `$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz` and output directory `runs/prediction_error/lamse5000_20220101`.
- A multi-model comparison script now exists for ground truth, pre-finetuned, AMSE-5000, LAMSE-0.1-5000, and optional extra checkpoints such as LAMSE-0.3-5000.
- A LAMSE-0.3 comparison artifact exists in `/Users/bobby/Downloads/20220101_lam0p3.zip`. Its `pairwise_metrics.csv` compares pre-finetuned, LAMSE-0.1, and LAMSE-0.3 against AMSE-5000 for one January 2022 initialization.
- The first observed `LAMSE_LAMBDA=0.3`, `LAMSE_LMAX=32` final job, `25960661`, did not train. Its log reached LAMSE geometry construction and started the first gradient compile, then `faulthandler` killed the process with `Timeout (0:10:00)!` inside JAX/XLA compilation before any `Received ... err=` lines, `Exiting` line, or final checkpoint. Treat this as a watchdog/compile-time failure, not a nonfinite-loss or model-quality result.
- Current uploaded 10m wind speed comparison plots show no significant visible difference between AMSE-5000 and LAMSE-0.1-5000 at 6h, 120h, or 240h for the `2022-01-01 00:00` initialization. AMSE/LAMSE spectra nearly overlap; the clearer contrast is pre-finetuned under-amplification at higher wavenumbers. At 240h, AMSE and LAMSE both retain more high-wavenumber amplitude than pre-finetuned, but coherence is low/noisy and does not show a stable LAMSE advantage. Do not overclaim LAMSE-0.1 improvement from these plots alone.
- Current uploaded AMSE-25000 comparison plots for 10m wind speed at 6h and 240h show AMSE-25000 is very close to AMSE-5000 and LAMSE-0.1-5000. At 6h, direct differences versus AMSE-5000 are tiny and mostly noisy. At 240h, AMSE-25000 differs spatially from AMSE-5000 but the absolute-error improvement map is mixed; the spectra show similar high-wavenumber amplitude retention to AMSE-5000/LAMSE-0.1, still clearly better than pre-finetuned. Do not claim AMSE-25000 is better without checking `pairwise_metrics.csv` and scorecard aggregates.
- LAMSE-0.3 does not look better than AMSE-5000 in the provided January comparison. In `pairwise_metrics.csv`, LAMSE-0.3 wins only 3/15 RMSE cases and 5/15 MAE cases versus AMSE-5000; for 10m wind speed it is slightly worse than AMSE-5000 at 6h, 120h, and 240h. Spectrally, LAMSE-0.3 changes high-wavenumber amplitude but does not improve coherence. Do not prioritize more lambda-0.3 training before full-year aggregate evaluation or a revised loss/weighting choice.
- Latest code-reading note: `needlet_construction.py` is the static setup layer for LAMSE needlets, including finite spectral filters, HEALPix center grids, coarse localization cells, and Gaussian local weights. `needlet_transform.py` is the runtime transform layer, interpolating lat-lon fields to HEALPix samples, computing `s2fft` spherical harmonic coefficients, applying each needlet filter, and returning per-scale HEALPix needlet coefficients.
- Current hypothesis for AMSE/LAMSE indistinguishability: the active LAMSE setup may be too close to AMSE because `LAMSE_LMAX=32` only sees degrees up to 32, `lambda=0.1` leaves the hybrid loss 90% AMSE, and the default localization radius `locality_c * B^-j` is broad (`locality_c=3`, `B=2`; finest scale at `j=5` has sigma about 5.4 degrees and FWHM about 12.6 degrees). Before another long LAMSE run, prefer a short sweep over higher `L` and smaller `locality_c`, with diagnostics comparing `amse_total` and `lamse_total` scales.
- AMSE and LAMSE are not using the same active spectral truncation in the current training wrappers. AMSE builds spectral coefficients from the full model latitude grid with `nspec = nlat - 1`, which is about degree 180 for the 1-degree GraphCast Small grid. LAMSE accepts `--lamse-lmax` as an inclusive maximum degree; `scripts/submit_final_lamse.sh` defaults this to `32`, so the current LAMSE loss is much more low-pass than AMSE. If `--lamse-lmax` is omitted entirely, LAMSE falls back to `len(latitudes) - 1`, matching AMSE's latitude-grid degree limit, but that is not what the current Sherlock LAMSE runs used.
- A paper-style hurricane diagnostic script now exists: `hurricane_performance_check.py`. It reads HURDAT2 or CSV best-track data, runs one or more checkpoints, finds each predicted storm center by minimum MSLP in a best-track-centered search window, scores maximum 10 m wind error, minimum central pressure error, and center-position error by lead time, and writes `hurricane_metrics.csv`, `hurricane_mean_errors.csv`, and `hurricane_error_summary.png`. Treat this as structurally similar to the paper's hurricane check but not identical, because our setup uses 1-degree GraphCast Small/ERA5 and best-track wind is not a grid-scale 1-degree model wind.
- A fixed high-L LAMSE submit wrapper now exists: `scripts/submit_final_lamse_lam0p5_lmax96.sh`. It delegates to `scripts/submit_final_lamse.sh` with `LAMSE_LAMBDA=0.5`, `LAMSE_LMAX=96`, `LAMSE_TAG=lam0p5_lmax96`, `FINAL_BATCH=5000`, `FIRST_STEP_TIMEOUT=7200`, `WATCHDOG_TIMEOUT=300`, and `TIME=36:00:00`.
- The earlier high-L LAMSE command used `ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 bash scripts/submit_final_lamse_lam0p5_lmax96.sh`. Run high-L training from the Sherlock project root after activating the project environment, and use the 2016-2017 training data path, not the held-out `_2022` evaluation path.
- Job `26501802` for `LAMSE_LAMBDA=0.5`, `LAMSE_LMAX=96` failed before training. It reached `Building LAMSE HEALPix/needlet geometry` and raised `ValueError: LAMSE input_nside=64 is too high for s2fft bandlimit L=97; HEALPix requires L >= 2*nside`. There were no `Received ... err=` lines, no optimizer updates, and no final checkpoint, so this is a geometry configuration failure, not a numerical training result. The fix is to rerun the same wrapper after the `LAMSE_INPUT_NSIDE=32` patch.
- Since a higher maximum degree is acceptable, prefer the new `scripts/submit_final_lamse_lam0p5_lmax127.sh` wrapper over the LMAX-96 retry. `LMAX=127` gives bandlimit 128, so automatic `input_nside=64` satisfies the s2fft HEALPix constraint without lowering input nside. This run is heavier and now defaults to `FIRST_STEP_TIMEOUT=21600`, `WATCHDOG_TIMEOUT=1200`, and `TIME=48:00:00`.
- Job `26581985` for `LAMSE_LAMBDA=0.5`, `LAMSE_LMAX=127` passed LAMSE geometry construction and built the training sample list, then timed out after `Timeout (3:00:00)!` inside JAX `backend_compile` during the first `grad_update`. There were no `Received ... err=` lines, optimizer updates, or final checkpoint. This is a first-compile watchdog failure, not a numerical training result. The `lmax127` wrapper now defaults to `FIRST_STEP_TIMEOUT=21600` and `WATCHDOG_TIMEOUT=1200`.
- The rerun of `LAMSE_LAMBDA=0.5`, `LAMSE_LMAX=127` is user-reported finished. A newer local downloaded `gc-lamse-final-*.log` is not present yet; verify on Sherlock before treating it as a validated training result. Expected artifacts are `$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz`, `$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz.opt`, and `runs/lamse_lam0p5_lmax127_final_000000_to_005000.csv`.
- The uploaded `20220101_lam0p5_lmax127.zip` comparison artifact contains the expected maps, `comparison_metrics.csv`, `pairwise_metrics.csv`, `spectral_metrics.csv`, and Zarr bundle. For this single `2022-01-01 00:00` case, `LAMSE_LAMBDA=0.5`, `LMAX=127` improves over AMSE-5000 in 6/15 RMSE rows and 8/15 MAE rows, with mean relative RMSE improvement still slightly negative at about `-0.28%`. It wins RMSE for `10m_wind_speed`, `t850`, and `z500` at 240h, and `2t` at 120h/240h, but it is worse than AMSE-5000 for all five 6h rows and for MSL at all three leads. Treat this as a mixed single-case result, not evidence of a robust LAMSE advantage.

## Local Code Patches Made

- `plot_prediction_error.py`
  - new utility to run one selected rollout, save prediction/target/error fields to Zarr, and write target/prediction/error PNG maps;
  - supports field aliases such as `z500`, `t850`, `2t`, `10m_wind_speed`, and `msl`;
  - supports selected leads such as `6`, `120`, and `240` hours;
  - now emits a clear install hint if `matplotlib` is missing;
  - materializes saved fields as plain NumPy-backed xarray arrays before `to_zarr`, avoiding `JaxArrayWrapper` serialization errors.
- `compare_four_predictions.py`
  - new utility to compare ground truth, pre-finetuned GraphCast Small, AMSE-5000, LAMSE-0.1-5000, and optional extra checkpoints for the same init date, fields, and lead times;
  - accepts `--extra-checkpoint NAME=PATH`, so a larger-lambda checkpoint can be included in the same comparison run;
  - labels extra AMSE checkpoints named like `amse25000` as `AMSE-25000` in plots and CSV outputs;
  - writes multi-panel value maps, model-minus-truth error maps, direct `model - reference` delta plots, reference-relative absolute-error improvement plots, a Zarr bundle of selected fields, and CSVs of weighted scalar and pairwise metrics;
  - defaults the pairwise reference to `--reference-model amse`, which is the useful view for distinguishing LAMSE variants from the AMSE baseline;
  - when GPU spherical harmonic diagnostics are available, also writes AMSE-style spectral amplitude ratio, coherence, amplitude-error, and decorrelation-error by total wavenumber, following the paper's amplitude/correlation decomposition.
- `hurricane_performance_check.py`
  - new paper-style tropical cyclone diagnostic using a supplied HURDAT2/CSV track file;
  - supports multiple checkpoints through repeated `--checkpoint NAME=PATH`;
  - writes per-case and lead-aggregated wind, pressure, and position-error metrics plus a three-panel lead-time summary plot.
- `plot_hurricane_figure5.py`
  - new Figure-5-style Hurricane Ian map utility;
  - runs one forecast initialization, crops around the best-track storm center, and plots shaded 10 m wind speed plus MSLP contours for the analysis target and selected checkpoint forecasts;
  - defaults to the paper-style 5-day Ian case: init `23 Sep 2022 12:00`, lead `120h`, valid `28 Sep 2022 12:00`.
- `scripts/sbatch_scorecard_2022.sh`, `scripts/submit_scorecard_2022_all.sh`
  - new Sherlock wrappers for full-year 2022 aggregate scorecard/spectral evaluation;
  - default to 12-hourly 2022 initializations, 10-day rollouts, and spectrum leads `6 120 240`;
  - submit separate jobs for control, AMSE-5000, AMSE-25000, LAMSE-0.1-LMAX32-5000, and LAMSE-0.5-LMAX127-5000;
  - climatology override is `CLIMATO_PATH=...`, not `CPATH=...`, because Sherlock modules set `CPATH` to compiler include directories.
- `scripts/summarize_scorecards.py`
  - new post-processing utility to summarize full-year scorecard Zarr outputs into one CSV table by model, variable, stat, lead time, and remaining coordinates such as pressure level.
  - made compatible with older Sherlock `python3` by removing `from __future__ import annotations` and modern builtin generic annotations.
  - now loads each compact scorecard Zarr before reducing, avoiding Dask's unsupported full-axis `nanmedian` path.
  - now bypasses xarray reductions for summary statistics and computes mean/median/count with NumPy after materializing each selected slice, avoiding Dask `nanmedian` even when arrays remain Dask-backed.
- `scripts/plot_scorecard_summary.py`
  - new full-year aggregate plotting helper for `scorecard_summary.csv`;
  - writes per-field error-by-lead PNGs, AMSE-relative percent-improvement PNGs, and per-lead ranking CSVs for selected variables/levels.
  - forces Matplotlib's noninteractive `Agg` backend to avoid macOS/Python.app GUI backend crashes and work cleanly on Sherlock.
  - now labels `mse5000` as `MSE-5000` and uses a stable model order for aggregate comparison plots.
- `scripts/plot_mse_scorecard_comparison.sh`
  - new convenience wrapper that rebuilds a summary CSV from existing scorecard Zarr stores and plots aggregate comparison graphs with MSE-5000 included;
  - reads Zarr paths directly instead of relying on `scorecard_manifest.tsv`, so it still works if a partial MSE-only scorecard submission rewrote the manifest.
- `scripts/build_report_plots_with_mse.sh`
  - report-oriented plot wrapper that regenerates the existing top-level report plot filenames with MSE-5000 replacing AMSE-25000;
  - optionally regenerates the January single-case value/error/spectral plots and Hurricane Ian summary with the same replacement if run on Sherlock with the MSE checkpoint available;
  - supports `INCLUDE_AMSE25000=1` and `INCLUDE_LAMSE0P3=1` for opt-in ablation plots.
- `plot_hurricane_figure5.py`
  - labels `mse5000` as `MSE-5000` when included in Figure-5-style storm maps.
- `scripts/requirements_sherlock.txt`
  - added `matplotlib==3.8.3` for `plot_prediction_error.py` PNG output.
- `scripts/run_mse_training.sh`, `scripts/sbatch_mse_training.sh`, `scripts/submit_final_mse.sh`
  - new conventional weighted-MSE control wrappers analogous to the AMSE final-run scripts;
  - intentionally pass neither `--spectral-amse` nor `--lamse`, selecting `train.py`'s built-in spatial MSE loss;
  - save checkpoints as `$PARAMS_DIR/graphcast_small_mse.<batch>.npz` and default the final run to 5000 batches with 500-batch checkpoint cadence.
- MSE evaluation plumbing
  - `scripts/submit_scorecard_2022_all.sh` supports `INCLUDE_MSE=1` to add `mse5000` to the full-year scorecard model set without breaking existing runs before the checkpoint exists;
  - `compare_four_predictions.py` labels `mse5000` as `MSE-5000` in plots and CSV outputs;
  - `README_SMALL_LAMSE.md` documents the MSE control submission path and includes it in the multi-model comparison example.
- `scripts/run_lamse_training.sh`, `scripts/sbatch_lamse_training.sh`, `scripts/submit_final_lamse.sh`
  - new long-run LAMSE wrappers analogous to the AMSE final-run scripts;
  - checkpoint names include lambda and bandlimit, e.g. `$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz`;
  - default `LAMSE_LAMBDA=0.1`, `LAMSE_LMAX=32`, `FINAL_BATCH=5000`, `CHECKPOINT_EVERY=500`.
  - verified executable and syntax-clean locally with `bash -n`.
- `scripts/submit_final_lamse_lam0p5_lmax96.sh`
  - fixed convenience wrapper for the 5000-batch `lambda=0.5`, `LMAX=96` LAMSE experiment;
  - defaults to `LAMSE_INPUT_NSIDE=32`, a longer first-compile timeout, and a 36-hour allocation because the high-L graph is expected to compile more slowly than LMAX-32.
- `scripts/submit_final_lamse_lam0p5_lmax127.sh`
  - fixed convenience wrapper for the preferred 5000-batch `lambda=0.5`, `LMAX=127` LAMSE experiment;
  - leaves `LAMSE_INPUT_NSIDE` unset so the automatic `input_nside=64` path is used, and defaults to a six-hour first-step timeout plus a 48-hour allocation.
- LAMSE training CLI/scripts
  - added `--lamse-input-nside` and `LAMSE_INPUT_NSIDE` plumbing so high-L experiments can lower the HEALPix input nside without changing the requested spectral max degree.
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
- `train.py`, training runners, and Sherlock submit wrappers
  - first-step watchdog is now configurable with `--first-step-timeout` / `FIRST_STEP_TIMEOUT`, defaulting to 1800 seconds instead of the old 600-second first-compile timeout;
  - steady-state watchdog remains configurable with `--watchdog-timeout` / `WATCHDOG_TIMEOUT`, defaulting to 180 seconds.

## Immediate Next Steps

1. Verify the finished AMSE-25000 run artifacts:

```bash
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

ls -lh \
  "$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  "$PARAMS_DIR/graphcast_small_amse.025000.npz.opt"

tail -n 120 logs/gc-amse-final-*.log

python3 - <<'PY'
import glob
import numpy as np
import pandas as pd

paths = sorted(glob.glob("runs/amse_final_*_to_025000.csv"))
print(paths)
for p in paths:
    df = pd.read_csv(p)
    numeric = df.select_dtypes("number")
    print("\n==", p)
    print("rows", len(df))
    print(df.tail())
    print("nonfinite rows", (~np.isfinite(numeric)).any(axis=1).sum())
    for col in numeric.columns:
        print(col, "first500", df[col].head(500).mean(), "last500", df[col].tail(500).mean())
PY
```

2. Submit the full-year 2022 aggregate scorecard/spectral evaluation:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

bash scripts/submit_scorecard_2022_all.sh
```

Defaults:

- `INIT_INTERVAL=12` for 12-hourly 2022 initializations;
- `FORECAST_LENGTH=240` for 10-day rollouts;
- `SPECTRUM_LEADS="6 120 240"`;
- `CLIMATO_PATH=none` for no-climatology scorecard fields;
- output root `runs/eval/2022_full_i12`.

Use `INIT_INTERVAL=24 bash scripts/submit_scorecard_2022_all.sh` for a cheaper daily-initialization aggregate if the 12-hourly run is too expensive.

After all `gc-score-*` jobs finish:

```bash
python3 scripts/summarize_scorecards.py \
  --manifest runs/eval/2022_full_i12/scorecard_manifest.tsv \
  --out-csv runs/eval/2022_full_i12/scorecard_summary.csv
```

Interpret `stat=std` rows as deterministic error summaries; lower is better. Keep the spectral Zarrs for amplitude/coherence analysis.

Plot the main full-year aggregate curves:

```bash
python3 scripts/plot_scorecard_summary.py \
  --summary-csv runs/eval/2022_full_i12/scorecard_summary.csv \
  --out-dir runs/eval/2022_full_i12/plots \
  --fields z:500 t:850 2t 10m_wind_speed msl \
  --stat std \
  --metric mean \
  --reference-model amse5000
```

For the retry root, replace `2022_full_i12` with `2022_full_i12_retry`.

3. Run the Hurricane Ian 2022 paper-style case study using the same model set as the full-year aggregate. Prefer `--truth-source analysis` for the main course-project plot because it compares to the staged 1-degree analysis fields; use `--truth-source best_track` as an optional observed-track/intensity sensitivity check.

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
export TRACK_DIR="${SCRATCH:?Set SCRATCH}/graphcast-small-lamse/tracks"
mkdir -p "${TRACK_DIR}"

# If this file is not already present, download the current Atlantic HURDAT2 text
# file from the NHC HURDAT2 page and save it as:
#   ${TRACK_DIR}/hurdat2_atlantic.txt

python3 hurricane_performance_check.py \
  --track-file "${TRACK_DIR}/hurdat2_atlantic.txt" \
  --storm-name IAN \
  --storm-year 2022 \
  --apath "${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --checkpoint "prefinetuned=${PARAMS_DIR}/graphcast_small_lamse.000000.npz" \
  --checkpoint "amse5000=${PARAMS_DIR}/graphcast_small_amse.005000.npz" \
  --checkpoint "amse25000=${PARAMS_DIR}/graphcast_small_amse.025000.npz" \
  --checkpoint "lamse0p1_lmax32=${PARAMS_DIR}/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --checkpoint "lamse0p5_lmax127=${PARAMS_DIR}/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
  --init-start "19 Sep 2022 00:00" \
  --init-end "27 Sep 2022 00:00" \
  --init-interval-hours 24 \
  --forecast-length 240 \
  --lead-hours 6 12 18 24 36 48 72 96 120 144 168 192 216 240 \
  --truth-source analysis \
  --out-dir runs/hurricane_check/ian2022_analysis \
  --overwrite
```

4. Run the Figure-5-style Ian map for the same event as the paper: valid `28 Sep 2022 12:00 UTC`, using a 5-day forecast initialized `23 Sep 2022 12:00 UTC`.

```bash
python3 plot_hurricane_figure5.py \
  --track-file "${TRACK_DIR}/hurdat2_atlantic.txt" \
  --storm-name IAN \
  --storm-year 2022 \
  --apath "${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --checkpoint "prefinetuned=${PARAMS_DIR}/graphcast_small_lamse.000000.npz" \
  --checkpoint "amse5000=${PARAMS_DIR}/graphcast_small_amse.005000.npz" \
  --checkpoint "lamse0p5_lmax127=${PARAMS_DIR}/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
  --init-date "23 Sep 2022 12:00" \
  --lead-hours 120 \
  --forecast-length 120 \
  --lat-min 24 \
  --lat-max 28 \
  --lon-min -85 \
  --lon-max -81 \
  --wind-vmin 18 \
  --wind-vmax 40 \
  --out-dir runs/hurricane_check/ian2022_figure5 \
  --overwrite
```

This writes `runs/hurricane_check/ian2022_figure5/ian2022_figure5.png`. Add more `--checkpoint` lines, such as `amse25000` or `lamse0p1_lmax32`, if a wider panel stack is acceptable.

5. Run a January 2022 AMSE-25000 scorecard/spectral smoke evaluation, then promote the same scorecard path to all of 2022 before making final claims:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
mkdir -p runs/eval

python3 build_scorecard.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --cpath none \
  --start-date "1 Jan 2022 00:00" \
  --end-date "31 Jan 2022 18:00" \
  --forecast-length 240 \
  --init-interval 24 \
  --spectrum-leads 6 120 240 \
  --to-path runs/eval/amse25000_score_jan2022.zarr \
  --spectrum-output runs/eval/amse25000_spec_jan2022.zarr
```

For the paper-aligned full-year aggregate, switch the end date and output names:

```bash
python3 build_scorecard.py \
  --model-checkpoint "$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --cpath none \
  --start-date "1 Jan 2022 00:00" \
  --end-date "31 Dec 2022 18:00" \
  --forecast-length 240 \
  --init-interval 24 \
  --spectrum-leads 6 120 240 \
  --to-path runs/eval/amse25000_score_2022.zarr \
  --spectrum-output runs/eval/amse25000_spec_2022.zarr
```

4. Include AMSE-25000 in the combined visual comparison:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
mkdir -p runs/prediction_compare

python3 compare_four_predictions.py \
  --prefinetuned-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --amse-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --lamse-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --extra-checkpoint "amse25000=$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --reference-model amse \
  --out-dir runs/prediction_compare/20220101_amse25000 \
  --overwrite
```

5. Verify the finished LAMSE-5000 run artifacts:

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

6. Verify the finished high-L `lambda=0.5`, `LMAX=127` 5000-batch LAMSE run:

```bash
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

ls -lh \
  "$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
  "$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz.opt"

tail -n 120 logs/gc-lamse-final-*.log

python3 - <<'PY'
import glob
import numpy as np
import pandas as pd

paths = sorted(glob.glob("runs/lamse_lam0p5_lmax127_final_*_to_005000.csv"))
print(paths)
for p in paths:
    df = pd.read_csv(p)
    numeric = df.select_dtypes("number")
    print("\n==", p)
    print("rows", len(df))
    print(df.tail())
    print("nonfinite rows", (~np.isfinite(numeric)).any(axis=1).sum())
    for col in numeric.columns:
        print(col, "first500", df[col].head(500).mean(), "last500", df[col].tail(500).mean())
PY
```

Treat the run as verified only if the final checkpoint and `.opt` exist, the CSV reaches batch 4999/5000 without nonfinite values, and the log has clean final checkpoint/exit lines.

7. Run the combined January 2022 comparison including the high-L `lambda=0.5`, `LMAX=127` checkpoint:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"
mkdir -p runs/prediction_compare

python3 compare_four_predictions.py \
  --prefinetuned-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --amse-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --lamse-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
  --extra-checkpoint "amse25000=$PARAMS_DIR/graphcast_small_amse.025000.npz" \
  --extra-checkpoint "lamse0p5_lmax127=$PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
  --apath "$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2_2022" \
  --norm-factors stats \
  --init-date "1 Jan 2022 00:00" \
  --forecast-length 240 \
  --lead-hours 6 120 240 \
  --fields z500 t850 2t 10m_wind_speed msl \
  --reference-model amse \
  --out-dir runs/prediction_compare/20220101_lam0p5_lmax127 \
  --overwrite
```

This writes `comparison_fields.zarr`, `comparison_metrics.csv`, `pairwise_metrics.csv`, optional `spectral_metrics.csv`, and value/error/direct-delta/spectral PNGs under `runs/prediction_compare/20220101_lam0p5_lmax127`.

8. Run a short larger-lambda LAMSE gate:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LOSS_MODE=lamse LAMSE_LAMBDA=0.3 LAMSE_LMAX=32 FIRST_STEP_TIMEOUT=3600 \
  sbatch scripts/sbatch_lamse_100_batches.sh
```

Check the resulting `logs/gc-lamse-100-*.log` and `runs/lamse_0.3_job_*.csv` for finite losses and clean `Exiting` lines.

9. If the gate is stable, launch the 5000-batch larger-lambda LAMSE run:

```bash
ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
LAMSE_LAMBDA=0.3 LAMSE_LMAX=32 FIRST_STEP_TIMEOUT=3600 \
  bash scripts/submit_final_lamse.sh
```

Expected final artifacts:

- `$PARAMS_DIR/graphcast_small_lamse_lam0p3_lmax32.005000.npz`
- `$PARAMS_DIR/graphcast_small_lamse_lam0p3_lmax32.005000.npz.opt`
- `runs/lamse_lam0p3_lmax32_final_000000_to_005000.csv`

10. After LAMSE-0.3-5000 exists, run the combined January 2022 visual comparison:

```bash
source .venv-sherlock/activate_graphcast_small_lamse.sh
python3 -m pip install matplotlib==3.8.3
export JAX_PLATFORMS=cuda,cpu
export PARAMS_DIR="${PARAMS_DIR:-${SCRATCH:?Set SCRATCH or PARAMS_DIR}/graphcast-small-lamse/params}"

python3 compare_four_predictions.py \
  --prefinetuned-checkpoint "$PARAMS_DIR/graphcast_small_lamse.000000.npz" \
  --amse-checkpoint "$PARAMS_DIR/graphcast_small_amse.005000.npz" \
  --lamse-checkpoint "$PARAMS_DIR/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
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

This writes:

- `runs/prediction_compare/20220101_lam0p3/comparison_fields.zarr`
- `runs/prediction_compare/20220101_lam0p3/comparison_metrics.csv`
- `runs/prediction_compare/20220101_lam0p3/pairwise_metrics.csv`
- `runs/prediction_compare/20220101_lam0p3/spectral_metrics.csv` if the AMSE-style spherical harmonic diagnostics succeed
- `*_values.png`, `*_errors.png`, `*_delta_vs_amse.png`, and, when available, `*_spectra.png` for each selected field/lead.

11. Run the January 2022 LAMSE-0.1-5000 scorecard/spectral smoke evaluation:

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

12. Save and plot example LAMSE-0.1-5000 prediction-error maps:

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

13. Inspect the AMSE-5000 and LAMSE-0.1-5000 January outputs side by side:

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

14. Save and plot example AMSE-5000 prediction-error maps if not already done:

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

15. Run the same January scorecard/spectral smoke test for the control checkpoint:

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

16. Compare AMSE-5000 vs AMSE-25000 vs LAMSE-0.1-5000 vs LAMSE-0.3-5000 vs control:

- direct scorecard `std` by lead time;
- `runs/prediction_compare/20220101_amse25000/pairwise_metrics.csv`, especially `rmse_improvement_vs_reference`, `mae_improvement_vs_reference`, and `mean_abs_error_improvement` for `model=amse25000`;
- spectral amplitude ratio and coherence at `6h`, `120h`, and `240h`;
- focus first on `z`, `t`, `2t`, `10u`, `10v`, and `msl`.

17. If January comparison is sane, repeat for:

- `$PARAMS_DIR/graphcast_small_amse.001000.npz` as a mid-training AMSE reference;
- full-year 2022 with `--init-interval 12`;
- full-year 2022 for LAMSE-5000 with `--init-interval 12`.
- full-year 2022 for LAMSE-0.3-5000 with `--init-interval 12` if the January results justify it.

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
