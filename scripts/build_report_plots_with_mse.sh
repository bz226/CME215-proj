#!/usr/bin/env bash
set -euo pipefail

# Regenerate report-style plots with MSE-5000 replacing AMSE-25000.
#
# The default aggregate step rewrites the same top-level plot names used by the
# current report, e.g. plots/10m_wind_speed_std_mean_error_by_lead.png.
# The default report model set is:
#   Pre-finetuned, MSE-5000, AMSE-5000, LAMSE-0.1-LMAX32, LAMSE-0.5-LMAX127.
# Set INCLUDE_AMSE25000=1 only if you want the long-AMSE ablation back.
# Optional single-case and hurricane steps need Sherlock/JAX and the trained
# MSE checkpoint.
#
# Typical aggregate-only use after MSE scorecard evaluation finishes:
#   EVAL_ROOT=runs/eval/2022_full_i12_retry REPORT_PLOTS_DIR=plots \
#     bash scripts/build_report_plots_with_mse.sh
#
# To also regenerate the January showcase/spectral plots:
#   RUN_SINGLE_CASE=1 EVAL_ROOT=runs/eval/2022_full_i12_retry REPORT_PLOTS_DIR=plots \
#     bash scripts/build_report_plots_with_mse.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

EVAL_ROOT="${EVAL_ROOT:-runs/eval/2022_full_i12_retry}"
REPORT_PLOTS_DIR="${REPORT_PLOTS_DIR:-plots}"
SUMMARY_CSV="${SUMMARY_CSV:-${REPORT_PLOTS_DIR}/scorecard_summary_with_mse.csv}"
FIELDS="${FIELDS:-z:500 t:850 2t 10m_wind_speed msl}"
STAT="${STAT:-std}"
METRIC="${METRIC:-mean}"
REFERENCE_MODEL="${REFERENCE_MODEL:-amse5000}"
MSE_SCORE_PATH="${MSE_SCORE_PATH:-${EVAL_ROOT}/mse5000_score_2022.zarr}"

mkdir -p "${REPORT_PLOTS_DIR}"

declare -a SCORE_SPECS=(
  "control|${EVAL_ROOT}/control_score_2022.zarr|required"
  "mse5000|${MSE_SCORE_PATH}|required"
  "amse5000|${EVAL_ROOT}/amse5000_score_2022.zarr|required"
  "lamse0p1_lmax32_5000|${EVAL_ROOT}/lamse0p1_lmax32_5000_score_2022.zarr|optional"
  "lamse0p5_lmax127_5000|${EVAL_ROOT}/lamse0p5_lmax127_5000_score_2022.zarr|optional"
)

if [[ "${INCLUDE_AMSE25000:-0}" == "1" ]]; then
  SCORE_SPECS+=("amse25000|${EVAL_ROOT}/amse25000_score_2022.zarr|optional")
fi
if [[ "${INCLUDE_LAMSE0P3:-0}" == "1" ]]; then
  SCORE_SPECS+=("lamse0p3_lmax32_5000|${EVAL_ROOT}/lamse0p3_lmax32_5000_score_2022.zarr|optional")
fi

declare -a SCORE_ARGS=()
for spec in "${SCORE_SPECS[@]}"; do
  IFS='|' read -r model path requiredness <<< "${spec}"
  if [[ -e "${path}" ]]; then
    SCORE_ARGS+=(--score "${model}=${path}")
  elif [[ "${requiredness}" == "required" ]]; then
    cat >&2 <<EOF
Missing required scorecard zarr for ${model}: ${path}

If MSE training is done but MSE-5000 has not been evaluated yet, run:
  APPEND_MANIFEST=1 INCLUDE_MSE=1 MODELS_TO_RUN="mse5000" OUTPUT_ROOT=${EVAL_ROOT} \\
    bash scripts/submit_scorecard_2022_all.sh
EOF
    exit 2
  else
    echo "Skipping optional missing scorecard: ${model} (${path})" >&2
  fi
done

read -r -a FIELD_ARGS <<< "${FIELDS}"

python3 scripts/summarize_scorecards.py \
  "${SCORE_ARGS[@]}" \
  --out-csv "${SUMMARY_CSV}"

python3 scripts/plot_scorecard_summary.py \
  --summary-csv "${SUMMARY_CSV}" \
  --out-dir "${REPORT_PLOTS_DIR}" \
  --fields "${FIELD_ARGS[@]}" \
  --stat "${STAT}" \
  --metric "${METRIC}" \
  --reference-model "${REFERENCE_MODEL}"

echo "Report aggregate plots with MSE: ${REPORT_PLOTS_DIR}"

if [[ "${RUN_SINGLE_CASE:-0}" == "1" ]]; then
  if [[ -z "${SCRATCH:-}" && -z "${APATH:-}" ]]; then
    echo "Set APATH, or run on Sherlock with SCRATCH set, for RUN_SINGLE_CASE=1." >&2
    exit 2
  fi
  if [[ -z "${PARAMS_DIR:-}" ]]; then
    if [[ -z "${SCRATCH:-}" ]]; then
      echo "Set PARAMS_DIR, or run on Sherlock with SCRATCH set, for RUN_SINGLE_CASE=1." >&2
      exit 2
    fi
    PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
  fi
  APATH="${APATH:-${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022}"
  SINGLE_CASE_DIR="${SINGLE_CASE_DIR:-${REPORT_PLOTS_DIR}/prediction_compare_with_mse}"

  single_case_args=(
    python3 compare_four_predictions.py
    --prefinetuned-checkpoint "${PARAMS_DIR}/graphcast_small_lamse.000000.npz" \
    --amse-checkpoint "${PARAMS_DIR}/graphcast_small_amse.005000.npz" \
    --lamse-checkpoint "${PARAMS_DIR}/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
    --extra-checkpoint "mse5000=${PARAMS_DIR}/graphcast_small_mse.005000.npz" \
    --extra-checkpoint "lamse0p5_lmax127=${PARAMS_DIR}/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
    --apath "${APATH}" \
    --norm-factors stats \
    --init-date "${INIT_DATE:-1 Jan 2022 00:00}" \
    --forecast-length "${FORECAST_LENGTH:-240}" \
    --lead-hours ${SINGLE_CASE_LEADS:-240} \
    --fields ${SINGLE_CASE_FIELDS:-10m_wind_speed} \
    --reference-model amse \
    --out-dir "${SINGLE_CASE_DIR}" \
    --overwrite
  )
  if [[ "${INCLUDE_AMSE25000:-0}" == "1" ]]; then
    single_case_args+=(--extra-checkpoint "amse25000=${PARAMS_DIR}/graphcast_small_amse.025000.npz")
  fi
  if [[ "${INCLUDE_LAMSE0P3:-0}" == "1" ]]; then
    single_case_args+=(--extra-checkpoint "lamse0p3=${PARAMS_DIR}/graphcast_small_lamse_lam0p3_lmax32.005000.npz")
  fi
  "${single_case_args[@]}"

  echo "Report single-case plots with MSE: ${SINGLE_CASE_DIR}"
fi

if [[ "${RUN_HURRICANE:-0}" == "1" ]]; then
  if [[ -z "${SCRATCH:-}" && -z "${APATH:-}" ]]; then
    echo "Set APATH, or run on Sherlock with SCRATCH set, for RUN_HURRICANE=1." >&2
    exit 2
  fi
  if [[ -z "${PARAMS_DIR:-}" ]]; then
    if [[ -z "${SCRATCH:-}" ]]; then
      echo "Set PARAMS_DIR, or run on Sherlock with SCRATCH set, for RUN_HURRICANE=1." >&2
      exit 2
    fi
    PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
  fi
  if [[ -z "${TRACK_FILE:-}" ]]; then
    if [[ -z "${SCRATCH:-}" ]]; then
      echo "Set TRACK_FILE, or run on Sherlock with SCRATCH set, for RUN_HURRICANE=1." >&2
      exit 2
    fi
    TRACK_FILE="${SCRATCH}/graphcast-small-lamse/tracks/hurdat2_atlantic.txt"
  fi
  APATH="${APATH:-${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022}"
  HURRICANE_DIR="${HURRICANE_DIR:-${REPORT_PLOTS_DIR}/hurricane_ian_with_mse}"

  hurricane_args=(
    python3 hurricane_performance_check.py
    --track-file "${TRACK_FILE}" \
    --storm-name IAN \
    --storm-year 2022 \
    --apath "${APATH}" \
    --norm-factors stats \
    --checkpoint "prefinetuned=${PARAMS_DIR}/graphcast_small_lamse.000000.npz" \
    --checkpoint "mse5000=${PARAMS_DIR}/graphcast_small_mse.005000.npz" \
    --checkpoint "amse5000=${PARAMS_DIR}/graphcast_small_amse.005000.npz" \
    --checkpoint "lamse0p1_lmax32=${PARAMS_DIR}/graphcast_small_lamse_lam0p1_lmax32.005000.npz" \
    --checkpoint "lamse0p5_lmax127=${PARAMS_DIR}/graphcast_small_lamse_lam0p5_lmax127.005000.npz" \
    --init-start "${IAN_INIT_START:-19 Sep 2022 00:00}" \
    --init-end "${IAN_INIT_END:-27 Sep 2022 00:00}" \
    --init-interval-hours "${IAN_INIT_INTERVAL_HOURS:-24}" \
    --forecast-length "${IAN_FORECAST_LENGTH:-240}" \
    --lead-hours ${IAN_LEAD_HOURS:-6 12 18 24 36 48 72 96 120 144 168 192 216 240} \
    --truth-source "${IAN_TRUTH_SOURCE:-analysis}" \
    --out-dir "${HURRICANE_DIR}" \
    --overwrite
  )
  if [[ "${INCLUDE_AMSE25000:-0}" == "1" ]]; then
    hurricane_args+=(--checkpoint "amse25000=${PARAMS_DIR}/graphcast_small_amse.025000.npz")
  fi
  if [[ "${INCLUDE_LAMSE0P3:-0}" == "1" ]]; then
    hurricane_args+=(--checkpoint "lamse0p3_lmax32=${PARAMS_DIR}/graphcast_small_lamse_lam0p3_lmax32.005000.npz")
  fi
  "${hurricane_args[@]}"

  echo "Report hurricane plot with MSE: ${HURRICANE_DIR}/hurricane_error_summary.png"
fi
