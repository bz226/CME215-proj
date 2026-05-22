#!/usr/bin/env bash
set -euo pipefail

GRAPHCAST_DIR="${GRAPHCAST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CHECKPOINT="${CHECKPOINT:-params/graphcast_small_lamse.000000.npz}"
ANALYSIS_PATH="${ANALYSIS_PATH:-/path/to/era5_or_weatherbench_1deg}"
NORM_FACTORS="${NORM_FACTORS:-stats}"
LOSS_MODE="${LOSS_MODE:-amse}"
LAMSE_LAMBDA="${LAMSE_LAMBDA:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_NUMBER="${BATCH_NUMBER:-100}"
FORECAST_LENGTH="${FORECAST_LENGTH:-1}"
START_DATE="${START_DATE:-1 Jan 2016 00:00}"
END_DATE="${END_DATE:-31 Dec 2017 18:00}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
CSV_PATH="${CSV_PATH:-}"
LAMSE_LMAX="${LAMSE_LMAX:-}"

cd "${GRAPHCAST_DIR}"

train_args=(
  train.py
  --model-checkpoint "${CHECKPOINT}"
  --apath "${ANALYSIS_PATH}"
  --norm-factors "${NORM_FACTORS}"
  --start-date "${START_DATE}"
  --end-date "${END_DATE}"
  --forecast-length "${FORECAST_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --batch-number "${BATCH_NUMBER}"
  --checkpoint-every "${BATCH_NUMBER}"
  --learning-rate "${LEARNING_RATE}"
)

case "${LOSS_MODE}" in
  amse)
    train_args+=(--spectral-amse)
    ;;
  lamse)
    train_args+=(--lamse --lamse-lambda "${LAMSE_LAMBDA}")
    if [[ -n "${LAMSE_LMAX}" ]]; then
      train_args+=(--lamse-lmax "${LAMSE_LMAX}")
    fi
    ;;
  mse)
    ;;
  *)
    echo "Unknown LOSS_MODE=${LOSS_MODE}; expected amse, lamse, or mse." >&2
    exit 2
    ;;
esac

if [[ -n "${CSV_PATH}" ]]; then
  mkdir -p "$(dirname "${CSV_PATH}")"
  train_args+=(--to-csv "${CSV_PATH}")
fi

PYTHONPATH=. python "${train_args[@]}"
