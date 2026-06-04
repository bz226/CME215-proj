#!/usr/bin/env bash
set -euo pipefail

# Default weighted-MSE training runner. This script intentionally passes neither
# --spectral-amse nor --lamse, so train.py uses the built-in spatial MSE loss.
#
# Typical Sherlock interactive use:
#   source .venv-sherlock/activate_graphcast_small_lamse.sh
#   ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
#   BATCH_NUMBER=1000 \
#     scripts/run_mse_training.sh

GRAPHCAST_DIR="${GRAPHCAST_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -n "${SCRATCH:-}" ]]; then
  DEFAULT_PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
else
  DEFAULT_PARAMS_DIR="${GRAPHCAST_DIR}/params"
fi
PARAMS_DIR="${PARAMS_DIR:-${DEFAULT_PARAMS_DIR}}"
export PARAMS_DIR
CHECKPOINT="${CHECKPOINT:-${PARAMS_DIR}/graphcast_small_mse.000000.npz}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PARAMS_DIR}/graphcast_small_lamse.000000.npz}"
DEFAULT_DATA_DIR="${SCRATCH:-${GRAPHCAST_DIR}/data}/graphcast-small-lamse/era5_1deg_weatherbench2"
ANALYSIS_PATH="${ANALYSIS_PATH:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
NORM_FACTORS="${NORM_FACTORS:-stats}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_NUMBER="${BATCH_NUMBER:-1000}"
FORECAST_LENGTH="${FORECAST_LENGTH:-1}"
START_DATE="${START_DATE:-1 Jan 2016 00:00}"
END_DATE="${END_DATE:-31 Dec 2017 18:00}"
NUM_PRELOAD="${NUM_PRELOAD:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY:-100}"
CSV_PATH="${CSV_PATH:-runs/mse_full_${SLURM_JOB_ID:-manual}.csv}"
SEED="${SEED:-0}"
FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT:-1800}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-180}"

cd "${GRAPHCAST_DIR}"
mkdir -p "${PARAMS_DIR}"

# train.py needs both backends: CPU for canonical params and CUDA for gradients.
if [[ -z "${JAX_PLATFORMS:-}" || "${JAX_PLATFORMS}" == "cuda" ]]; then
  export JAX_PLATFORMS="cuda,cpu"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  echo "Activate the Sherlock venv first: source .venv-sherlock/activate_graphcast_small_lamse.sh" >&2
  exit 2
fi

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python >= 3.10 is required, got {sys.version}. "
        "Activate .venv-sherlock/activate_graphcast_small_lamse.sh first."
    )
print("python", sys.version.split()[0])
PY

if [[ ! -d "${ANALYSIS_PATH}" ]] || ! compgen -G "${ANALYSIS_PATH}/*/*" >/dev/null; then
  cat >&2 <<EOF
ANALYSIS_PATH does not contain monthly zarr stores: ${ANALYSIS_PATH}

Expected layout:
  ${ANALYSIS_PATH}/2016/01
  ${ANALYSIS_PATH}/2016/02
  ...

Stage data first:
  DATA_DIR=${ANALYSIS_PATH} sbatch scripts/sbatch_fetch_training_data.sh
EOF
  exit 2
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  if [[ "${CHECKPOINT}" == *".000000.npz" ]]; then
    if [[ ! -f "${SOURCE_CHECKPOINT}" ]]; then
      "${PYTHON_BIN}" scripts/download_graphcast_small.py --output-dir "${PARAMS_DIR}"
      "${PYTHON_BIN}" scripts/prepare_graphcast_small_checkpoint.py --target "${SOURCE_CHECKPOINT}"
    fi
    "${PYTHON_BIN}" scripts/prepare_graphcast_small_checkpoint.py \
      --source "${SOURCE_CHECKPOINT}" \
      --target "${CHECKPOINT}" \
      --mode symlink
  else
    echo "Missing non-initial checkpoint: ${CHECKPOINT}" >&2
    exit 2
  fi
fi

"${PYTHON_BIN}" -u scripts/download_graphcast_stats.py

echo "LOSS_MODE=mse"
echo "PARAMS_DIR=${PARAMS_DIR}"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
echo "ANALYSIS_PATH=${ANALYSIS_PATH}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "BATCH_NUMBER=${BATCH_NUMBER}"
echo "FORECAST_LENGTH=${FORECAST_LENGTH}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "OPT_CHECKPOINT_EVERY=${OPT_CHECKPOINT_EVERY}"
echo "CSV_PATH=${CSV_PATH}"
echo "FIRST_STEP_TIMEOUT=${FIRST_STEP_TIMEOUT}"
echo "WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT}"

mkdir -p "$(dirname "${CSV_PATH}")"

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
  --num-preload "${NUM_PRELOAD}"
  --checkpoint-every "${CHECKPOINT_EVERY}"
  --opt-checkpoint-every "${OPT_CHECKPOINT_EVERY}"
  --learning-rate "${LEARNING_RATE}"
  --seed "${SEED}"
  --to-csv "${CSV_PATH}"
  --first-step-timeout "${FIRST_STEP_TIMEOUT}"
  --watchdog-timeout "${WATCHDOG_TIMEOUT}"
)

if [[ -n "${COSINE_WARMUP:-}" || -n "${COSINE_TOTAL:-}" ]]; then
  if [[ -z "${COSINE_WARMUP:-}" || -z "${COSINE_TOTAL:-}" ]]; then
    echo "Set both COSINE_WARMUP and COSINE_TOTAL, or neither." >&2
    exit 2
  fi
  train_args+=(--cosine-anneal "${COSINE_WARMUP}" "${COSINE_TOTAL}")
  if [[ -n "${COSINE_END_RATE:-}" ]]; then
    train_args+=(--cosine-anneal-end-rate "${COSINE_END_RATE}")
  fi
fi

if [[ "${DEBUG_MEMORY:-0}" == "1" ]]; then
  train_args+=(--debug-memory)
fi

PYTHONPATH=. "${PYTHON_BIN}" "${train_args[@]}"
