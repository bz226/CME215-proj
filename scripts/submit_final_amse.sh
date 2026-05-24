#!/usr/bin/env bash
set -euo pipefail

# Submit the final pure-AMSE baseline run on Sherlock.
#
# Default final target:
#   total AMSE batches: 5000
#   checkpoint cadence: every 500 batches
#   seed: 0
#
# Fresh final run from the GraphCast Small checkpoint:
#   bash scripts/submit_final_amse.sh
#
# Resume after the 1000-batch pilot and continue to batch 5000:
#   START_BATCH=1000 bash scripts/submit_final_amse.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

if [[ -z "${ANALYSIS_PATH:-}" ]]; then
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "Set ANALYSIS_PATH, or run on Sherlock with SCRATCH set." >&2
    exit 2
  fi
  ANALYSIS_PATH="${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2"
fi

FINAL_BATCH="${FINAL_BATCH:-5000}"
START_BATCH="${START_BATCH:-0}"
SEED="${SEED:-0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-500}"
OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY:-500}"
TIME="${TIME:-24:00:00}"
FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT:-1800}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-180}"
if [[ -n "${SCRATCH:-}" ]]; then
  DEFAULT_PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
else
  DEFAULT_PARAMS_DIR="${PROJECT_DIR}/params"
fi
PARAMS_DIR="${PARAMS_DIR:-${DEFAULT_PARAMS_DIR}}"
mkdir -p "${PARAMS_DIR}"

if ! [[ "${FINAL_BATCH}" =~ ^[0-9]+$ && "${START_BATCH}" =~ ^[0-9]+$ ]]; then
  echo "FINAL_BATCH and START_BATCH must be non-negative integers." >&2
  exit 2
fi
if (( FINAL_BATCH <= START_BATCH )); then
  echo "FINAL_BATCH=${FINAL_BATCH} must be greater than START_BATCH=${START_BATCH}." >&2
  exit 2
fi

RUN_BATCHES=$((FINAL_BATCH - START_BATCH))
START_TAG="$(printf "%06d" "${START_BATCH}")"
FINAL_TAG="$(printf "%06d" "${FINAL_BATCH}")"

CHECKPOINT="${CHECKPOINT:-${PARAMS_DIR}/graphcast_small_amse.${START_TAG}.npz}"
CSV_PATH="${CSV_PATH:-runs/amse_final_${START_TAG}_to_${FINAL_TAG}.csv}"

cat <<EOF
Submitting final AMSE baseline:
  ANALYSIS_PATH=${ANALYSIS_PATH}
  PARAMS_DIR=${PARAMS_DIR}
  CHECKPOINT=${CHECKPOINT}
  START_BATCH=${START_BATCH}
  FINAL_BATCH=${FINAL_BATCH}
  RUN_BATCHES=${RUN_BATCHES}
  SEED=${SEED}
  CSV_PATH=${CSV_PATH}
  FIRST_STEP_TIMEOUT=${FIRST_STEP_TIMEOUT}
  WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT}
EOF

sbatch \
  --job-name=gc-amse-final \
  --partition=serc \
  --gpus=1 \
  --constraint=GPU_MEM:80GB \
  --time="${TIME}" \
  --cpus-per-task=8 \
  --mem=256G \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ANALYSIS_PATH="${ANALYSIS_PATH}",PARAMS_DIR="${PARAMS_DIR}",CHECKPOINT="${CHECKPOINT}",BATCH_NUMBER="${RUN_BATCHES}",CHECKPOINT_EVERY="${CHECKPOINT_EVERY}",OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY}",CSV_PATH="${CSV_PATH}",SEED="${SEED}",FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT}",WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
  scripts/sbatch_amse_training.sh
