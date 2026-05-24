#!/usr/bin/env bash
set -euo pipefail

# Submit the final hybrid-LAMSE run on Sherlock.
#
# Default final target:
#   total LAMSE batches: 5000
#   lambda: 0.1
#   lmax: 32
#   checkpoint cadence: every 500 batches
#   seed: 0
#
# Fresh final run from the GraphCast Small checkpoint:
#   bash scripts/submit_final_lamse.sh
#
# Resume after a 1000-batch pilot and continue to batch 5000:
#   START_BATCH=1000 bash scripts/submit_final_lamse.sh

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
LAMSE_LAMBDA="${LAMSE_LAMBDA:-0.1}"
LAMSE_LMAX="${LAMSE_LMAX:-32}"
LAMSE_TAG="${LAMSE_TAG:-lam${LAMSE_LAMBDA//./p}_lmax${LAMSE_LMAX}}"

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

CHECKPOINT="${CHECKPOINT:-params/graphcast_small_lamse_${LAMSE_TAG}.${START_TAG}.npz}"
CSV_PATH="${CSV_PATH:-runs/lamse_${LAMSE_TAG}_final_${START_TAG}_to_${FINAL_TAG}.csv}"

cat <<EOF
Submitting final LAMSE run:
  ANALYSIS_PATH=${ANALYSIS_PATH}
  CHECKPOINT=${CHECKPOINT}
  START_BATCH=${START_BATCH}
  FINAL_BATCH=${FINAL_BATCH}
  RUN_BATCHES=${RUN_BATCHES}
  LAMSE_LAMBDA=${LAMSE_LAMBDA}
  LAMSE_LMAX=${LAMSE_LMAX}
  SEED=${SEED}
  CSV_PATH=${CSV_PATH}
EOF

sbatch \
  --job-name=gc-lamse-final \
  --partition=serc \
  --gpus=1 \
  --constraint=GPU_MEM:80GB \
  --time="${TIME}" \
  --cpus-per-task=8 \
  --mem=256G \
  --export=ALL,PROJECT_DIR="${PROJECT_DIR}",ANALYSIS_PATH="${ANALYSIS_PATH}",CHECKPOINT="${CHECKPOINT}",BATCH_NUMBER="${RUN_BATCHES}",CHECKPOINT_EVERY="${CHECKPOINT_EVERY}",OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY}",CSV_PATH="${CSV_PATH}",SEED="${SEED}",LAMSE_LAMBDA="${LAMSE_LAMBDA}",LAMSE_LMAX="${LAMSE_LMAX}",LAMSE_TAG="${LAMSE_TAG}" \
  scripts/sbatch_lamse_training.sh
