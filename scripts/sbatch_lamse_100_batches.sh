#!/usr/bin/env bash
# Submit with:
#   ANALYSIS_PATH=/path/to/era5_or_weatherbench_1deg LAMSE_LAMBDA=0.0 \
#     sbatch scripts/sbatch_lamse_100_batches.sh
#
# After lambda-zero is stable, repeat with:
#   ANALYSIS_PATH=/path/to/era5_or_weatherbench_1deg LAMSE_LAMBDA=0.1 \
#     sbatch scripts/sbatch_lamse_100_batches.sh

#SBATCH --job-name=gc-lamse-100
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

DEFAULT_DATA_DIR="${SCRATCH:-${PROJECT_DIR}/data}/graphcast-small-lamse/era5_1deg_weatherbench2"
ANALYSIS_PATH="${ANALYSIS_PATH:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"

if [[ -z "${ANALYSIS_PATH:-}" ]]; then
  cat >&2 <<'EOF'
ANALYSIS_PATH is required.

Example:
  ANALYSIS_PATH=/path/to/era5_or_weatherbench_1deg LAMSE_LAMBDA=0.0 \
    sbatch scripts/sbatch_lamse_100_batches.sh

To stage the default WeatherBench 2 data first:
  DATA_DIR=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
    sbatch scripts/sbatch_fetch_training_data.sh
EOF
  exit 2
fi

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

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"
echo "ANALYSIS_PATH=${ANALYSIS_PATH}"
echo "LAMSE_LAMBDA=${LAMSE_LAMBDA:-0.0}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found in PATH"
fi

if [[ ! -f .venv-sherlock/activate_graphcast_small_lamse.sh ]]; then
  cat >&2 <<'EOF'
Missing .venv-sherlock/activate_graphcast_small_lamse.sh.

Create the Sherlock environment first:
  sbatch scripts/sbatch_setup_env.sh
EOF
  exit 2
fi

source .venv-sherlock/activate_graphcast_small_lamse.sh

python - <<'PY'
import jax
print("jax", jax.__version__)
print("x64", jax.config.jax_enable_x64)
print("devices", jax.devices())
PY

python scripts/prepare_graphcast_small_checkpoint.py
python scripts/inspect_graphcast_checkpoint.py

LAMSE_LAMBDA="${LAMSE_LAMBDA:-0.0}" \
ANALYSIS_PATH="${ANALYSIS_PATH}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
BATCH_NUMBER="${BATCH_NUMBER:-100}" \
FORECAST_LENGTH="${FORECAST_LENGTH:-1}" \
START_DATE="${START_DATE:-1 Jan 2016 00:00}" \
END_DATE="${END_DATE:-31 Dec 2017 18:00}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
scripts/run_lamse_100_batches.sh
