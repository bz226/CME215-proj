#!/usr/bin/env bash
# Submit with:
#   DATA_DIR=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
#     sbatch scripts/sbatch_fetch_training_data.sh
#
# Purpose:
#   Stage the WeatherBench 2 ERA5 data needed by the 100-batch training gate.
#   This is a CPU/network job; do not spend a GPU allocation on data staging.

#SBATCH --job-name=gc-lamse-data
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-gc-lamse-data}-${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

DEFAULT_DATA_DIR="${SCRATCH:-${PROJECT_DIR}/data}/graphcast-small-lamse/era5_1deg_weatherbench2"
DATA_DIR="${DATA_DIR:-${DEFAULT_DATA_DIR}}"
START_DATE="${START_DATE:-1 Jan 2016 00:00}"
END_DATE="${END_DATE:-31 Dec 2017 18:00}"
FORECAST_LENGTH="${FORECAST_LENGTH:-1}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "Log file: ${LOG_FILE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"
echo "DATA_DIR=${DATA_DIR}"
echo "START_DATE=${START_DATE}"
echo "END_DATE=${END_DATE}"
echo "FORECAST_LENGTH=${FORECAST_LENGTH}"

if [[ ! -f .venv-sherlock/activate_graphcast_small_lamse.sh ]]; then
  cat >&2 <<'EOF'
Missing .venv-sherlock/activate_graphcast_small_lamse.sh.

Create the Sherlock environment first:
  sbatch scripts/sbatch_setup_env.sh
EOF
  exit 2
fi

source .venv-sherlock/activate_graphcast_small_lamse.sh

python scripts/download_graphcast_small.py
python scripts/prepare_graphcast_small_checkpoint.py

JAX_PLATFORMS=cpu \
python scripts/download_weatherbench2_era5_1deg.py \
  --output-dir "${DATA_DIR}" \
  --train-start "${START_DATE}" \
  --train-end "${END_DATE}" \
  --forecast-length "${FORECAST_LENGTH}" \
  --threads "${THREADS}"

cat <<EOF

Training data staging finished.

Use this for the training job:
  ANALYSIS_PATH=${DATA_DIR} LAMSE_LAMBDA=0.0 sbatch scripts/sbatch_lamse_100_batches.sh
EOF
