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
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_lamse_100_batches.sh" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

if [[ ! -f "${PROJECT_DIR}/scripts/run_lamse_100_batches.sh" ]]; then
  cat >&2 <<EOF
Could not locate the graphcast-small-lamse project directory.

Submit from the project root:
  cd /path/to/graphcast-small-lamse
  sbatch scripts/sbatch_lamse_100_batches.sh

Or pass PROJECT_DIR explicitly:
  PROJECT_DIR=/path/to/graphcast-small-lamse sbatch scripts/sbatch_lamse_100_batches.sh
EOF
  exit 2
fi

cd "${PROJECT_DIR}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-gc-lamse-100}-${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs}"
mkdir -p "${OUTPUT_DIR}"
DEFAULT_DATA_DIR="${SCRATCH:-${PROJECT_DIR}/data}/graphcast-small-lamse/era5_1deg_weatherbench2"
ANALYSIS_PATH="${ANALYSIS_PATH:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
LAMSE_LAMBDA="${LAMSE_LAMBDA:-0.0}"
CSV_PATH="${CSV_PATH:-${OUTPUT_DIR}/lamse_${LAMSE_LAMBDA}_job_${SLURM_JOB_ID:-manual}.csv}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-40000}"

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

echo "Log file: ${LOG_FILE}"
echo "CSV file: ${CSV_PATH}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"
echo "ANALYSIS_PATH=${ANALYSIS_PATH}"
echo "LAMSE_LAMBDA=${LAMSE_LAMBDA}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if ! [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse GPU memory from nvidia-smi: ${GPU_MEM_MB}" >&2
    exit 2
  fi
  if [[ "${GPU_MEM_MB}" -lt "${MIN_GPU_MEM_MB}" ]]; then
    cat >&2 <<EOF
Allocated GPU has ${GPU_MEM_MB} MiB, below MIN_GPU_MEM_MB=${MIN_GPU_MEM_MB}.

This GraphCast Small + AMSE/LAMSE gate should run on a modern high-memory GPU.
Resubmit with an explicit Sherlock GPU constraint, for example:
  sbatch -C GPU_MEM:80GB scripts/sbatch_lamse_100_batches.sh

If your account uses a different owner partition or feature names, inspect:
  sh_node_feat -p gpu | grep GPU_
EOF
    exit 2
  fi
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
export JAX_ENABLE_X64=1
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

python - <<'PY'
import jax
import jax.numpy as jnp
print("jax", jax.__version__)
print("x64", jax.config.jax_enable_x64)
print("devices", jax.devices())
print("cpu devices", jax.devices("cpu"))
print("tiny gpu op", (jnp.asarray([1.0], dtype=jnp.float32) + 1.0).block_until_ready())
PY

python -u scripts/download_graphcast_stats.py
python scripts/prepare_graphcast_small_checkpoint.py
python scripts/inspect_graphcast_checkpoint.py

LAMSE_LAMBDA="${LAMSE_LAMBDA}" \
ANALYSIS_PATH="${ANALYSIS_PATH}" \
CSV_PATH="${CSV_PATH}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
BATCH_NUMBER="${BATCH_NUMBER:-100}" \
FORECAST_LENGTH="${FORECAST_LENGTH:-1}" \
START_DATE="${START_DATE:-1 Jan 2016 00:00}" \
END_DATE="${END_DATE:-31 Dec 2017 18:00}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
scripts/run_lamse_100_batches.sh
