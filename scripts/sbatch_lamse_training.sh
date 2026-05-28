#!/usr/bin/env bash
# Hybrid LAMSE Sherlock training job. This wrapper always runs --lamse.
#
# Submit from the project root:
#   ANALYSIS_PATH=$SCRATCH/graphcast-small-lamse/era5_1deg_weatherbench2 \
#   LAMSE_LAMBDA=0.1 LAMSE_LMAX=32 BATCH_NUMBER=1000 \
#     sbatch scripts/sbatch_lamse_training.sh

#SBATCH --job-name=gc-lamse-train
#SBATCH --partition=serc
#SBATCH --gpus=1
#SBATCH --constraint=GPU_MEM:80GB
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/run_lamse_training.sh" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

if [[ ! -f "${PROJECT_DIR}/scripts/run_lamse_training.sh" ]]; then
  cat >&2 <<EOF
Could not locate the graphcast-small-lamse project directory.

Submit from the project root:
  cd /path/to/graphcast-small-lamse
  sbatch scripts/sbatch_lamse_training.sh

Or pass PROJECT_DIR explicitly:
  PROJECT_DIR=/path/to/graphcast-small-lamse sbatch scripts/sbatch_lamse_training.sh
EOF
  exit 2
fi

cd "${PROJECT_DIR}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-gc-lamse-train}-${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs}"
mkdir -p "${OUTPUT_DIR}"
DEFAULT_DATA_DIR="${SCRATCH:-${PROJECT_DIR}/data}/graphcast-small-lamse/era5_1deg_weatherbench2"
ANALYSIS_PATH="${ANALYSIS_PATH:-${DATA_DIR:-${DEFAULT_DATA_DIR}}}"
LAMSE_LAMBDA="${LAMSE_LAMBDA:-0.1}"
LAMSE_LMAX="${LAMSE_LMAX:-32}"
LAMSE_INPUT_NSIDE="${LAMSE_INPUT_NSIDE:-}"
LAMSE_TAG="${LAMSE_TAG:-lam${LAMSE_LAMBDA//./p}_lmax${LAMSE_LMAX}}"
if [[ -n "${SCRATCH:-}" ]]; then
  DEFAULT_PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
else
  DEFAULT_PARAMS_DIR="${PROJECT_DIR}/params"
fi
PARAMS_DIR="${PARAMS_DIR:-${DEFAULT_PARAMS_DIR}}"
export PARAMS_DIR
mkdir -p "${PARAMS_DIR}"
CHECKPOINT="${CHECKPOINT:-${PARAMS_DIR}/graphcast_small_lamse_${LAMSE_TAG}.000000.npz}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${PARAMS_DIR}/graphcast_small_lamse.000000.npz}"
CSV_PATH="${CSV_PATH:-${OUTPUT_DIR}/lamse_${LAMSE_TAG}_full_job_${SLURM_JOB_ID:-manual}.csv}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-40000}"
BATCH_NUMBER="${BATCH_NUMBER:-1000}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-100}"
OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY:-100}"
FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT:-1800}"
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-180}"

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
echo "PARAMS_DIR=${PARAMS_DIR}"
echo "LOSS_MODE=lamse"
echo "CHECKPOINT=${CHECKPOINT}"
echo "SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT}"
echo "LAMSE_LAMBDA=${LAMSE_LAMBDA}"
echo "LAMSE_LMAX=${LAMSE_LMAX}"
echo "LAMSE_INPUT_NSIDE=${LAMSE_INPUT_NSIDE:-auto}"
echo "BATCH_NUMBER=${BATCH_NUMBER}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "OPT_CHECKPOINT_EVERY=${OPT_CHECKPOINT_EVERY}"
echo "FIRST_STEP_TIMEOUT=${FIRST_STEP_TIMEOUT}"
echo "WATCHDOG_TIMEOUT=${WATCHDOG_TIMEOUT}"

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

Resubmit with an explicit Sherlock GPU constraint, for example:
  sbatch -C GPU_MEM:80GB scripts/sbatch_lamse_training.sh
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
# train.py keeps canonical parameters on CPU and evaluates gradients on GPU.
export JAX_PLATFORMS="${JAX_PLATFORMS_OVERRIDE:-cuda,cpu}"

python3 - <<'PY'
import jax
import jax.numpy as jnp
print("jax", jax.__version__)
print("x64", jax.config.jax_enable_x64)
print("devices", jax.devices())
print("cpu devices", jax.devices("cpu"))
print("tiny gpu op", (jnp.asarray([1.0], dtype=jnp.float32) + 1.0).block_until_ready())
PY

PYTHON_BIN="${PYTHON_BIN:-python3}" \
PARAMS_DIR="${PARAMS_DIR}" \
CHECKPOINT="${CHECKPOINT}" \
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
ANALYSIS_PATH="${ANALYSIS_PATH}" \
CSV_PATH="${CSV_PATH}" \
BATCH_SIZE="${BATCH_SIZE:-1}" \
BATCH_NUMBER="${BATCH_NUMBER}" \
FORECAST_LENGTH="${FORECAST_LENGTH:-1}" \
NUM_PRELOAD="${NUM_PRELOAD:-1}" \
START_DATE="${START_DATE:-1 Jan 2016 00:00}" \
END_DATE="${END_DATE:-31 Dec 2017 18:00}" \
LEARNING_RATE="${LEARNING_RATE:-1e-6}" \
CHECKPOINT_EVERY="${CHECKPOINT_EVERY}" \
OPT_CHECKPOINT_EVERY="${OPT_CHECKPOINT_EVERY}" \
SEED="${SEED:-0}" \
FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT}" \
WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT}" \
LAMSE_LAMBDA="${LAMSE_LAMBDA}" \
LAMSE_LMAX="${LAMSE_LMAX}" \
LAMSE_INPUT_NSIDE="${LAMSE_INPUT_NSIDE}" \
LAMSE_TAG="${LAMSE_TAG}" \
COSINE_WARMUP="${COSINE_WARMUP:-}" \
COSINE_TOTAL="${COSINE_TOTAL:-}" \
COSINE_END_RATE="${COSINE_END_RATE:-}" \
DEBUG_MEMORY="${DEBUG_MEMORY:-0}" \
scripts/run_lamse_training.sh
