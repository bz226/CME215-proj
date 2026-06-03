#!/usr/bin/env bash
# Full-year 2022 deterministic scorecard/spectral evaluation for one checkpoint.
#
# Submit from the project root through scripts/submit_scorecard_2022_all.sh, or directly:
#   MODEL_NAME=amse5000 \
#   MODEL_CHECKPOINT=$SCRATCH/graphcast-small-lamse/params/graphcast_small_amse.005000.npz \
#     sbatch scripts/sbatch_scorecard_2022.sh

#SBATCH --job-name=gc-score-2022
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
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/build_scorecard.py" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

cd "${PROJECT_DIR}"

MODEL_NAME="${MODEL_NAME:?Set MODEL_NAME, e.g. amse5000}"
MODEL_CHECKPOINT="${MODEL_CHECKPOINT:?Set MODEL_CHECKPOINT to a .npz checkpoint}"

if [[ -z "${SCRATCH:-}" && -z "${APATH:-}" ]]; then
  echo "Set APATH, or run on Sherlock with SCRATCH set." >&2
  exit 2
fi

APATH="${APATH:-${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022}"
NORM_FACTORS="${NORM_FACTORS:-stats}"
# Do not use CPATH for climatology here: Sherlock modules set CPATH to compiler
# include paths, which would be misread by build_scorecard.py as a zarr store.
CLIMATO_PATH="${CLIMATO_PATH:-${SCORECARD_CPATH:-none}}"
START_DATE="${START_DATE:-1 Jan 2022 00:00}"
END_DATE="${END_DATE:-31 Dec 2022 18:00}"
FORECAST_LENGTH="${FORECAST_LENGTH:-240}"
INIT_INTERVAL="${INIT_INTERVAL:-12}"
SPECTRUM_LEADS="${SPECTRUM_LEADS:-6 120 240}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval/2022_full_i${INIT_INTERVAL}}"
TO_PATH="${TO_PATH:-${OUTPUT_ROOT}/${MODEL_NAME}_score_2022.zarr}"
SPEC_PATH="${SPEC_PATH:-${OUTPUT_ROOT}/${MODEL_NAME}_spec_2022.zarr}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-40000}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-gc-score-2022}-${MODEL_NAME}-${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

if [[ ! -f "${MODEL_CHECKPOINT}" ]]; then
  echo "Missing MODEL_CHECKPOINT: ${MODEL_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -d "${APATH}" ]] || ! compgen -G "${APATH}/*/*" >/dev/null; then
  echo "APATH does not contain monthly zarr stores: ${APATH}" >&2
  exit 2
fi

echo "Log file: ${LOG_FILE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_CHECKPOINT=${MODEL_CHECKPOINT}"
echo "APATH=${APATH}"
echo "NORM_FACTORS=${NORM_FACTORS}"
echo "CLIMATO_PATH=${CLIMATO_PATH}"
echo "START_DATE=${START_DATE}"
echo "END_DATE=${END_DATE}"
echo "FORECAST_LENGTH=${FORECAST_LENGTH}"
echo "INIT_INTERVAL=${INIT_INTERVAL}"
echo "SPECTRUM_LEADS=${SPECTRUM_LEADS}"
echo "TO_PATH=${TO_PATH}"
echo "SPEC_PATH=${SPEC_PATH}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
  GPU_MEM_MB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
  if [[ "${GPU_MEM_MB}" =~ ^[0-9]+$ && "${GPU_MEM_MB}" -lt "${MIN_GPU_MEM_MB}" ]]; then
    echo "Allocated GPU has ${GPU_MEM_MB} MiB, below MIN_GPU_MEM_MB=${MIN_GPU_MEM_MB}." >&2
    exit 2
  fi
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
export JAX_ENABLE_X64="${JAX_ENABLE_X64:-1}"
export JAX_PLATFORMS="${JAX_PLATFORMS_OVERRIDE:-cuda,cpu}"

SPECTRUM_ARGS=()
if [[ -n "${SPECTRUM_LEADS//[[:space:]]/}" ]]; then
  read -r -a SPECTRUM_LEAD_ARGS <<< "${SPECTRUM_LEADS}"
  SPECTRUM_ARGS=(--spectrum-leads "${SPECTRUM_LEAD_ARGS[@]}" --spectrum-output "${SPEC_PATH}")
fi

python3 build_scorecard.py \
  --model-checkpoint "${MODEL_CHECKPOINT}" \
  --apath "${APATH}" \
  --norm-factors "${NORM_FACTORS}" \
  --cpath "${CLIMATO_PATH}" \
  --start-date "${START_DATE}" \
  --end-date "${END_DATE}" \
  --forecast-length "${FORECAST_LENGTH}" \
  --init-interval "${INIT_INTERVAL}" \
  --to-path "${TO_PATH}" \
  "${SPECTRUM_ARGS[@]}"
