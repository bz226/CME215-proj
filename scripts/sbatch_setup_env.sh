#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/sbatch_setup_env.sh
#
# Purpose:
#   Build/update the project-local .venv-sherlock environment on a Sherlock
#   compute node. This uses a CPU batch allocation and skips CUDA device discovery.

#SBATCH --job-name=gc-lamse-env
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

if [[ -z "${PROJECT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/scripts/setup_sherlock_env.sh" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  else
    PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  fi
fi

if [[ ! -f "${PROJECT_DIR}/scripts/setup_sherlock_env.sh" ]]; then
  cat >&2 <<EOF
Could not locate the graphcast-small-lamse project directory.

Submit from the project root:
  cd /path/to/graphcast-small-lamse
  sbatch scripts/sbatch_setup_env.sh

Or pass PROJECT_DIR explicitly:
  PROJECT_DIR=/path/to/graphcast-small-lamse sbatch scripts/sbatch_setup_env.sh
EOF
  exit 2
fi

cd "${PROJECT_DIR}"

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${SLURM_JOB_NAME:-gc-lamse-env}-${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"

RUN_JAX_SMOKE="${RUN_JAX_SMOKE:-0}" \
  bash scripts/setup_sherlock_env.sh

echo "Environment setup finished."
echo "Activate with:"
echo "  source ${PROJECT_DIR}/.venv-sherlock/activate_graphcast_small_lamse.sh"
