#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/sbatch_setup_env.sh
#
# Purpose:
#   Build/update the project-local .venv-sherlock environment on a Sherlock
#   compute node. This uses a CPU allocation and skips CUDA device discovery.

#SBATCH --job-name=gc-lamse-env
#SBATCH --partition=dev
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Project: ${PROJECT_DIR}"

RUN_JAX_SMOKE="${RUN_JAX_SMOKE:-0}" \
  bash scripts/setup_sherlock_env.sh

echo "Environment setup finished."
echo "Activate with:"
echo "  source ${PROJECT_DIR}/.venv-sherlock/activate_graphcast_small_lamse.sh"
