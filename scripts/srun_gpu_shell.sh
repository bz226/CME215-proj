#!/usr/bin/env bash
# Start an interactive Sherlock GPU shell.
#
# Usage:
#   bash scripts/srun_gpu_shell.sh
#
# Optional overrides:
#   PARTITION=serc GPUS=1 TIME=02:00:00 CPUS=8 MEM=256G bash scripts/srun_gpu_shell.sh
#   CONSTRAINT='GPU_MEM:80GB' bash scripts/srun_gpu_shell.sh

set -euo pipefail

PARTITION="${PARTITION:-serc}"
GPUS="${GPUS:-1}"
TIME="${TIME:-02:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-256G}"
CONSTRAINT="${CONSTRAINT:-}"

cmd=(
  srun
  --partition="${PARTITION}"
  --gpus="${GPUS}"
  --time="${TIME}"
  --cpus-per-task="${CPUS}"
  --mem="${MEM}"
)

if [[ -n "${CONSTRAINT}" ]]; then
  cmd+=(--constraint="${CONSTRAINT}")
fi

cmd+=(--pty bash -l)

printf 'Launching interactive GPU allocation:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
