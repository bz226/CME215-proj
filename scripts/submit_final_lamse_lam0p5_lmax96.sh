#!/usr/bin/env bash
set -euo pipefail

# Submit a 5000-batch LAMSE run with lambda=0.5 and LMAX=96.
#
# Fresh run:
#   bash scripts/submit_final_lamse_lam0p5_lmax96.sh
#
# Resume from an existing checkpoint:
#   START_BATCH=1000 bash scripts/submit_final_lamse_lam0p5_lmax96.sh
#
# Expected fresh-run artifacts:
#   $PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax96.005000.npz
#   $PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax96.005000.npz.opt
#   runs/lamse_lam0p5_lmax96_final_000000_to_005000.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAMSE_LAMBDA="0.5"
export LAMSE_LMAX="96"
export LAMSE_INPUT_NSIDE="${LAMSE_INPUT_NSIDE:-32}"
export LAMSE_TAG="lam0p5_lmax96"
export FINAL_BATCH="${FINAL_BATCH:-5000}"

# LMAX=96 needs input_nside=32 under the current s2fft HEALPix constraint
# L >= 2*nside. It also builds a larger graph than LMAX=32, so give the first
# JAX compile substantially more room by default.
export FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT:-7200}"
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-300}"
export TIME="${TIME:-36:00:00}"

exec bash "${SCRIPT_DIR}/submit_final_lamse.sh"
