#!/usr/bin/env bash
set -euo pipefail

# Submit a 5000-batch LAMSE run with lambda=0.5 and LMAX=127.
#
# Fresh run:
#   bash scripts/submit_final_lamse_lam0p5_lmax127.sh
#
# Resume from an existing checkpoint:
#   START_BATCH=1000 bash scripts/submit_final_lamse_lam0p5_lmax127.sh
#
# Expected fresh-run artifacts:
#   $PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz
#   $PARAMS_DIR/graphcast_small_lamse_lam0p5_lmax127.005000.npz.opt
#   runs/lamse_lam0p5_lmax127_final_000000_to_005000.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LAMSE_LAMBDA="0.5"
export LAMSE_LMAX="127"
export LAMSE_TAG="lam0p5_lmax127"
export FINAL_BATCH="${FINAL_BATCH:-5000}"

# LMAX=127 gives s2fft bandlimit 128, so the automatic input_nside=64 is
# valid under the HEALPix L >= 2*nside constraint. The graph is heavier than
# LMAX=32/96, so default to a longer compile watchdog and allocation.
export FIRST_STEP_TIMEOUT="${FIRST_STEP_TIMEOUT:-10800}"
export WATCHDOG_TIMEOUT="${WATCHDOG_TIMEOUT:-600}"
export TIME="${TIME:-48:00:00}"

exec bash "${SCRIPT_DIR}/submit_final_lamse.sh"
