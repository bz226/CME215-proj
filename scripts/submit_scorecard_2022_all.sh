#!/usr/bin/env bash
set -euo pipefail

# Submit full-year 2022 scorecard/spectral jobs for the main checkpoints.
#
# Default model set:
#   control, AMSE-5000, AMSE-25000, LAMSE-0.1-LMAX32-5000,
#   LAMSE-0.5-LMAX127-5000.
#
# Submit from the project root:
#   bash scripts/submit_scorecard_2022_all.sh
#
# Common overrides:
#   INIT_INTERVAL=24 bash scripts/submit_scorecard_2022_all.sh
#   MODELS_TO_RUN="amse5000 lamse0p5_lmax127_5000" bash scripts/submit_scorecard_2022_all.sh
#   CLIMATO_PATH=/path/to/era5_1deg_climatology.zarr bash scripts/submit_scorecard_2022_all.sh
#   SKIP_MISSING=1 bash scripts/submit_scorecard_2022_all.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

if [[ -z "${SCRATCH:-}" && -z "${APATH:-}" ]]; then
  echo "Set APATH, or run on Sherlock with SCRATCH set." >&2
  exit 2
fi

APATH="${APATH:-${SCRATCH}/graphcast-small-lamse/era5_1deg_weatherbench2_2022}"
if [[ -z "${PARAMS_DIR:-}" ]]; then
  if [[ -z "${SCRATCH:-}" ]]; then
    echo "Set PARAMS_DIR, or run on Sherlock with SCRATCH set." >&2
    exit 2
  fi
  PARAMS_DIR="${SCRATCH}/graphcast-small-lamse/params"
fi
START_DATE="${START_DATE:-1 Jan 2022 00:00}"
END_DATE="${END_DATE:-31 Dec 2022 18:00}"
FORECAST_LENGTH="${FORECAST_LENGTH:-240}"
INIT_INTERVAL="${INIT_INTERVAL:-12}"
SPECTRUM_LEADS="${SPECTRUM_LEADS:-6 120 240}"
# Do not use CPATH for climatology here: Sherlock modules set CPATH to compiler
# include paths, which would be misread by build_scorecard.py as a zarr store.
CLIMATO_PATH="${CLIMATO_PATH:-${SCORECARD_CPATH:-none}}"
NORM_FACTORS="${NORM_FACTORS:-stats}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/eval/2022_full_i${INIT_INTERVAL}}"
TIME="${TIME:-12:00:00}"
SKIP_MISSING="${SKIP_MISSING:-0}"
MODELS_TO_RUN="${MODELS_TO_RUN:-}"

mkdir -p "${OUTPUT_ROOT}"
MANIFEST="${OUTPUT_ROOT}/scorecard_manifest.tsv"
printf "model\tcheckpoint\tscore_path\tspectrum_path\tjob_id\n" > "${MANIFEST}"

declare -a MODELS=(
  "control|${PARAMS_DIR}/graphcast_small_lamse.000000.npz"
  "amse5000|${PARAMS_DIR}/graphcast_small_amse.005000.npz"
  "amse25000|${PARAMS_DIR}/graphcast_small_amse.025000.npz"
  "lamse0p1_lmax32_5000|${PARAMS_DIR}/graphcast_small_lamse_lam0p1_lmax32.005000.npz"
  "lamse0p5_lmax127_5000|${PARAMS_DIR}/graphcast_small_lamse_lam0p5_lmax127.005000.npz"
)

if [[ "${INCLUDE_LAMSE0P3:-0}" == "1" ]]; then
  MODELS+=("lamse0p3_lmax32_5000|${PARAMS_DIR}/graphcast_small_lamse_lam0p3_lmax32.005000.npz")
fi

want_model() {
  local model="$1"
  if [[ -z "${MODELS_TO_RUN}" ]]; then
    return 0
  fi
  [[ " ${MODELS_TO_RUN} " == *" ${model} "* ]]
}

for spec in "${MODELS[@]}"; do
  IFS='|' read -r model checkpoint <<< "${spec}"
  if ! want_model "${model}"; then
    continue
  fi

  score_path="${OUTPUT_ROOT}/${model}_score_2022.zarr"
  spec_path="${OUTPUT_ROOT}/${model}_spec_2022.zarr"

  if [[ ! -f "${checkpoint}" ]]; then
    msg="Missing checkpoint for ${model}: ${checkpoint}"
    if [[ "${SKIP_MISSING}" == "1" ]]; then
      echo "Skipping: ${msg}" >&2
      continue
    fi
    echo "${msg}" >&2
    exit 2
  fi

  job_id="$(sbatch --parsable \
    --job-name="gc-score-${model}" \
    --time="${TIME}" \
    --export=ALL,PROJECT_DIR="${PROJECT_DIR}",MODEL_NAME="${model}",MODEL_CHECKPOINT="${checkpoint}",APATH="${APATH}",NORM_FACTORS="${NORM_FACTORS}",CLIMATO_PATH="${CLIMATO_PATH}",START_DATE="${START_DATE}",END_DATE="${END_DATE}",FORECAST_LENGTH="${FORECAST_LENGTH}",INIT_INTERVAL="${INIT_INTERVAL}",SPECTRUM_LEADS="${SPECTRUM_LEADS}",OUTPUT_ROOT="${OUTPUT_ROOT}",TO_PATH="${score_path}",SPEC_PATH="${spec_path}" \
    scripts/sbatch_scorecard_2022.sh)"

  printf "%s\t%s\t%s\t%s\t%s\n" "${model}" "${checkpoint}" "${score_path}" "${spec_path}" "${job_id}" >> "${MANIFEST}"
  echo "Submitted ${model}: ${job_id}"
done

echo "Manifest: ${MANIFEST}"
echo "Watch with:"
echo "  squeue -u \"\$USER\" -n 'gc-score-*'"
echo "  tail -n 80 logs/gc-score-*.log"
