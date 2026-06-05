#!/usr/bin/env bash
set -euo pipefail

# Build aggregate comparison plots including the MSE-5000 control from existing
# scorecard zarr outputs. This reads zarr stores directly, so it is safe even if
# a partial MODELS_TO_RUN submission overwrote scorecard_manifest.tsv.
#
# Typical use after all scorecard jobs, including MSE-5000, finish:
#   bash scripts/plot_mse_scorecard_comparison.sh

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${PROJECT_DIR}"

EVAL_ROOT="${EVAL_ROOT:-runs/eval/2022_full_i12}"
SUMMARY_CSV="${SUMMARY_CSV:-${EVAL_ROOT}/scorecard_summary_with_mse.csv}"
OUT_DIR="${OUT_DIR:-${EVAL_ROOT}/plots_with_mse}"
FIELDS="${FIELDS:-z:500 t:850 2t 10m_wind_speed msl}"
STAT="${STAT:-std}"
METRIC="${METRIC:-mean}"
REFERENCE_MODEL="${REFERENCE_MODEL:-amse5000}"
MSE_SCORE_PATH="${MSE_SCORE_PATH:-${EVAL_ROOT}/mse5000_score_2022.zarr}"

declare -a SPECS=(
  "control|${EVAL_ROOT}/control_score_2022.zarr|required"
  "mse5000|${MSE_SCORE_PATH}|required"
  "amse5000|${EVAL_ROOT}/amse5000_score_2022.zarr|required"
  "amse25000|${EVAL_ROOT}/amse25000_score_2022.zarr|optional"
  "lamse0p1_lmax32_5000|${EVAL_ROOT}/lamse0p1_lmax32_5000_score_2022.zarr|optional"
  "lamse0p5_lmax127_5000|${EVAL_ROOT}/lamse0p5_lmax127_5000_score_2022.zarr|optional"
  "lamse0p3_lmax32_5000|${EVAL_ROOT}/lamse0p3_lmax32_5000_score_2022.zarr|optional"
)

declare -a SCORE_ARGS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read -r model path requiredness <<< "${spec}"
  if [[ -e "${path}" ]]; then
    SCORE_ARGS+=(--score "${model}=${path}")
  elif [[ "${requiredness}" == "required" ]]; then
    cat >&2 <<EOF
Missing required scorecard zarr for ${model}: ${path}

If the MSE checkpoint is trained but not evaluated yet, run:
  INCLUDE_MSE=1 MODELS_TO_RUN="mse5000" OUTPUT_ROOT=${EVAL_ROOT} \\
    bash scripts/submit_scorecard_2022_all.sh
EOF
    exit 2
  else
    echo "Skipping optional missing scorecard: ${model} (${path})" >&2
  fi
done

read -r -a FIELD_ARGS <<< "${FIELDS}"

python3 scripts/summarize_scorecards.py \
  "${SCORE_ARGS[@]}" \
  --out-csv "${SUMMARY_CSV}"

python3 scripts/plot_scorecard_summary.py \
  --summary-csv "${SUMMARY_CSV}" \
  --out-dir "${OUT_DIR}" \
  --fields "${FIELD_ARGS[@]}" \
  --stat "${STAT}" \
  --metric "${METRIC}" \
  --reference-model "${REFERENCE_MODEL}"

echo "Summary CSV: ${SUMMARY_CSV}"
echo "Plots: ${OUT_DIR}"
