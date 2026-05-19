#!/usr/bin/env bash
# Set up a Stanford Sherlock Python virtual environment for GraphCast Small LAMSE.
#
# Usage on Sherlock:
#   cd /path/to/code/graphcast-small-lamse
#   bash scripts/setup_sherlock_env.sh
#
# Optional overrides:
#   ENV_DIR=$GROUP_HOME/venvs/graphcast-small-lamse bash scripts/setup_sherlock_env.sh
#   PYTHON_MODULE=python/3.12.1 CUDA_MODULE=cuda/12.6.1 bash scripts/setup_sherlock_env.sh
#
# Notes:
# - Sherlock recommends module-provided system libraries plus Python venvs over
#   Conda. This script therefore uses ml/module and python -m venv.
# - Module versions change. If PYTHON_MODULE or CUDA_MODULE is unavailable,
#   run `ml av python` and `ml av cuda` on Sherlock and rerun with overrides.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-${SCRATCH:-$HOME}/venvs/graphcast-small-lamse}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.1}"
CUDNN_MODULE="${CUDNN_MODULE:-}"
NCCL_MODULE="${NCCL_MODULE:-}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12-local}"
INSTALL_CARTOPY="${INSTALL_CARTOPY:-0}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing required command: $1" >&2
    exit 1
  }
}

load_module_or_explain() {
  local mod="$1"
  if [[ -z "${mod}" ]]; then
    return 0
  fi
  if ml load "${mod}"; then
    return 0
  fi
  cat >&2 <<EOF

Could not load module: ${mod}

On Sherlock, check available modules with:
  ml av python
  ml av cuda
  ml av cudnn
  ml av nccl

Then rerun with overrides, for example:
  PYTHON_MODULE=python/<version> CUDA_MODULE=cuda/<version> bash scripts/setup_sherlock_env.sh

EOF
  exit 1
}

if ! command -v ml >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
fi
need_command ml

log "Resetting modules and loading Sherlock modules"
ml reset
load_module_or_explain "${PYTHON_MODULE}"
load_module_or_explain "${CUDA_MODULE}"
load_module_or_explain "${CUDNN_MODULE}"
load_module_or_explain "${NCCL_MODULE}"

log "Python executable: $(command -v python)"
python --version
log "CUDA compiler, if available"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
else
  echo "nvcc not found; continuing because some CUDA modules expose runtime libraries only."
fi

log "Creating virtual environment at ${ENV_DIR}"
mkdir -p "$(dirname "${ENV_DIR}")"
python -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

log "Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

log "Installing JAX for Sherlock CUDA modules: jax[${JAX_CUDA_EXTRA}]==0.4.30"
python -m pip install --upgrade "jax[${JAX_CUDA_EXTRA}]==0.4.30"

log "Installing GraphCast, AMSE, and LAMSE Python dependencies"
python -m pip install \
  "absl-py==2.1.0" \
  "cads-api-client==1.0.3" \
  "cdsapi==0.7.0" \
  "chex==0.1.86" \
  "cf_xarray==0.8.8" \
  "dask[distributed]==2024.1.0" \
  "dateparser==1.2.0" \
  "dm-haiku==0.0.12" \
  "dm-tree==0.1.8" \
  "eccodes==1.7.1" \
  "ecmwf-api-client==1.6.3" \
  "ecmwflibs==0.6.3" \
  "etils==1.9.2" \
  "findlibs==0.0.5" \
  "flax==0.8.5" \
  "fsspec==2023.12.2" \
  "gcsfs==2023.12.2.post1" \
  "h5netcdf==1.3.0" \
  "h5py==3.10.0" \
  "healpy==1.17.3" \
  "huggingface_hub>=0.23" \
  "immutabledict==4.1.0" \
  "jmp==0.0.4" \
  "jraph==0.0.6.dev0" \
  "netCDF4==1.6.5" \
  "numcodecs==0.12.1" \
  "numpy<2" \
  "optax==0.2.2" \
  "orbax-checkpoint==0.5.20" \
  "pandas>=2.0,<2.3" \
  "pytest>=7" \
  "rich==13.7.1" \
  "s2fft==1.3.0" \
  "scipy>=1.11,<1.14" \
  "tabulate==0.9.0" \
  "tensorstore==0.1.63" \
  "xarray==2024.1.1" \
  "zarr==2.16.1"

if [[ "${INSTALL_CARTOPY}" == "1" ]]; then
  log "Installing optional plotting/geospatial packages"
  python -m pip install "cartopy==0.22.0"
fi

ACTIVATE_FILE="${ENV_DIR}/activate_graphcast_small_lamse.sh"
log "Writing activation helper to ${ACTIVATE_FILE}"
cat > "${ACTIVATE_FILE}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
if ! command -v ml >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
ml reset
ml load ${PYTHON_MODULE}
ml load ${CUDA_MODULE}
EOF
if [[ -n "${CUDNN_MODULE}" ]]; then
  echo "ml load ${CUDNN_MODULE}" >> "${ACTIVATE_FILE}"
fi
if [[ -n "${NCCL_MODULE}" ]]; then
  echo "ml load ${NCCL_MODULE}" >> "${ACTIVATE_FILE}"
fi
cat >> "${ACTIVATE_FILE}" <<EOF
source "${ENV_DIR}/bin/activate"
export PYTHONPATH="${PROJECT_DIR}:\${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_CPP_MIN_LOG_LEVEL=1
EOF
chmod +x "${ACTIVATE_FILE}"

log "Running import smoke checks"
python - <<'PY'
import jax
import s2fft
import healpy
import xarray
import haiku
import optax
print("jax", jax.__version__)
print("jax devices", jax.devices())
print("s2fft", getattr(s2fft, "__version__", "unknown"))
print("healpy", healpy.__version__)
print("xarray", xarray.__version__)
print("haiku", haiku.__version__)
print("optax", optax.__version__)
PY

log "Running lightweight repo test"
cd "${PROJECT_DIR}"
PYTHONPATH=. pytest tests/test_loss_cli_config.py -q

cat <<EOF

Sherlock environment setup complete.

To activate later:
  source "${ACTIVATE_FILE}"

Then prepare the checkpoint:
  cd "${PROJECT_DIR}"
  python scripts/download_graphcast_small.py
  python scripts/prepare_graphcast_small_checkpoint.py
  python scripts/inspect_graphcast_checkpoint.py

For jobs, source the activation helper inside the sbatch script after #SBATCH lines.
EOF
