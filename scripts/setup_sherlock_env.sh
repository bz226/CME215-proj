#!/usr/bin/env bash
# Set up a Stanford Sherlock Python virtual environment for GraphCast Small LAMSE.
#
# Usage on Sherlock:
#   sh_dev
#   cd /path/to/code/graphcast-small-lamse
#   bash scripts/setup_sherlock_env.sh
#
# Optional overrides:
#   ENV_DIR=/path/to/local/venv bash scripts/setup_sherlock_env.sh
#   PYTHON_MODULE=python/3.12.1 CUDA_MODULE=cuda/12.6.1 GCC_MODULE=gcc/12.4.0 bash scripts/setup_sherlock_env.sh
#
# Notes:
# - Sherlock recommends module-provided system libraries plus Python venvs over
#   Conda. This script therefore uses ml/module and python -m venv.
# - Per Sherlock docs, create/install environments from a compute node, for
#   example via `sh_dev`, not from a login node.
# - Inside an activated venv, do not use `pip install --user`; pip installs into
#   the venv directory.
# - Module versions change. If PYTHON_MODULE or CUDA_MODULE is unavailable,
#   run `ml av python` and `ml av cuda` on Sherlock and rerun with overrides.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-${PROJECT_DIR}/.venv-sherlock}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12.1}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.6.1}"
GCC_MODULE="${GCC_MODULE:-gcc/12.4.0}"
CUDNN_MODULE="${CUDNN_MODULE:-}"
NCCL_MODULE="${NCCL_MODULE:-}"
JAX_CUDA_EXTRA="${JAX_CUDA_EXTRA:-cuda12-local}"
INSTALL_CARTOPY="${INSTALL_CARTOPY:-0}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_DIR}/scripts/requirements_sherlock.txt}"
NUMPY_VERSION="${NUMPY_VERSION:-1.26.4}"
SCIPY_VERSION="${SCIPY_VERSION:-1.12.0}"
ML_DTYPES_VERSION="${ML_DTYPES_VERSION:-0.4.0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1}"

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
  ml av gcc
  ml av cudnn
  ml av nccl

Then rerun with overrides, for example:
  PYTHON_MODULE=python/<version> CUDA_MODULE=cuda/<version> GCC_MODULE=gcc/<version> bash scripts/setup_sherlock_env.sh

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

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  cat >&2 <<EOF

Warning: SLURM_JOB_ID is not set.
Sherlock recommends creating virtual environments and installing packages on a
compute node, not a login node. Start an interactive compute session with:

  sh_dev

Then rerun this script from the allocation.

Continuing anyway.
EOF
fi

log "Resetting modules and loading Sherlock modules"
ml reset
load_module_or_explain "${PYTHON_MODULE}"
load_module_or_explain "${CUDA_MODULE}"
load_module_or_explain "${GCC_MODULE}"
load_module_or_explain "${CUDNN_MODULE}"
load_module_or_explain "${NCCL_MODULE}"

need_command python3
log "Python executable: $(command -v python3)"
python3 --version
log "CUDA compiler, if available"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version | tail -n 1
else
  echo "nvcc not found; continuing because some CUDA modules expose runtime libraries only."
fi

log "Creating virtual environment at ${ENV_DIR}"
mkdir -p "$(dirname "${ENV_DIR}")"
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

log "Upgrading packaging tools"
python -m pip install --upgrade pip setuptools wheel

log "Installing binary NumPy/SciPy stack before JAX"
python -m pip install --only-binary=:all: \
  "numpy==${NUMPY_VERSION}" \
  "scipy==${SCIPY_VERSION}" \
  "ml-dtypes==${ML_DTYPES_VERSION}" \
  "opt-einsum==3.3.0"

log "Installing JAX for Sherlock CUDA modules: jax[${JAX_CUDA_EXTRA}]==0.4.30"
python -m pip install --upgrade "jax[${JAX_CUDA_EXTRA}]==0.4.30"

log "Installing GraphCast, AMSE, and LAMSE dependencies from ${REQUIREMENTS_FILE}"
python -m pip install -r "${REQUIREMENTS_FILE}"

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
if [[ -n "${GCC_MODULE}" ]]; then
  echo "ml load ${GCC_MODULE}" >> "${ACTIVATE_FILE}"
fi
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
export JAX_ENABLE_X64=True
export XLA_FLAGS="${XLA_FLAGS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=True
EOF
chmod +x "${ACTIVATE_FILE}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_ENABLE_X64=True
export XLA_FLAGS="${XLA_FLAGS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}"
export TF_CPP_MIN_LOG_LEVEL=1
export PYTHONUNBUFFERED=True

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
