#!/bin/bash
# ============================================================================
# Setup micromamba environment on UArizona HPC for JS edge detection
#
# Run this ONCE from an interactive session:
#   interactive -a sbyrne -n 4 -m 16gb -t 1:00:00
#   bash setup_env.sh
# ============================================================================

set -eo pipefail

# --- Configuration ---
ENV_NAME="mars_js"
ENV_PREFIX="/groups/sbyrne/phillipsm/micromamba"

echo "=== Setting up micromamba environment: ${ENV_NAME} ==="

# Load micromamba
module load micromamba

# Initialize micromamba if not already done
if [ ! -f ~/.bashrc ] || ! grep -q "micromamba" ~/.bashrc 2>/dev/null; then
    micromamba shell init -s bash -r "${ENV_PREFIX}"
    echo "export MAMBA_ROOT_PREFIX=${ENV_PREFIX}" >> ~/.bashrc
    source ~/.bashrc
fi

export MAMBA_ROOT_PREFIX="${ENV_PREFIX}"

# Configure channels
micromamba config append channels conda-forge 2>/dev/null || true
micromamba config append channels nodefaults 2>/dev/null || true
micromamba config set channel_priority strict 2>/dev/null || true

# Create environment with all dependencies
echo "Creating environment ${ENV_NAME}..."
micromamba create -y -n "${ENV_NAME}" \
    python=3.11 \
    numpy \
    scipy \
    numba \
    rasterio \
    gdal \
    scikit-learn \
    joblib \
    geopandas \
    matplotlib

echo ""
echo "=== Environment created successfully ==="
echo ""
echo "To activate in batch scripts, add:"
echo "  module load micromamba"
echo "  source ~/.bashrc"
echo "  micromamba activate ${ENV_NAME}"
echo ""
echo "Or use: micromamba run -n ${ENV_NAME} python3 your_script.py"
echo ""

# Quick verification
micromamba run -n "${ENV_NAME}" python3 -c "
import numpy, scipy, numba, rasterio, sklearn
print(f'numpy:    {numpy.__version__}')
print(f'scipy:    {scipy.__version__}')
print(f'numba:    {numba.__version__}')
print(f'rasterio: {rasterio.__version__}')
print(f'sklearn:  {sklearn.__version__}')
print('All imports OK')
"

echo "=== Setup complete ==="
