#!/bin/bash
# ============================================================================
# SLURM Job: Phase 2 — Binary Edge Map from multi-band JS Divergence
#
# Reads the multi-band JS output, computes nanmean across bands, applies Otsu
# thresholding to produce a binary edge map.
#
# Required env vars (pass via sbatch --export):
#   JS_INPUT     — path to Phase 1 JS output (*_js.tif)
#   EDGE_OUTPUT  — path for binary edge output (*_edges.tif)
#
# Submit AFTER Phase 1:
#   sbatch --dependency=afterok:<phase1_jobid> submit_phase2_edges.slurm
# ============================================================================

#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64gb
#SBATCH --time=04:00:00
#SBATCH --output=phase2_%x_%j.out
#SBATCH --error=phase2_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=phillipsm@arizona.edu

# ---- Configuration ----
WORK_DIR="/groups/sbyrne/phillipsm/mars_js"
ENV_NAME="mars_js"

# ---- Validate required vars ----
if [ -z "${JS_INPUT}" ] || [ -z "${EDGE_OUTPUT}" ]; then
    echo "ERROR: JS_INPUT and EDGE_OUTPUT must be set via --export"
    exit 1
fi

# ---- Environment ----
module load micromamba
source ~/.bashrc
micromamba activate "${ENV_NAME}"

echo "============================================"
echo "Phase 2: Binary Edge Map"
echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "JS Input:  ${JS_INPUT}"
echo "Edge Out:  ${EDGE_OUTPUT}"
echo "============================================"

if [ ! -f "${JS_INPUT}" ]; then
    echo "ERROR: JS input not found: ${JS_INPUT}"
    echo "Make sure Phase 1 completed successfully."
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "${EDGE_OUTPUT}")"

# Phase 2: accumulate Otsu histogram from JS file, then threshold
python3 -c "
import sys
sys.path.insert(0, '${WORK_DIR}/scripts')
from js_edge_detect import run_phase2
import numpy as np
import rasterio
import math
from rasterio.windows import Window
import logging

logging.basicConfig(level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('phase2')

js_path = '${JS_INPUT}'
edge_path = '${EDGE_OUTPUT}'

# Accumulate histogram from the JS file
logger.info('Accumulating Otsu histogram from JS file...')
OTSU_HIST_BINS = 10000
hist_counts = np.zeros(OTSU_HIST_BINS, dtype=np.int64)
hist_edges = np.linspace(0.0, 1.0, OTSU_HIST_BINS + 1)

with rasterio.open(js_path) as src:
    n_bands = src.count
    block_size = 1024
    n_rows = math.ceil(src.height / block_size)
    n_cols = math.ceil(src.width / block_size)
    for br in range(n_rows):
        rs = br * block_size
        re = min(rs + block_size, src.height)
        for bc in range(n_cols):
            cs = bc * block_size
            ce = min(cs + block_size, src.width)
            w = Window(cs, rs, ce - cs, re - rs)
            if n_bands > 1:
                data = np.nanmean(src.read(window=w), axis=0).astype(np.float32)
            else:
                data = src.read(1, window=w)
            valid = data[~np.isnan(data)]
            if valid.size > 0:
                c, _ = np.histogram(valid, bins=hist_edges)
                hist_counts += c
        if br % 20 == 0:
            logger.info(f'  Histogram row {br}/{n_rows}')

logger.info('Running Phase 2...')
run_phase2(js_path, edge_path, hist_counts, hist_edges, compress='lzw')
logger.info('Done.')
"

EXIT_CODE=$?

echo ""
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
ls -lh "${EDGE_OUTPUT}" 2>/dev/null

exit ${EXIT_CODE}
