#!/bin/bash
# ============================================================================
# SLURM Job: Phase 1 — Multi-band JS Divergence (parameterized)
#
# Computes 5-band JS divergence (one band per radius: 5, 7, 10, 14, 20 px).
# Accepts any single-band raster (uint8, float32, int16, etc.)
#
# Required env vars (pass via sbatch --export):
#   INPUT          — path to input GeoTIFF
#   OUTPUT_PREFIX  — output path prefix (produces <prefix>_js.tif)
#
# Optional env vars:
#   NODATA         — nodata value: "auto" (default), "nan", "none", or number
#   BAND           — band number to read (default: 1)
#   RADII          — space-separated radii (default: "5 7 10 14 20")
#   RESUME_ROW     — strip row to resume from (default: 0)
#
# Resource defaults below are conservative; override per-dataset via sbatch:
#   sbatch --cpus-per-task=48 --mem=128gb --time=12:00:00 submit_phase1.slurm
# ============================================================================

#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=94
#SBATCH --mem=256gb
#SBATCH --time=48:00:00
#SBATCH --output=phase1_%x_%j.out
#SBATCH --error=phase1_%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=phillipsm@arizona.edu

# ---- Configuration ----
WORK_DIR="/groups/sbyrne/phillipsm/mars_js"
SCRIPT="${WORK_DIR}/scripts/js_edge_detect.py"
ENV_NAME="mars_js"

# Apply defaults for optional vars
NODATA="${NODATA:-auto}"
BAND="${BAND:-1}"
RADII="${RADII:-5 7 10 14 20}"
RESUME_ROW="${RESUME_ROW:-0}"
LOCAL_RESCALE="${LOCAL_RESCALE:-0}"

# ---- Validate required vars ----
if [ -z "${INPUT}" ] || [ -z "${OUTPUT_PREFIX}" ]; then
    echo "ERROR: INPUT and OUTPUT_PREFIX must be set via --export"
    echo "Usage: sbatch --export=INPUT=/path/to/input.tif,OUTPUT_PREFIX=/path/to/prefix submit_phase1.slurm"
    exit 1
fi

# ---- Environment ----
module load micromamba
source ~/.bashrc
micromamba activate "${ENV_NAME}"

# Let numba use all available cores
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---- Diagnostics ----
echo "============================================"
echo "Phase 1: JS Divergence"
echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Memory:    ${SLURM_MEM_PER_NODE}"
echo "Started:   $(date)"
echo "Input:     ${INPUT}"
echo "Output:    ${OUTPUT_PREFIX}"
echo "NoData:    ${NODATA}"
echo "Band:      ${BAND}"
echo "Radii:     ${RADII}"
echo "LocalResc: ${LOCAL_RESCALE}"
echo "============================================"

# Verify input exists
if [ ! -f "${INPUT}" ]; then
    echo "ERROR: Input file not found: ${INPUT}"
    exit 1
fi

# Print raster metadata for the log
echo ""
echo "--- Input raster info ---"
gdalinfo -stats "${INPUT}" 2>/dev/null | head -30 || echo "(gdalinfo not available)"
echo "-------------------------"
echo ""

# Create output directory
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

# ---- Build nodata flag ----
NODATA_FLAG=""
if [ "${NODATA}" != "auto" ]; then
    NODATA_FLAG="--nodata=${NODATA}"
fi

# ---- Build resume flag ----
RESUME_FLAG=""
if [ "${RESUME_ROW}" -gt 0 ] 2>/dev/null; then
    RESUME_FLAG="--resume-row ${RESUME_ROW}"
fi

# ---- Build local-rescale flag ----
LOCAL_RESCALE_FLAG=""
if [ "${LOCAL_RESCALE}" = "1" ]; then
    LOCAL_RESCALE_FLAG="--local-rescale"
fi

# ---- Run Phase 1: multi-band JS divergence ----
python3 "${SCRIPT}" \
    "${INPUT}" \
    -o "${OUTPUT_PREFIX}" \
    --radii ${RADII} \
    --band "${BAND}" \
    ${NODATA_FLAG} \
    --bins 32 \
    --tile-size 1024 \
    --compress lzw \
    --js-only \
    --log-level INFO \
    ${RESUME_FLAG} \
    ${LOCAL_RESCALE_FLAG}

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"

# Report output file sizes
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"_*.tif 2>/dev/null || echo "No output files found"

exit ${EXIT_CODE}
