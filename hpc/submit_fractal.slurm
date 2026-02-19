#!/bin/bash
# ============================================================================
# SLURM Job: Fractal Dimension + K-Means Classification (parameterized)
#
# Takes a binary edge map and computes multiscale edge density fractal:
#   1. Fractal dimension map: D = 2 + slope(log(density) vs log(R))
#   2. K-Means unsupervised classification (k=10) on density vectors
#   3. Fractal-binned classification
#
# Required env vars (pass via sbatch --export):
#   INPUT          — path to Phase 2 edge map (*_edges.tif)
#   OUTPUT_PREFIX  — output path prefix
#
# Optional env vars:
#   RADII          — space-separated radii (default: "8 16 32 64 128")
#
# Submit AFTER Phase 2:
#   sbatch --dependency=afterok:<phase2_jobid> submit_fractal.slurm
# ============================================================================

#SBATCH --account=sbyrne
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128gb
#SBATCH --time=12:00:00
#SBATCH --output=fractal_%x_%j.out
#SBATCH --error=fractal_%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=phillipsm@arizona.edu

# ---- Configuration ----
WORK_DIR="/groups/sbyrne/phillipsm/mars_js"
SCRIPT="${WORK_DIR}/scripts/js_fractal_classify.py"
ENV_NAME="mars_js"

RADII="${RADII:-8 16 32 64 128}"

# ---- Validate required vars ----
if [ -z "${INPUT}" ] || [ -z "${OUTPUT_PREFIX}" ]; then
    echo "ERROR: INPUT and OUTPUT_PREFIX must be set via --export"
    exit 1
fi

# ---- Environment ----
module load micromamba
source ~/.bashrc
micromamba activate "${ENV_NAME}"

echo "============================================"
echo "Fractal + K-Means Classification"
echo "============================================"
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Job Name:  ${SLURM_JOB_NAME}"
echo "Node:      $(hostname)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Started:   $(date)"
echo "Input:     ${INPUT}"
echo "Output:    ${OUTPUT_PREFIX}"
echo "Radii:     ${RADII}"
echo "============================================"

if [ ! -f "${INPUT}" ]; then
    echo "ERROR: Input file not found: ${INPUT}"
    echo "Make sure Phase 2 (edge detection) completed successfully."
    exit 1
fi

# Create output directory
mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

# ---- Run fractal + classification ----
python3 "${SCRIPT}" \
    "${INPUT}" \
    -o "${OUTPUT_PREFIX}" \
    --radii ${RADII} \
    --n-clusters 10 \
    --n-samples 2000000 \
    --tile-size 1024 \
    --compress lzw \
    --log-level INFO

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Finished:  $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
echo "Output files:"
ls -lh "${OUTPUT_PREFIX}"_fractal*.tif "${OUTPUT_PREFIX}"_kmeans*.tif 2>/dev/null

exit ${EXIT_CODE}
