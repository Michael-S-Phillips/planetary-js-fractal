#!/bin/bash
# ============================================================================
# Master Submission Script: JS Edge Detection Pipeline — All Datasets
#
# Submits Phase 1 (JS divergence), Phase 2 (binary edges), and Fractal
# classification for each input raster, with proper dependencies.
#
# Datasets:
#   1. THEMIS Day IR    — 213390x106696 uint8,   nodata=0       (~48h Phase 1)
#   2. THEMIS TI        — 35565x17783   float32, nodata=-3.4e38 (~1h  Phase 1)
#   3. HRSC-MOLA DEM    — 106694x53347  int16,   nodata=-32768  (~12h Phase 1)
#
# Prerequisites:
#   - Environment set up (bash setup_env.sh)
#   - Input files in /groups/sbyrne/phillipsm/mars_js/input/
#   - Output dir exists: mkdir -p /xdisk/sbyrne/phillipsm/mars_js
#
# Usage:
#   cd /groups/sbyrne/phillipsm/mars_js/slurm
#   bash submit_all.sh              # submit all 3 datasets
#   bash submit_all.sh ti           # submit only THEMIS TI
#   bash submit_all.sh dem          # submit only HRSC-MOLA DEM
#   bash submit_all.sh dayir        # submit only THEMIS Day IR
# ============================================================================

set -euo pipefail

WORK_DIR="/groups/sbyrne/phillipsm/mars_js"
OUT_DIR="/xdisk/sbyrne/phillipsm/mars_js"
SLURM_DIR="${WORK_DIR}/slurm"
INPUT_DIR="${WORK_DIR}/input"

# What to submit (default: all)
TARGETS="${1:-all}"

echo "============================================"
echo "Mars JS Edge Detection — Pipeline Submission"
echo "============================================"
echo "Output root: ${OUT_DIR}"
echo "Target(s):   ${TARGETS}"
echo ""

# Verify slurm scripts exist
for f in submit_phase1.slurm submit_phase2_edges.slurm submit_fractal.slurm; do
    if [ ! -f "${SLURM_DIR}/${f}" ]; then
        echo "ERROR: Missing ${SLURM_DIR}/${f}"
        exit 1
    fi
done

# Helper: submit one dataset through the full pipeline
submit_dataset() {
    local NAME="$1"        # short name for job labels
    local INPUT="$2"       # input raster path
    local PREFIX="$3"      # output prefix (inside OUT_DIR)
    local NODATA="$4"      # nodata value string
    local BAND="$5"        # band number
    local P1_CPUS="$6"     # Phase 1 CPUs
    local P1_MEM="$7"      # Phase 1 memory
    local P1_TIME="$8"     # Phase 1 walltime
    local FR_CPUS="$9"     # Fractal CPUs
    local FR_MEM="${10}"    # Fractal memory
    local FR_TIME="${11}"   # Fractal walltime
    local LOCAL_RESCALE="${12:-0}"  # 1 = per-tile normalization (recommended for DEMs)

    echo "--- ${NAME} ---"

    # Check input exists
    if [ ! -f "${INPUT}" ]; then
        echo "  WARNING: Input not found: ${INPUT}"
        echo "  Skipping ${NAME}."
        echo ""
        return
    fi

    # Create output directory
    mkdir -p "$(dirname "${PREFIX}")"

    # Pass env vars inline (avoids --export comma-parsing issues with
    # scientific notation like -3.4e+38 in nodata values)
    local JS_FILE="${PREFIX}_js.tif"
    local EDGE_FILE="${PREFIX}_edges.tif"

    # Phase 1: JS divergence
    local JID_P1
    JID_P1=$(INPUT="${INPUT}" OUTPUT_PREFIX="${PREFIX}" NODATA="${NODATA}" BAND="${BAND}" LOCAL_RESCALE="${LOCAL_RESCALE}" \
        sbatch --parsable --export=ALL \
        --job-name="${NAME}_p1" \
        --cpus-per-task="${P1_CPUS}" --mem="${P1_MEM}" --time="${P1_TIME}" \
        "${SLURM_DIR}/submit_phase1.slurm")
    echo "  Phase 1 (JS div):  job ${JID_P1}  [${P1_CPUS} CPUs, ${P1_MEM}, ${P1_TIME}]"

    # Phase 2: Binary edges (depends on Phase 1)
    local JID_P2
    JID_P2=$(JS_INPUT="${JS_FILE}" EDGE_OUTPUT="${EDGE_FILE}" \
        sbatch --parsable --export=ALL \
        --dependency=afterok:${JID_P1} \
        --job-name="${NAME}_p2" \
        --cpus-per-task=16 --mem=64gb --time=04:00:00 \
        "${SLURM_DIR}/submit_phase2_edges.slurm")
    echo "  Phase 2 (edges):   job ${JID_P2}  [after ${JID_P1}]"

    # Phase 3: Fractal + K-Means (depends on Phase 2 edge map)
    local JID_FR
    JID_FR=$(INPUT="${EDGE_FILE}" OUTPUT_PREFIX="${PREFIX}" \
        sbatch --parsable --export=ALL \
        --dependency=afterok:${JID_P2} \
        --job-name="${NAME}_fr" \
        --cpus-per-task="${FR_CPUS}" --mem="${FR_MEM}" --time="${FR_TIME}" \
        "${SLURM_DIR}/submit_fractal.slurm")
    echo "  Fractal (classify): job ${JID_FR}  [after ${JID_P2}]"
    echo ""
}

# ============================================================
# Dataset 1: THEMIS Day IR
#   213390 x 106696 px, uint8, nodata=0
#   Phase 1: ~48h with 94 CPUs
# ============================================================
if [ "${TARGETS}" = "all" ] || [ "${TARGETS}" = "dayir" ]; then
    submit_dataset \
        "dayir" \
        "${INPUT_DIR}/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12.tif" \
        "${OUT_DIR}/themis_dayir/themis_day_100m" \
        "auto" \
        "1" \
        "94" "256gb" "48:00:00" \
        "48" "128gb" "12:00:00"
fi

# ============================================================
# Dataset 2: THEMIS Thermal Inertia
#   35565 x 17783 px, float32, nodata=-3.4028226550889045e+38
#   Phase 1: ~1h with 48 CPUs (small raster, 632M pixels)
# ============================================================
if [ "${TARGETS}" = "all" ] || [ "${TARGETS}" = "ti" ]; then
    submit_dataset \
        "ti" \
        "${INPUT_DIR}/THEMIS_TI_Mosaic_Quant_60S300E_100mpp_ESRI104971.tif" \
        "${OUT_DIR}/themis_ti/themis_ti_100m" \
        "-3.4028226550889045e+38" \
        "1" \
        "48" "128gb" "04:00:00" \
        "48" "64gb" "02:00:00"
fi

# ============================================================
# Dataset 3: HRSC-MOLA Blended DEM
#   106694 x 53347 px, int16, nodata=-32768
#   Phase 1: ~12h with 94 CPUs (5.7B pixels, ~1/4 of THEMIS Day IR)
# ============================================================
if [ "${TARGETS}" = "all" ] || [ "${TARGETS}" = "dem" ]; then
    submit_dataset \
        "dem" \
        "${INPUT_DIR}/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif" \
        "${OUT_DIR}/hrsc_mola_dem/hrsc_mola_dem_200m" \
        "-32768" \
        "1" \
        "94" "256gb" "24:00:00" \
        "48" "128gb" "08:00:00" \
        "1"
fi

echo "============================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue --user=phillipsm"
echo ""
echo "Expected outputs in ${OUT_DIR}/:"
echo "  themis_dayir/     — themis_day_100m_{js,edges,fractal,kmeans_k10,fractal_classes}.tif"
echo "  themis_ti/        — themis_ti_100m_{js,edges,fractal,kmeans_k10,fractal_classes}.tif"
echo "  hrsc_mola_dem/    — hrsc_mola_dem_200m_{js,edges,fractal,kmeans_k10,fractal_classes}.tif"
echo "============================================"
