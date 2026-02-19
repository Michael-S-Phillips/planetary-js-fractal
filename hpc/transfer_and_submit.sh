#!/bin/bash
# ============================================================================
# Transfer scripts and data to UArizona HPC
#
# Syncs Python scripts, SLURM job files, and (optionally) input rasters.
# Does NOT submit jobs — use submit_all.sh on the HPC for that.
#
# Usage:
#   bash transfer_and_submit.sh             # scripts + slurm only (fast)
#   bash transfer_and_submit.sh --data      # also transfer input rasters
# ============================================================================

set -euo pipefail

# ---- Configuration ----
NETID="phillipsm"
REMOTE="filexfer.hpc.arizona.edu"
WORK_DIR="/groups/sbyrne/${NETID}/mars_js"

LOCAL_BASE="/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges"

# Input rasters (local paths)
LOCAL_THEMIS_DAYIR="/Volumes/Rohan/Mars_GIS_Data/THEMIS/global/Mars_MO_THEMIS-IR-Day_mosaic_global_100m_v12.tif"
LOCAL_THEMIS_TI="/Volumes/Rohan/Mars_GIS_Data/THEMIS/thermal_inertia/THEMIS_TI_Mosaic_Quant_60S300E_100mpp_ESRI104971.tif"
LOCAL_HRSC_DEM="/Volumes/Rohan/Mars_GIS_Data/MOLA_HRSC/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif"

TRANSFER_DATA=false
if [ "${1:-}" = "--data" ]; then
    TRANSFER_DATA=true
fi

echo "============================================"
echo "Mars JS Edge Detection — HPC Transfer"
echo "============================================"
echo ""

# ---- Step 1: Create remote directory structure ----
echo ">>> Creating remote directory structure..."
ssh "${NETID}@${REMOTE}" \
    "mkdir -p /groups/sbyrne/phillipsm/mars_js/{scripts,input,slurm} && \
     mkdir -p /xdisk/sbyrne/phillipsm/mars_js/{themis_dayir,themis_ti,hrsc_mola_dem} && \
     echo 'Directories created.'"
echo ""

# ---- Step 2: Transfer Python scripts ----
echo ">>> Transferring Python scripts..."
rsync -avP \
    "${LOCAL_BASE}/scripts/js_edge_detect.py" \
    "${LOCAL_BASE}/scripts/js_fractal_classify.py" \
    "${NETID}@${REMOTE}:${WORK_DIR}/scripts/"
echo ""

# ---- Step 3: Transfer SLURM job files ----
echo ">>> Transferring SLURM scripts..."
rsync -avP \
    "${LOCAL_BASE}/hpc/submit_phase1.slurm" \
    "${LOCAL_BASE}/hpc/submit_phase2_edges.slurm" \
    "${LOCAL_BASE}/hpc/submit_fractal.slurm" \
    "${LOCAL_BASE}/hpc/submit_all.sh" \
    "${LOCAL_BASE}/hpc/setup_env.sh" \
    "${NETID}@${REMOTE}:${WORK_DIR}/slurm/"
echo ""

# ---- Step 4: Transfer input rasters (optional) ----
if [ "${TRANSFER_DATA}" = true ]; then
    echo ">>> Transferring input rasters..."

    for LOCAL_FILE in "${LOCAL_THEMIS_DAYIR}" "${LOCAL_THEMIS_TI}" "${LOCAL_HRSC_DEM}"; do
        if [ -f "${LOCAL_FILE}" ]; then
            BASENAME=$(basename "${LOCAL_FILE}")
            echo "  Transferring ${BASENAME}..."
            rsync -avP "${LOCAL_FILE}" "${NETID}@${REMOTE}:${WORK_DIR}/input/"
        else
            echo "  SKIPPING (not found): ${LOCAL_FILE}"
        fi
    done
    echo ""
else
    echo ">>> Skipping data transfer (use --data to include rasters)"
    echo ""
fi

echo "============================================"
echo "Transfer complete!"
echo ""
echo "Next steps (on HPC):"
echo "  1. ssh ${NETID}@hpc.arizona.edu"
echo "  2. shell"
echo "  3. cd ${WORK_DIR}/slurm"
echo "  4. bash setup_env.sh       # (if first time — run in interactive session)"
echo "  5. bash submit_all.sh      # submit all datasets"
echo "     bash submit_all.sh ti   # or just one dataset"
echo ""
echo "Monitor: squeue --user=${NETID}"
echo "============================================"
