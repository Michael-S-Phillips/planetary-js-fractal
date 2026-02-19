#!/usr/bin/env python3
"""
Plot Martian geological age vs JS divergence edge density.

Rasterizes the Tanaka et al. (2014) global geologic map (SIM3292) onto the
THEMIS JS divergence grid at reduced resolution, then computes edge density
and mean JS divergence per geological epoch.

Author: Michael S. Phillips
Date: 2026-02-17
"""

import logging
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.warp import transform_geom
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logger = logging.getLogger("plot_age_vs_edges")

# ---------------------------------------------------------------------------
# Epoch classification from unit codes
# ---------------------------------------------------------------------------

# Martian epoch approximate age ranges (Ga) from Tanaka et al. (2014)
# Using Michael (2013) chronology system
EPOCH_INFO = {
    "Late Amazonian":       {"order": 1, "age_mid": 0.3,  "age_range": (0.0, 0.6),   "color": "#FFF2B2"},
    "Middle Amazonian":     {"order": 2, "age_mid": 1.2,  "age_range": (0.6, 1.8),   "color": "#FFE066"},
    "Early Amazonian":      {"order": 3, "age_mid": 2.4,  "age_range": (1.8, 3.0),   "color": "#FFD700"},
    "Amazonian":            {"order": 4, "age_mid": 1.5,  "age_range": (0.0, 3.0),   "color": "#FFE880"},
    "Amazonian-Hesperian":  {"order": 5, "age_mid": 3.2,  "age_range": (3.0, 3.4),   "color": "#D4E88B"},
    "Late Hesperian":       {"order": 6, "age_mid": 3.35, "age_range": (3.2, 3.5),   "color": "#93C47D"},
    "Early Hesperian":      {"order": 7, "age_mid": 3.55, "age_range": (3.5, 3.6),   "color": "#6AA84F"},
    "Hesperian":            {"order": 8, "age_mid": 3.45, "age_range": (3.2, 3.6),   "color": "#7FB870"},
    "Hesperian-Noachian":   {"order": 9, "age_mid": 3.65, "age_range": (3.6, 3.7),   "color": "#A2C4A2"},
    "Late Noachian":        {"order": 10, "age_mid": 3.75, "age_range": (3.7, 3.8),  "color": "#CC7755"},
    "Middle Noachian":      {"order": 11, "age_mid": 3.85, "age_range": (3.8, 3.9),  "color": "#B35533"},
    "Early Noachian":       {"order": 12, "age_mid": 3.95, "age_range": (3.9, 4.1),  "color": "#993311"},
    "Noachian":             {"order": 13, "age_mid": 3.85, "age_range": (3.7, 4.1),  "color": "#AA6644"},
    "Amazonian-Noachian":   {"order": 14, "age_mid": 2.0,  "age_range": (0.0, 4.1),  "color": "#CCCCAA"},
}


def unit_to_epoch(unit_code):
    """Map a SIM3292 geologic unit code to its epoch name."""
    u = unit_code.strip()

    # Spanning units first (check 2-char prefixes)
    if u.startswith("AH"):
        return "Amazonian-Hesperian"
    if u.startswith("HN"):
        return "Hesperian-Noachian"
    if u.startswith("AN"):
        return "Amazonian-Noachian"

    # Specific sub-epochs (check 2-char prefixes)
    if u.startswith("lA"):
        return "Late Amazonian"
    if u.startswith("mA"):
        return "Middle Amazonian"
    if u.startswith("eA"):
        return "Early Amazonian"
    if u.startswith("lH"):
        return "Late Hesperian"
    if u.startswith("eH"):
        return "Early Hesperian"
    if u.startswith("lN"):
        return "Late Noachian"
    if u.startswith("mN"):
        return "Middle Noachian"
    if u.startswith("eN"):
        return "Early Noachian"

    # Undivided epochs (1-char prefix)
    if u.startswith("N"):
        return "Noachian"
    if u.startswith("H"):
        return "Hesperian"
    if u.startswith("A"):
        return "Amazonian"

    return "Unknown"


# ---------------------------------------------------------------------------
# Reproject shapefile geometries to raster CRS
# ---------------------------------------------------------------------------

def reproject_geologic_map(shp_path, target_crs):
    """Load and reproject geologic map, adding epoch classification."""
    gdf = gpd.read_file(shp_path)
    logger.info(f"Loaded {len(gdf)} geologic polygons, CRS: {gdf.crs}")

    # Add epoch column
    gdf["epoch"] = gdf["Unit"].apply(unit_to_epoch)
    logger.info(f"Epoch distribution:\n{gdf.groupby('epoch')['SphArea_km'].sum().sort_values(ascending=False).to_string()}")

    # Reproject to target CRS
    gdf_proj = gdf.to_crs(target_crs)
    logger.info(f"Reprojected to {target_crs}")

    return gdf_proj


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Paths
    js_path = "/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/themis_day_100m_js.tif"
    edge_path = "/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/themis_day_100m_edges.tif"
    shp_path = "/Volumes/Rohan/Mars_GIS_Data/SIM3292_MarsGlobalGeologicGIS_20M/SIM3292_Shapefiles/SIM3292_Global_Geology.shp"
    output_dir = Path("/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges")

    # Downsampling factor: work at ~5km resolution (50x coarser than 100m)
    downsample = 50

    with rasterio.open(js_path) as js_src:
        full_h = js_src.height
        full_w = js_src.width
        full_transform = js_src.transform
        crs = js_src.crs

        # Compute reduced dimensions
        red_h = full_h // downsample
        red_w = full_w // downsample
        # Adjusted transform for reduced resolution
        red_transform = Affine(
            full_transform.a * downsample, full_transform.b, full_transform.c,
            full_transform.d, full_transform.e * downsample, full_transform.f
        )

        logger.info(f"Full raster: {full_w}x{full_h}")
        logger.info(f"Reduced raster: {red_w}x{red_h} (downsample={downsample}x, ~{abs(red_transform.a)/1000:.1f}km)")

        # ----- Step 1: Reproject and rasterize geologic map -----
        logger.info("Reprojecting geologic map...")
        gdf = reproject_geologic_map(shp_path, crs)

        # Assign integer epoch codes for rasterization
        epochs_present = sorted(
            [e for e in gdf["epoch"].unique() if e != "Unknown"],
            key=lambda e: EPOCH_INFO.get(e, {}).get("order", 99)
        )
        epoch_to_id = {e: i + 1 for i, e in enumerate(epochs_present)}
        id_to_epoch = {v: k for k, v in epoch_to_id.items()}

        # Build (geometry, value) pairs
        shapes = []
        for _, row in gdf.iterrows():
            eid = epoch_to_id.get(row["epoch"], 0)
            if eid > 0:
                shapes.append((row.geometry, eid))

        logger.info(f"Rasterizing {len(shapes)} polygons into {red_w}x{red_h} grid...")
        epoch_raster = rasterize(
            shapes,
            out_shape=(red_h, red_w),
            transform=red_transform,
            fill=0,
            dtype=np.uint8,
        )
        logger.info(f"Rasterized: {np.count_nonzero(epoch_raster)} classified pixels "
                    f"of {epoch_raster.size} total")

        # ----- Step 2: Read JS raster at reduced resolution -----
        logger.info("Reading JS raster at reduced resolution...")
        js_data = js_src.read(
            1,
            out_shape=(red_h, red_w),
            resampling=rasterio.enums.Resampling.average
        )
        logger.info(f"JS data shape: {js_data.shape}, "
                    f"valid: {np.count_nonzero(~np.isnan(js_data))}")

    # ----- Step 3: Compute edge density at reduced resolution -----
    # Read edge raster in wide strips and vectorize the block aggregation
    logger.info("Computing edge density at reduced resolution (strip-based)...")
    edge_density = np.full((red_h, red_w), np.nan, dtype=np.float32)
    usable_w = red_w * downsample  # columns we can cleanly reshape

    with rasterio.open(edge_path) as edge_src:
        for row in range(red_h):
            full_row_start = row * downsample
            full_row_end = min(full_row_start + downsample, full_h)
            actual_h = full_row_end - full_row_start

            window = Window(0, full_row_start, usable_w, actual_h)
            strip = edge_src.read(1, window=window).astype(np.float32)

            # Mark nodata as NaN for nanmean
            strip[strip == 255] = np.nan

            if actual_h == downsample:
                # Reshape to (downsample, red_w, downsample) and nanmean
                block = strip[:, :usable_w].reshape(downsample, red_w, downsample)
                with np.errstate(all="ignore"):
                    edge_density[row, :] = np.nanmean(block, axis=(0, 2))
            else:
                # Edge row — reshape column dimension only
                block = strip[:, :usable_w].reshape(actual_h, red_w, downsample)
                with np.errstate(all="ignore"):
                    edge_density[row, :] = np.nanmean(block, axis=(0, 2))

            if row % 500 == 0:
                logger.info(f"  Edge density row {row}/{red_h}")

    logger.info(f"Edge density computed: valid={np.count_nonzero(~np.isnan(edge_density))}")

    # ----- Step 4: Compute statistics per epoch -----
    logger.info("Computing statistics per epoch...")

    results = []
    for epoch_name in epochs_present:
        eid = epoch_to_id[epoch_name]
        info = EPOCH_INFO.get(epoch_name, {})
        mask = (epoch_raster == eid)

        # JS divergence stats
        js_vals = js_data[mask & ~np.isnan(js_data)]
        # Edge density stats
        ed_vals = edge_density[mask & ~np.isnan(edge_density)]

        if js_vals.size < 10:
            continue

        results.append({
            "epoch": epoch_name,
            "order": info.get("order", 99),
            "age_mid": info.get("age_mid", 0),
            "age_range": info.get("age_range", (0, 0)),
            "color": info.get("color", "#888888"),
            "n_pixels": js_vals.size,
            "js_mean": np.mean(js_vals),
            "js_median": np.median(js_vals),
            "js_std": np.std(js_vals),
            "js_p25": np.percentile(js_vals, 25),
            "js_p75": np.percentile(js_vals, 75),
            "ed_mean": np.mean(ed_vals) if ed_vals.size > 0 else np.nan,
            "ed_median": np.median(ed_vals) if ed_vals.size > 0 else np.nan,
            "ed_std": np.std(ed_vals) if ed_vals.size > 0 else np.nan,
            "ed_p25": np.percentile(ed_vals, 25) if ed_vals.size > 0 else np.nan,
            "ed_p75": np.percentile(ed_vals, 75) if ed_vals.size > 0 else np.nan,
        })

    # Sort by age (oldest first on left)
    results.sort(key=lambda r: -r["age_mid"])

    # Print table
    logger.info("\n" + "=" * 90)
    logger.info(f"{'Epoch':<25s} {'Age(Ga)':>8s} {'N pixels':>10s} "
                f"{'JS mean':>8s} {'JS med':>8s} {'Edge%':>8s}")
    logger.info("-" * 90)
    for r in results:
        logger.info(f"{r['epoch']:<25s} {r['age_mid']:>8.2f} {r['n_pixels']:>10,d} "
                    f"{r['js_mean']:>8.4f} {r['js_median']:>8.4f} "
                    f"{r['ed_mean']*100:>7.2f}%")

    # ----- Step 5: Create plots -----
    logger.info("Creating plots...")

    # Filter out spanning/undivided epochs for cleaner plot
    # Keep only specific sub-epochs
    specific_epochs = [r for r in results if r["epoch"] not in
                       ("Amazonian", "Hesperian", "Noachian",
                        "Amazonian-Hesperian", "Hesperian-Noachian", "Amazonian-Noachian")]
    specific_epochs.sort(key=lambda r: -r["age_mid"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Mars Geological Age vs Edge Characteristics\n"
                 "(Tanaka et al. 2014 units / THEMIS Day IR 100m JS divergence)",
                 fontsize=13, fontweight="bold")

    x_ages = [r["age_mid"] for r in specific_epochs]
    x_labels = [r["epoch"] for r in specific_epochs]
    colors = [r["color"] for r in specific_epochs]

    # -- Panel 1: Mean JS divergence --
    js_means = [r["js_mean"] for r in specific_epochs]
    js_p25 = [r["js_p25"] for r in specific_epochs]
    js_p75 = [r["js_p75"] for r in specific_epochs]
    js_err_low = [m - p25 for m, p25 in zip(js_means, js_p25)]
    js_err_high = [p75 - m for m, p75 in zip(js_means, js_p75)]

    bars1 = ax1.bar(range(len(specific_epochs)), js_means, color=colors,
                    edgecolor="black", linewidth=0.5, zorder=3)
    ax1.errorbar(range(len(specific_epochs)), js_means,
                 yerr=[js_err_low, js_err_high],
                 fmt="none", ecolor="black", capsize=3, capthick=1, zorder=4)
    ax1.set_ylabel("JS Divergence\n(mean with IQR)", fontsize=11)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.set_axisbelow(True)

    # Add pixel count annotations
    for i, r in enumerate(specific_epochs):
        ax1.annotate(f"n={r['n_pixels']:,}", (i, js_means[i]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=6, color="gray")

    # -- Panel 2: Edge density --
    ed_means = [r["ed_mean"] * 100 for r in specific_epochs]
    ed_p25 = [r["ed_p25"] * 100 for r in specific_epochs]
    ed_p75 = [r["ed_p75"] * 100 for r in specific_epochs]
    ed_err_low = [m - p25 for m, p25 in zip(ed_means, ed_p25)]
    ed_err_high = [p75 - m for m, p75 in zip(ed_means, ed_p75)]

    bars2 = ax2.bar(range(len(specific_epochs)), ed_means, color=colors,
                    edgecolor="black", linewidth=0.5, zorder=3)
    ax2.errorbar(range(len(specific_epochs)), ed_means,
                 yerr=[ed_err_low, ed_err_high],
                 fmt="none", ecolor="black", capsize=3, capthick=1, zorder=4)
    ax2.set_ylabel("Edge Density (%)\n(mean with IQR)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3, zorder=0)
    ax2.set_axisbelow(True)

    # X-axis: epoch labels with ages
    ax2.set_xticks(range(len(specific_epochs)))
    ax2.set_xticklabels([f"{r['epoch']}\n({r['age_mid']:.1f} Ga)"
                         for r in specific_epochs],
                        rotation=45, ha="right", fontsize=9)
    ax2.set_xlabel("Geological Epoch (older → younger →)", fontsize=11)

    # Era backgrounds
    for ax in [ax1, ax2]:
        ymin, ymax = ax.get_ylim()
        # Find boundaries between eras
        for i, r in enumerate(specific_epochs):
            if "Noachian" in r["epoch"] and not any("Noachian" in specific_epochs[j]["epoch"]
                                                     for j in range(max(0, i-1), i)):
                pass  # first Noachian

    plt.tight_layout()

    out_path = output_dir / "age_vs_edge_density.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved plot: {out_path}")
    plt.close()

    # ----- Scatter plot: Age vs metrics -----
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
    fig2.suptitle("Mars Geological Age vs Edge Characteristics (Scatter)",
                  fontsize=13, fontweight="bold")

    for r in specific_epochs:
        ax3.scatter(r["age_mid"], r["js_mean"], s=max(20, min(200, r["n_pixels"] / 500)),
                   c=r["color"], edgecolors="black", linewidth=0.8, zorder=3)
        ax3.annotate(r["epoch"].replace("Early ", "e").replace("Middle ", "m").replace("Late ", "l"),
                    (r["age_mid"], r["js_mean"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax3.set_xlabel("Approximate Age (Ga)", fontsize=11)
    ax3.set_ylabel("Mean JS Divergence", fontsize=11)
    ax3.set_title("JS Divergence vs Age")
    ax3.invert_xaxis()
    ax3.grid(alpha=0.3)

    for r in specific_epochs:
        ax4.scatter(r["age_mid"], r["ed_mean"] * 100,
                   s=max(20, min(200, r["n_pixels"] / 500)),
                   c=r["color"], edgecolors="black", linewidth=0.8, zorder=3)
        ax4.annotate(r["epoch"].replace("Early ", "e").replace("Middle ", "m").replace("Late ", "l"),
                    (r["age_mid"], r["ed_mean"] * 100),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax4.set_xlabel("Approximate Age (Ga)", fontsize=11)
    ax4.set_ylabel("Edge Density (%)", fontsize=11)
    ax4.set_title("Edge Density vs Age")
    ax4.invert_xaxis()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    out_path2 = output_dir / "age_vs_edge_scatter.png"
    fig2.savefig(out_path2, dpi=200, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved scatter plot: {out_path2}")
    plt.close()


if __name__ == "__main__":
    main()
