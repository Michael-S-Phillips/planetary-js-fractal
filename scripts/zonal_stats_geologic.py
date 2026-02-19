#!/usr/bin/env python3
"""
Per-polygon zonal statistics of JS divergence and edge density on the
Tanaka et al. (2014) global geologic map of Mars (SIM3292).

Reads the JS and edge rasters at reduced resolution, rasterizes polygon IDs,
computes per-polygon stats, saves to GeoPackage, and generates scatter plots.

Author: Michael S. Phillips
Date: 2026-02-17
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import Affine
from rasterio.windows import Window
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

logger = logging.getLogger("zonal_stats")

# ---------------------------------------------------------------------------
# Epoch classification
# ---------------------------------------------------------------------------

EPOCH_INFO = {
    "Late Amazonian":       {"order": 1, "age_mid": 0.3,  "color": "#FFF2B2"},
    "Middle Amazonian":     {"order": 2, "age_mid": 1.2,  "color": "#FFE066"},
    "Early Amazonian":      {"order": 3, "age_mid": 2.4,  "color": "#FFD700"},
    "Amazonian":            {"order": 4, "age_mid": 1.5,  "color": "#FFE880"},
    "Amazonian-Hesperian":  {"order": 5, "age_mid": 3.2,  "color": "#D4E88B"},
    "Late Hesperian":       {"order": 6, "age_mid": 3.35, "color": "#93C47D"},
    "Early Hesperian":      {"order": 7, "age_mid": 3.55, "color": "#6AA84F"},
    "Hesperian":            {"order": 8, "age_mid": 3.45, "color": "#7FB870"},
    "Hesperian-Noachian":   {"order": 9, "age_mid": 3.65, "color": "#A2C4A2"},
    "Late Noachian":        {"order": 10, "age_mid": 3.75, "color": "#CC7755"},
    "Middle Noachian":      {"order": 11, "age_mid": 3.85, "color": "#B35533"},
    "Early Noachian":       {"order": 12, "age_mid": 3.95, "color": "#993311"},
    "Noachian":             {"order": 13, "age_mid": 3.85, "color": "#AA6644"},
    "Amazonian-Noachian":   {"order": 14, "age_mid": 2.0,  "color": "#CCCCAA"},
}


def unit_to_epoch(unit_code):
    u = unit_code.strip()
    if u.startswith("AH"): return "Amazonian-Hesperian"
    if u.startswith("HN"): return "Hesperian-Noachian"
    if u.startswith("AN"): return "Amazonian-Noachian"
    if u.startswith("lA"): return "Late Amazonian"
    if u.startswith("mA"): return "Middle Amazonian"
    if u.startswith("eA"): return "Early Amazonian"
    if u.startswith("lH"): return "Late Hesperian"
    if u.startswith("eH"): return "Early Hesperian"
    if u.startswith("lN"): return "Late Noachian"
    if u.startswith("mN"): return "Middle Noachian"
    if u.startswith("eN"): return "Early Noachian"
    if u.startswith("N"): return "Noachian"
    if u.startswith("H"): return "Hesperian"
    if u.startswith("A"): return "Amazonian"
    return "Unknown"


def unit_to_type(unit_code):
    """Classify a SIM3292 unit code into a geological unit type."""
    u = unit_code.strip()
    # Strip epoch prefix to get the suffix
    for pfx in ("AH", "HN", "AN", "lA", "mA", "eA", "lH", "eH", "lN", "mN", "eN",
                "N", "H", "A"):
        if u.startswith(pfx):
            suffix = u[len(pfx):]
            break
    else:
        return "Unknown"

    if suffix in ("h", "hm", "hu", "he"):
        return "Highland"
    if suffix == "i":
        return "Impact"
    if suffix in ("t", "to", "tu"):
        return "Transition"
    if suffix in ("v", "ve", "vf"):
        return "Volcanic"
    if suffix == "l":
        return "Lowland"
    if suffix == "a":
        return "Apron"
    if suffix in ("p", "pc", "pd", "pe", "pu"):
        return "Polar"
    if suffix == "b":
        return "Basin"
    return "Unknown"


# ---------------------------------------------------------------------------
# Main
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
    gpkg_path = output_dir / "tanaka2014_js_stats.gpkg"

    downsample = 50  # ~5 km resolution

    # ---- Step 1: Load and reproject geologic map ----
    logger.info("Loading geologic map...")
    gdf = gpd.read_file(shp_path)
    gdf["epoch"] = gdf["Unit"].apply(unit_to_epoch)
    gdf["age_mid"] = gdf["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("age_mid", np.nan))

    with rasterio.open(js_path) as js_src:
        full_h = js_src.height
        full_w = js_src.width
        full_transform = js_src.transform
        crs = js_src.crs

        red_h = full_h // downsample
        red_w = full_w // downsample
        red_transform = Affine(
            full_transform.a * downsample, full_transform.b, full_transform.c,
            full_transform.d, full_transform.e * downsample, full_transform.f,
        )

        logger.info(f"Reduced grid: {red_w}x{red_h} (~{abs(red_transform.a)/1000:.1f} km)")

        # Reproject shapefile to raster CRS
        logger.info("Reprojecting to raster CRS...")
        gdf_proj = gdf.to_crs(crs)

        # ---- Step 2: Rasterize per-polygon IDs ----
        # Use 1-based OBJECTID as raster value so 0 = no polygon
        logger.info("Rasterizing polygon IDs...")
        # Build a stable integer ID for each polygon
        gdf_proj = gdf_proj.reset_index(drop=True)
        gdf_proj["_poly_id"] = np.arange(1, len(gdf_proj) + 1, dtype=np.int32)

        shapes = [(row.geometry, row._poly_id) for _, row in gdf_proj.iterrows()]
        poly_raster = rasterize(
            shapes,
            out_shape=(red_h, red_w),
            transform=red_transform,
            fill=0,
            dtype=np.int32,
        )
        n_classified = np.count_nonzero(poly_raster)
        logger.info(f"Rasterized: {n_classified:,} pixels classified of {poly_raster.size:,}")

        # ---- Step 3: Read JS raster at reduced resolution ----
        logger.info("Reading JS raster at reduced resolution (this may take a while)...")
        t0 = time.time()
        js_data = js_src.read(
            1,
            out_shape=(red_h, red_w),
            resampling=rasterio.enums.Resampling.average,
        )
        logger.info(f"JS read complete in {(time.time()-t0)/60:.1f}m, "
                    f"valid={np.count_nonzero(~np.isnan(js_data)):,}")

    # ---- Step 4: Edge density at reduced resolution ----
    logger.info("Computing edge density at reduced resolution...")
    edge_density = np.full((red_h, red_w), np.nan, dtype=np.float32)
    usable_w = red_w * downsample
    t0 = time.time()

    with rasterio.open(edge_path) as edge_src:
        for row in range(red_h):
            full_row_start = row * downsample
            full_row_end = min(full_row_start + downsample, full_h)
            actual_h = full_row_end - full_row_start

            window = Window(0, full_row_start, usable_w, actual_h)
            strip = edge_src.read(1, window=window).astype(np.float32)
            strip[strip == 255] = np.nan

            block = strip[:, :usable_w].reshape(actual_h, red_w, downsample)
            with np.errstate(all="ignore"):
                edge_density[row, :] = np.nanmean(block, axis=(0, 2))

            if row % 500 == 0:
                logger.info(f"  Edge density row {row}/{red_h}")

    logger.info(f"Edge density complete in {(time.time()-t0)/60:.1f}m")

    # ---- Step 5: Compute per-polygon statistics ----
    logger.info("Computing per-polygon statistics...")
    unique_ids = np.unique(poly_raster)
    unique_ids = unique_ids[unique_ids > 0]  # skip 0 (no polygon)

    # Preallocate result arrays
    n_poly = len(gdf_proj)
    js_mean_arr = np.full(n_poly, np.nan, dtype=np.float64)
    js_std_arr = np.full(n_poly, np.nan, dtype=np.float64)
    js_median_arr = np.full(n_poly, np.nan, dtype=np.float64)
    js_p25_arr = np.full(n_poly, np.nan, dtype=np.float64)
    js_p75_arr = np.full(n_poly, np.nan, dtype=np.float64)
    ed_mean_arr = np.full(n_poly, np.nan, dtype=np.float64)
    ed_std_arr = np.full(n_poly, np.nan, dtype=np.float64)
    n_pixels_arr = np.zeros(n_poly, dtype=np.int64)

    for pid in unique_ids:
        idx = pid - 1  # 0-based index into gdf
        mask = poly_raster == pid

        # JS stats
        js_vals = js_data[mask]
        js_valid = js_vals[~np.isnan(js_vals)]
        n_pixels_arr[idx] = js_valid.size

        if js_valid.size >= 5:
            js_mean_arr[idx] = np.mean(js_valid)
            js_std_arr[idx] = np.std(js_valid)
            js_median_arr[idx] = np.median(js_valid)
            js_p25_arr[idx] = np.percentile(js_valid, 25)
            js_p75_arr[idx] = np.percentile(js_valid, 75)

        # Edge density stats
        ed_vals = edge_density[mask]
        ed_valid = ed_vals[~np.isnan(ed_vals)]
        if ed_valid.size >= 5:
            ed_mean_arr[idx] = np.mean(ed_valid)
            ed_std_arr[idx] = np.std(ed_valid)

    logger.info(f"Stats computed for {np.count_nonzero(~np.isnan(js_mean_arr))} polygons "
                f"(of {n_poly} total)")

    # ---- Step 6: Attach stats to GeoDataFrame and save ----
    # Work with the original unprojected GeoDataFrame for QGIS compatibility
    gdf["js_mean"] = js_mean_arr
    gdf["js_std"] = js_std_arr
    gdf["js_median"] = js_median_arr
    gdf["js_p25"] = js_p25_arr
    gdf["js_p75"] = js_p75_arr
    gdf["edge_dens"] = ed_mean_arr
    gdf["edge_std"] = ed_std_arr
    gdf["n_pixels"] = n_pixels_arr
    gdf["unit_type"] = gdf["Unit"].apply(unit_to_type)

    logger.info(f"Saving GeoPackage: {gpkg_path}")
    gdf.to_file(gpkg_path, driver="GPKG")
    logger.info(f"Saved {len(gdf)} polygons with zonal stats")

    # ---- Step 7: Aggregate per epoch and plot ----
    logger.info("Aggregating per epoch...")

    # Weighted mean per epoch (weight by n_pixels per polygon)
    specific_epoch_names = [
        "Early Noachian", "Middle Noachian", "Late Noachian",
        "Early Hesperian", "Late Hesperian",
        "Early Amazonian", "Middle Amazonian", "Late Amazonian",
    ]

    epoch_results = []
    for epoch_name in specific_epoch_names:
        info = EPOCH_INFO[epoch_name]
        emask = (gdf["epoch"] == epoch_name) & (gdf["n_pixels"] >= 5)
        subset = gdf[emask]
        if len(subset) == 0:
            continue

        weights = subset["n_pixels"].values.astype(np.float64)
        total_w = weights.sum()
        if total_w == 0:
            continue

        js_wm = np.average(subset["js_mean"].values, weights=weights)
        ed_wm = np.average(subset["edge_dens"].values, weights=weights)

        # Weighted std (between-polygon variation weighted by area)
        js_ws = np.sqrt(np.average((subset["js_mean"].values - js_wm)**2, weights=weights))
        ed_ws = np.sqrt(np.average((subset["edge_dens"].values - ed_wm)**2, weights=weights))

        epoch_results.append({
            "epoch": epoch_name,
            "age_mid": info["age_mid"],
            "color": info["color"],
            "js_mean": js_wm,
            "js_std": js_ws,
            "edge_mean": ed_wm * 100,
            "edge_std": ed_ws * 100,
            "n_polygons": len(subset),
            "n_pixels": int(total_w),
        })

    epoch_results.sort(key=lambda r: -r["age_mid"])

    logger.info("\n" + "=" * 85)
    logger.info(f"{'Epoch':<22s} {'Age':>5s} {'#poly':>6s} {'#px':>10s} "
                f"{'JS mean':>8s} {'JS std':>8s} {'Edge%':>7s} {'E std':>7s}")
    logger.info("-" * 85)
    for r in epoch_results:
        logger.info(f"{r['epoch']:<22s} {r['age_mid']:>5.2f} {r['n_polygons']:>6d} "
                    f"{r['n_pixels']:>10,d} {r['js_mean']:>8.4f} {r['js_std']:>8.4f} "
                    f"{r['edge_mean']:>6.2f}% {r['edge_std']:>6.2f}%")

    # ---- Step 8: Scatter plots with exponential fits ----
    logger.info("Creating scatter plots...")

    ages = np.array([r["age_mid"] for r in epoch_results])
    js_m = np.array([r["js_mean"] for r in epoch_results])
    ed_m = np.array([r["edge_mean"] for r in epoch_results])
    npx = np.array([r["n_pixels"] for r in epoch_results])
    cols = [r["color"] for r in epoch_results]
    enames = [r["epoch"] for r in epoch_results]
    short = [n.replace("Early ", "e").replace("Middle ", "m").replace("Late ", "l")
             for n in enames]
    sizes = np.clip(np.sqrt(npx) / 8, 30, 300)

    def exp_model(x, a, b, c):
        return a * np.exp(b * x) + c

    def do_fit(x, y, w, p0):
        popt, _ = curve_fit(exp_model, x, y, p0=p0, sigma=1.0/w, maxfev=10000)
        y_pred = exp_model(x, *popt)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        return popt, r2

    # Exclude mAmazonian index
    mAmaz_idx = next((i for i, e in enumerate(enames) if e == "Middle Amazonian"), None)
    mask_ex = np.ones(len(epoch_results), dtype=bool)
    if mAmaz_idx is not None:
        mask_ex[mAmaz_idx] = False

    w_all = np.sqrt(npx)

    # Fits: all and excluding mAmazonian
    popt_js_all, r2_js_all = do_fit(ages, js_m, w_all, [0.05, 0.3, 0.3])
    popt_ed_all, r2_ed_all = do_fit(ages, ed_m, w_all, [5, 0.3, 20])
    popt_js_ex, r2_js_ex = do_fit(ages[mask_ex], js_m[mask_ex], w_all[mask_ex], [0.05, 0.3, 0.3])
    popt_ed_ex, r2_ed_ex = do_fit(ages[mask_ex], ed_m[mask_ex], w_all[mask_ex], [5, 0.3, 20])

    age_smooth = np.linspace(0, 4.2, 200)

    # --- 2x2 comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Mars Geological Age vs Edge Characteristics (Per-Polygon Zonal Stats)\n"
                 "THEMIS Day IR 100m / Tanaka et al. (2014)",
                 fontsize=14, fontweight="bold", y=0.98)

    configs = [
        ("JS Divergence — All Epochs", popt_js_all, r2_js_all, js_m, "Mean JS Divergence", False),
        ("JS Divergence — Excl. mAmazonian", popt_js_ex, r2_js_ex, js_m, "Mean JS Divergence", True),
        ("Edge Density — All Epochs", popt_ed_all, r2_ed_all, ed_m, "Edge Density (%)", False),
        ("Edge Density — Excl. mAmazonian", popt_ed_ex, r2_ed_ex, ed_m, "Edge Density (%)", True),
    ]

    for ax, (title, popt, r2, yd, ylabel, is_excl) in zip(axes.flat, configs):
        a, b, c = popt
        fit_curve = exp_model(age_smooth, *popt)
        ax.plot(age_smooth, fit_curve, "-", color="#444444", linewidth=2.5, zorder=2,
                label=f"$y = {a:.4f}\,e^{{{b:.3f}x}} + {c:.3f}$\n$R^2 = {r2:.4f}$")

        for i in range(len(epoch_results)):
            is_mA = (i == mAmaz_idx)
            if is_excl and is_mA:
                ax.scatter(ages[i], yd[i], s=sizes[i], facecolors="none",
                          edgecolors="red", linewidth=1.5, zorder=5)
                ax.scatter(ages[i], yd[i], s=30, c="red", marker="x",
                          linewidth=1.5, zorder=6)
            else:
                ax.scatter(ages[i], yd[i], s=sizes[i], c=cols[i],
                          edgecolors="black", linewidth=0.8, zorder=3)

        for i, s in enumerate(short):
            if s == "mNoachian": ox, oy = 8, -14
            elif s == "lHesperian": ox, oy = 8, -14
            elif s == "lAmazonian": ox, oy = 8, -12
            elif s == "mAmazonian" and is_excl: ox, oy = 10, 10
            else: ox, oy = 8, 7
            color = "red" if (is_excl and i == mAmaz_idx) else "#333333"
            ax.annotate(s, (ages[i], yd[i]),
                       textcoords="offset points", xytext=(ox, oy),
                       fontsize=7.5, color=color,
                       fontweight="bold" if (is_excl and i == mAmaz_idx) else "normal")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Surface Age (Ga)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.invert_xaxis()
        ax.set_xlim(4.3, -0.1)
        ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

    plt.tight_layout()
    scatter_path = output_dir / "plots" / "age_vs_edge_zonal_stats.png"
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved scatter plot: {scatter_path}")
    plt.close()

    # --- Per-polygon scatter (all 1311 polygons) ---
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig2.suptitle("Per-Polygon JS Divergence & Edge Density vs Age\n"
                  "Each dot = one Tanaka et al. (2014) polygon",
                  fontsize=13, fontweight="bold")

    valid_mask = gdf["n_pixels"] >= 5
    gv = gdf[valid_mask].copy()
    gv_colors = gv["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    gv_sizes = np.clip(np.sqrt(gv["n_pixels"].values) / 3, 5, 150)

    ax1.scatter(gv["age_mid"], gv["js_mean"], s=gv_sizes, c=gv_colors,
               edgecolors="black", linewidth=0.3, alpha=0.7, zorder=3)
    ax1.plot(age_smooth, exp_model(age_smooth, *popt_js_ex), "-", color="black",
            linewidth=2, zorder=4, label=f"Exp. fit (excl. mAmaz)\n$R^2={r2_js_ex:.3f}$")
    ax1.set_xlabel("Surface Age (Ga)", fontsize=11)
    ax1.set_ylabel("Mean JS Divergence", fontsize=11)
    ax1.set_title("JS Divergence", fontsize=12)
    ax1.invert_xaxis()
    ax1.set_xlim(4.3, -0.1)
    ax1.legend(fontsize=9, loc="lower left")
    ax1.grid(alpha=0.25)

    ax2.scatter(gv["age_mid"], gv["edge_dens"] * 100, s=gv_sizes, c=gv_colors,
               edgecolors="black", linewidth=0.3, alpha=0.7, zorder=3)
    ax2.plot(age_smooth, exp_model(age_smooth, *popt_ed_ex), "-", color="black",
            linewidth=2, zorder=4, label=f"Exp. fit (excl. mAmaz)\n$R^2={r2_ed_ex:.3f}$")
    ax2.set_xlabel("Surface Age (Ga)", fontsize=11)
    ax2.set_ylabel("Edge Density (%)", fontsize=11)
    ax2.set_title("Edge Density", fontsize=12)
    ax2.invert_xaxis()
    ax2.set_xlim(4.3, -0.1)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    poly_scatter_path = output_dir / "plots" / "age_vs_edge_per_polygon.png"
    fig2.savefig(poly_scatter_path, dpi=200, bbox_inches="tight", facecolor="white")
    logger.info(f"Saved per-polygon scatter: {poly_scatter_path}")
    plt.close()

    logger.info("Done.")


if __name__ == "__main__":
    main()
