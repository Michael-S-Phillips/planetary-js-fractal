#!/usr/bin/env python3
"""
Comprehensive exploration of zonal statistics from the Tanaka et al. (2014)
geologic map / THEMIS Day IR 100m JS divergence analysis.

Reads the existing GeoPackage (no raster I/O), adds a unit_type column,
and generates a suite of plots examining age, unit type, and texture metrics.

Author: Michael S. Phillips
Date: 2026-02-17
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches

logger = logging.getLogger("explore_zonal")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GPKG_PATH = Path("/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/tanaka2014_js_stats.gpkg")
PLOT_DIR = Path("/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/plots")

EPOCH_INFO = {
    "Late Amazonian":       {"order": 1,  "age_mid": 0.3,  "color": "#FFF2B2"},
    "Middle Amazonian":     {"order": 2,  "age_mid": 1.2,  "color": "#FFE066"},
    "Early Amazonian":      {"order": 3,  "age_mid": 2.4,  "color": "#FFD700"},
    "Amazonian":            {"order": 4,  "age_mid": 1.5,  "color": "#FFE880"},
    "Amazonian-Hesperian":  {"order": 5,  "age_mid": 3.2,  "color": "#D4E88B"},
    "Late Hesperian":       {"order": 6,  "age_mid": 3.35, "color": "#93C47D"},
    "Early Hesperian":      {"order": 7,  "age_mid": 3.55, "color": "#6AA84F"},
    "Hesperian":            {"order": 8,  "age_mid": 3.45, "color": "#7FB870"},
    "Hesperian-Noachian":   {"order": 9,  "age_mid": 3.65, "color": "#A2C4A2"},
    "Late Noachian":        {"order": 10, "age_mid": 3.75, "color": "#CC7755"},
    "Middle Noachian":      {"order": 11, "age_mid": 3.85, "color": "#B35533"},
    "Early Noachian":       {"order": 12, "age_mid": 3.95, "color": "#993311"},
    "Noachian":             {"order": 13, "age_mid": 3.85, "color": "#AA6644"},
    "Amazonian-Noachian":   {"order": 14, "age_mid": 2.0,  "color": "#CCCCAA"},
}

# Specific (well-constrained) sub-epochs vs transitional/undivided
SPECIFIC_EPOCHS = {
    "Early Noachian", "Middle Noachian", "Late Noachian",
    "Early Hesperian", "Late Hesperian",
    "Early Amazonian", "Middle Amazonian", "Late Amazonian",
}

UNIT_TYPE_COLORS = {
    "Highland":   "#B35533",
    "Impact":     "#888888",
    "Transition": "#6AA84F",
    "Volcanic":   "#CC3333",
    "Lowland":    "#4488CC",
    "Apron":      "#CC8844",
    "Polar":      "#AADDEE",
    "Basin":      "#664488",
}

UNIT_TYPE_ORDER = ["Highland", "Impact", "Transition", "Volcanic",
                   "Lowland", "Apron", "Polar", "Basin"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def unit_to_type(unit_code):
    """Classify a SIM3292 unit code into a geological unit type."""
    u = unit_code.strip()
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


def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c


def do_fit(x, y, w, p0):
    popt, _ = curve_fit(exp_model, x, y, p0=p0, sigma=1.0 / w, maxfev=10000)
    y_pred = exp_model(x, *popt)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return popt, r2


def weighted_epoch_stats(gdf, epoch_names):
    """Compute weighted mean/std per epoch. Returns list of dicts."""
    results = []
    for epoch_name in epoch_names:
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
        js_ws = np.sqrt(np.average((subset["js_mean"].values - js_wm) ** 2, weights=weights))
        ed_ws = np.sqrt(np.average((subset["edge_dens"].values - ed_wm) ** 2, weights=weights))
        results.append({
            "epoch": epoch_name,
            "age_mid": info["age_mid"],
            "color": info["color"],
            "order": info["order"],
            "js_mean": js_wm,
            "js_std": js_ws,
            "edge_mean": ed_wm * 100,
            "edge_std": ed_ws * 100,
            "n_polygons": len(subset),
            "n_pixels": int(total_w),
            "is_specific": epoch_name in SPECIFIC_EPOCHS,
        })
    results.sort(key=lambda r: -r["age_mid"])
    return results


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot1_all_epoch_scatter(gdf, plot_dir):
    """All-epoch scatter with exp fits (specific + transitional)."""
    all_epochs = list(EPOCH_INFO.keys())
    results = weighted_epoch_stats(gdf, all_epochs)

    spec = [r for r in results if r["is_specific"]]
    trans = [r for r in results if not r["is_specific"]]

    # Fit on specific epochs only
    ages_s = np.array([r["age_mid"] for r in spec])
    js_s = np.array([r["js_mean"] for r in spec])
    ed_s = np.array([r["edge_mean"] for r in spec])
    npx_s = np.array([r["n_pixels"] for r in spec])
    w_s = np.sqrt(npx_s)

    # Exclude mAmazonian for the "excl" fit
    mask_ex = np.array([r["epoch"] != "Middle Amazonian" for r in spec])
    popt_js_all, r2_js_all = do_fit(ages_s, js_s, w_s, [0.05, 0.3, 0.3])
    popt_ed_all, r2_ed_all = do_fit(ages_s, ed_s, w_s, [5, 0.3, 20])
    popt_js_ex, r2_js_ex = do_fit(ages_s[mask_ex], js_s[mask_ex], w_s[mask_ex], [0.05, 0.3, 0.3])
    popt_ed_ex, r2_ed_ex = do_fit(ages_s[mask_ex], ed_s[mask_ex], w_s[mask_ex], [5, 0.3, 20])

    age_smooth = np.linspace(0, 4.2, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("All 14 Epochs: Age vs JS Divergence & Edge Density\n"
                 "Weighted means per epoch | THEMIS Day IR 100m / Tanaka et al. (2014)",
                 fontsize=13, fontweight="bold")

    for ax, metric, popt_a, r2_a, popt_e, r2_e, ylabel in [
        (ax1, "js_mean", popt_js_all, r2_js_all, popt_js_ex, r2_js_ex, "Mean JS Divergence"),
        (ax2, "edge_mean", popt_ed_all, r2_ed_all, popt_ed_ex, r2_ed_ex, "Edge Density (%)"),
    ]:
        # Fit curves
        ax.plot(age_smooth, exp_model(age_smooth, *popt_a), "-", color="#444444",
                linewidth=2, zorder=2, label=f"All specific ($R^2$={r2_a:.3f})")
        ax.plot(age_smooth, exp_model(age_smooth, *popt_e), "--", color="#CC4444",
                linewidth=2, zorder=2, label=f"Excl. mAmaz ($R^2$={r2_e:.3f})")

        # Specific epochs (circles)
        for r in spec:
            size = np.clip(np.sqrt(r["n_pixels"]) / 8, 30, 300)
            ax.scatter(r["age_mid"], r[metric], s=size, c=r["color"],
                       edgecolors="black", linewidth=0.8, zorder=3, marker="o")
            ax.errorbar(r["age_mid"], r[metric],
                        yerr=r[metric.replace("mean", "std")],
                        fmt="none", ecolor="#666666", capsize=3, zorder=2)

        # Transitional epochs (diamonds)
        for r in trans:
            size = np.clip(np.sqrt(r["n_pixels"]) / 8, 30, 300)
            ax.scatter(r["age_mid"], r[metric], s=size, c=r["color"],
                       edgecolors="black", linewidth=0.8, zorder=3, marker="D")
            ax.errorbar(r["age_mid"], r[metric],
                        yerr=r[metric.replace("mean", "std")],
                        fmt="none", ecolor="#666666", capsize=3, zorder=2)

        # Annotations
        for r in results:
            short = (r["epoch"]
                     .replace("Early ", "e").replace("Middle ", "m")
                     .replace("Late ", "l").replace("Amazonian", "A")
                     .replace("Hesperian", "H").replace("Noachian", "N"))
            ox, oy = 8, 7
            if "mN" in short or "lH" in short:
                oy = -14
            ax.annotate(short, (r["age_mid"], r[metric]),
                        textcoords="offset points", xytext=(ox, oy),
                        fontsize=6.5, color="#333333")

        ax.set_xlabel("Surface Age (Ga)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.invert_xaxis()
        ax.set_xlim(4.3, -0.1)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.9)
        ax.grid(alpha=0.25)
        ax.xaxis.set_major_locator(MultipleLocator(0.5))

    # Legend for marker shapes
    ax2.scatter([], [], s=60, c="white", edgecolors="black", marker="o", label="Specific epoch")
    ax2.scatter([], [], s=60, c="white", edgecolors="black", marker="D", label="Transitional/undivided")
    ax2.legend(fontsize=8, loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out = plot_dir / "all_epoch_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot2_per_polygon_all_epochs(gdf, plot_dir):
    """Per-polygon scatter with ALL 14 epochs (none excluded)."""
    valid = gdf[gdf["n_pixels"] >= 5].copy()

    # Fit using specific-epoch aggregated means (excl. mAmazonian)
    spec_results = weighted_epoch_stats(gdf, list(SPECIFIC_EPOCHS))
    spec_results = [r for r in spec_results if r["epoch"] != "Middle Amazonian"]
    ages_s = np.array([r["age_mid"] for r in spec_results])
    js_s = np.array([r["js_mean"] for r in spec_results])
    ed_s = np.array([r["edge_mean"] for r in spec_results])
    w_s = np.sqrt(np.array([r["n_pixels"] for r in spec_results]))
    popt_js, r2_js = do_fit(ages_s, js_s, w_s, [0.05, 0.3, 0.3])
    popt_ed, r2_ed = do_fit(ages_s, ed_s, w_s, [5, 0.3, 20])
    age_smooth = np.linspace(0, 4.2, 200)

    colors = valid["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 5, 150)
    markers = valid["epoch"].map(lambda e: "D" if e not in SPECIFIC_EPOCHS else "o")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Per-Polygon JS Divergence & Edge Density vs Age — All 14 Epochs\n"
                 f"Each dot = one Tanaka et al. (2014) polygon ({len(valid)} with valid stats)",
                 fontsize=13, fontweight="bold")

    # Plot specific epoch polygons as circles
    spec_mask = valid["epoch"].isin(SPECIFIC_EPOCHS)
    trans_mask = ~spec_mask

    for ax, y_col, y_label, popt, r2, y_scale in [
        (ax1, "js_mean", "Mean JS Divergence", popt_js, r2_js, 1),
        (ax2, "edge_dens", "Edge Density (%)", popt_ed, r2_ed, 100),
    ]:
        # Specific epoch polygons
        v_spec = valid[spec_mask]
        ax.scatter(v_spec["age_mid"], v_spec[y_col] * y_scale,
                   s=sizes[spec_mask.values], c=colors[spec_mask].values,
                   edgecolors="black", linewidth=0.3, alpha=0.7, zorder=3, marker="o")

        # Transitional polygons
        v_trans = valid[trans_mask]
        ax.scatter(v_trans["age_mid"], v_trans[y_col] * y_scale,
                   s=sizes[trans_mask.values], c=colors[trans_mask].values,
                   edgecolors="black", linewidth=0.3, alpha=0.7, zorder=3, marker="D")

        # Fit line
        ax.plot(age_smooth, exp_model(age_smooth, *popt), "-", color="black",
                linewidth=2, zorder=4,
                label=f"Exp. fit (specific, excl. mAmaz)\n$R^2={r2:.3f}$")

        ax.set_xlabel("Surface Age (Ga)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.invert_xaxis()
        ax.set_xlim(4.3, -0.1)
        ax.legend(fontsize=9, loc="lower left")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = plot_dir / "per_polygon_all_epochs.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot3_boxplot_js_by_type(gdf, plot_dir):
    """Box plots of JS divergence by unit type, ordered by median."""
    valid = gdf[gdf["n_pixels"] >= 5].copy()
    type_order = (valid.groupby("unit_type")["js_mean"]
                  .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = [valid[valid["unit_type"] == t]["js_mean"].dropna().values for t in type_order]
    counts = [len(d) for d in data_list]

    bp = ax.boxplot(data_list, tick_labels=type_order, patch_artist=True, widths=0.6,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, t in zip(bp["boxes"], type_order):
        patch.set_facecolor(UNIT_TYPE_COLORS.get(t, "#CCCCCC"))
        patch.set_alpha(0.8)

    for i, c in enumerate(counts):
        ax.annotate(f"n={c}", (i + 1, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=8, color="gray")

    ax.set_ylabel("Mean JS Divergence (per polygon)", fontsize=11)
    ax.set_title("JS Divergence by Geological Unit Type\n"
                 "Ordered by median | Tanaka et al. (2014)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = plot_dir / "boxplot_js_by_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot4_boxplot_edge_by_type(gdf, plot_dir):
    """Box plots of edge density by unit type, ordered by median."""
    valid = gdf[gdf["n_pixels"] >= 5].copy()
    valid["edge_pct"] = valid["edge_dens"] * 100
    type_order = (valid.groupby("unit_type")["edge_pct"]
                  .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(10, 6))
    data_list = [valid[valid["unit_type"] == t]["edge_pct"].dropna().values for t in type_order]
    counts = [len(d) for d in data_list]

    bp = ax.boxplot(data_list, tick_labels=type_order, patch_artist=True, widths=0.6,
                    showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.5))
    for patch, t in zip(bp["boxes"], type_order):
        patch.set_facecolor(UNIT_TYPE_COLORS.get(t, "#CCCCCC"))
        patch.set_alpha(0.8)

    for i, c in enumerate(counts):
        ax.annotate(f"n={c}", (i + 1, ax.get_ylim()[1]),
                    textcoords="offset points", xytext=(0, -12),
                    ha="center", fontsize=8, color="gray")

    ax.set_ylabel("Edge Density (%) per polygon", fontsize=11)
    ax.set_title("Edge Density by Geological Unit Type\n"
                 "Ordered by median | Tanaka et al. (2014)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = plot_dir / "boxplot_edge_by_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot5_violin_js_by_type_epoch(gdf, plot_dir):
    """Swarm/strip plot: JS by unit type, colored by epoch era."""
    valid = gdf[gdf["n_pixels"] >= 5].copy()
    type_order = (valid.groupby("unit_type")["js_mean"]
                  .median().sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(12, 7))

    # Color by era (Noachian/Hesperian/Amazonian)
    def epoch_to_era_color(epoch):
        if "Noachian" in epoch and "Amazonian" not in epoch:
            return "#B35533"
        if "Hesperian" in epoch and "Amazonian" not in epoch:
            return "#6AA84F"
        if "Amazonian" in epoch and "Noachian" not in epoch and "Hesperian" not in epoch:
            return "#FFD700"
        return "#AAAAAA"  # transitional

    for i, t in enumerate(type_order):
        subset = valid[valid["unit_type"] == t]
        era_colors = subset["epoch"].map(epoch_to_era_color)
        # Add jitter
        jitter = np.random.default_rng(42).uniform(-0.2, 0.2, len(subset))
        ax.scatter(i + jitter, subset["js_mean"], c=era_colors, s=15,
                   alpha=0.6, edgecolors="none", zorder=3)

    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels(type_order, rotation=30, ha="right")
    ax.set_ylabel("Mean JS Divergence", fontsize=11)
    ax.set_title("JS Divergence by Unit Type, Colored by Era\n"
                 "Tanaka et al. (2014) — each dot = one polygon",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#B35533", label="Noachian"),
        mpatches.Patch(color="#6AA84F", label="Hesperian"),
        mpatches.Patch(color="#FFD700", label="Amazonian"),
        mpatches.Patch(color="#AAAAAA", label="Transitional"),
    ]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper right")

    plt.tight_layout()
    out = plot_dir / "strip_js_by_type_era.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot6_scatter_by_unit_type(gdf, plot_dir):
    """Per-polygon scatter colored by unit type instead of epoch."""
    valid = gdf[gdf["n_pixels"] >= 5].copy()
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 5, 150)

    # Fit line (same as before)
    spec_results = weighted_epoch_stats(gdf, list(SPECIFIC_EPOCHS))
    spec_results = [r for r in spec_results if r["epoch"] != "Middle Amazonian"]
    ages_s = np.array([r["age_mid"] for r in spec_results])
    js_s = np.array([r["js_mean"] for r in spec_results])
    ed_s = np.array([r["edge_mean"] for r in spec_results])
    w_s = np.sqrt(np.array([r["n_pixels"] for r in spec_results]))
    popt_js, r2_js = do_fit(ages_s, js_s, w_s, [0.05, 0.3, 0.3])
    popt_ed, r2_ed = do_fit(ages_s, ed_s, w_s, [5, 0.3, 20])
    age_smooth = np.linspace(0, 4.2, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle("Per-Polygon Age vs Texture Metrics — Colored by Unit Type\n"
                 "Does the age trend persist within each type?",
                 fontsize=13, fontweight="bold")

    for ax, y_col, y_label, popt, r2, y_scale in [
        (ax1, "js_mean", "Mean JS Divergence", popt_js, r2_js, 1),
        (ax2, "edge_dens", "Edge Density (%)", popt_ed, r2_ed, 100),
    ]:
        for utype in UNIT_TYPE_ORDER:
            sub = valid[valid["unit_type"] == utype]
            if len(sub) == 0:
                continue
            sub_sizes = np.clip(np.sqrt(sub["n_pixels"].values) / 3, 5, 150)
            ax.scatter(sub["age_mid"], sub[y_col] * y_scale, s=sub_sizes,
                       c=UNIT_TYPE_COLORS[utype], edgecolors="black", linewidth=0.2,
                       alpha=0.65, zorder=3, label=f"{utype} ({len(sub)})")

        ax.plot(age_smooth, exp_model(age_smooth, *popt), "-", color="black",
                linewidth=2, zorder=4)

        ax.set_xlabel("Surface Age (Ga)", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.invert_xaxis()
        ax.set_xlim(4.3, -0.1)
        ax.legend(fontsize=7.5, loc="lower left", framealpha=0.9, ncol=2)
        ax.grid(alpha=0.25)

    plt.tight_layout()
    out = plot_dir / "scatter_by_unit_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot7_js_std_vs_mean(gdf, plot_dir):
    """JS std vs JS mean per polygon — are edgier regions more variable?"""
    valid = gdf[(gdf["n_pixels"] >= 5) & gdf["js_std"].notna()].copy()
    colors = valid["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 5, 150)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(valid["js_mean"], valid["js_std"], s=sizes, c=colors,
               edgecolors="black", linewidth=0.2, alpha=0.7, zorder=3)

    # Linear fit
    x = valid["js_mean"].values
    y = valid["js_std"].values
    m, b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (m * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "-", color="black", linewidth=2, zorder=4,
            label=f"Linear fit: slope={m:.3f}\n$R^2$={r2:.3f}")

    ax.set_xlabel("Mean JS Divergence", fontsize=11)
    ax.set_ylabel("Std JS Divergence", fontsize=11)
    ax.set_title("JS Variability vs Mean — Per Polygon\n"
                 "Are rougher regions more heterogeneous?",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = plot_dir / "js_std_vs_mean.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot8_js_iqr_vs_age(gdf, plot_dir):
    """JS IQR (p75-p25) vs age — dispersion trend."""
    valid = gdf[(gdf["n_pixels"] >= 5) & gdf["js_p75"].notna()].copy()
    valid["js_iqr"] = valid["js_p75"] - valid["js_p25"]
    colors = valid["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 5, 150)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(valid["age_mid"], valid["js_iqr"], s=sizes, c=colors,
               edgecolors="black", linewidth=0.2, alpha=0.7, zorder=3)

    ax.set_xlabel("Surface Age (Ga)", fontsize=11)
    ax.set_ylabel("JS Divergence IQR (p75 - p25)", fontsize=11)
    ax.set_title("Within-Polygon JS Dispersion vs Age\n"
                 "Each dot = one Tanaka et al. (2014) polygon",
                 fontsize=12, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xlim(4.3, -0.1)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = plot_dir / "js_iqr_vs_age.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def plot9_js_vs_edge_density(gdf, plot_dir):
    """JS mean vs edge density scatter — how tightly coupled?"""
    valid = gdf[(gdf["n_pixels"] >= 5) & gdf["js_mean"].notna()].copy()
    colors = valid["epoch"].map(lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 5, 150)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(valid["js_mean"], valid["edge_dens"] * 100, s=sizes, c=colors,
               edgecolors="black", linewidth=0.2, alpha=0.7, zorder=3)

    # Linear fit
    x = valid["js_mean"].values
    y = valid["edge_dens"].values * 100
    m, b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (m * x + b)) ** 2) / np.sum((y - y.mean()) ** 2)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, "-", color="black", linewidth=2, zorder=4,
            label=f"Linear fit: slope={m:.1f}\n$R^2$={r2:.3f}")

    ax.set_xlabel("Mean JS Divergence", fontsize=11)
    ax.set_ylabel("Edge Density (%)", fontsize=11)
    ax.set_title("JS Divergence vs Edge Density — Per Polygon\n"
                 "Colored by epoch",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    out = plot_dir / "js_vs_edge_density.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")


def pca_analysis(gdf, plot_dir):
    """PCA on texture features + regression variance decomposition.

    Generates: scree plot, loadings heatmap, biplots (by epoch & unit type).
    Prints: OLS-style variance decomposition table for js_mean.
    """
    feat_cols = ["js_mean", "js_std", "js_median", "js_p25", "js_p75",
                 "edge_dens", "edge_std"]
    valid = gdf[(gdf["n_pixels"] >= 5) & gdf[feat_cols].notna().all(axis=1)].copy()
    logger.info(f"PCA: {len(valid)} polygons with complete texture data")

    X = valid[feat_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=len(feat_cols))
    scores = pca.fit_transform(X_scaled)
    loadings = pca.components_.T  # (n_features, n_components)
    evr = pca.explained_variance_ratio_

    logger.info("PCA explained variance: %s", np.round(evr * 100, 1))

    # --- Plot 10: Scree plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    pcs = np.arange(1, len(evr) + 1)
    bars = ax.bar(pcs, evr * 100, color="#4488CC", edgecolor="black", linewidth=0.6)
    cumulative = np.cumsum(evr) * 100
    ax.plot(pcs, cumulative, "o-", color="#CC4444", linewidth=2, markersize=6,
            label="Cumulative")
    for i, (v, c) in enumerate(zip(evr * 100, cumulative)):
        ax.text(i + 1, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color="#333333")
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance (%)", fontsize=11)
    ax.set_title("PCA Scree Plot — 7 Texture / Morphometric Features\n"
                 "Standardized features from Tanaka et al. (2014) polygons",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(pcs)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="center right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = plot_dir / "pca_scree.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")

    # --- Plot 11: Loadings heatmap ---
    n_show = min(4, len(feat_cols))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(loadings[:, :n_show], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"PC{i+1}\n({evr[i]*100:.1f}%)" for i in range(n_show)],
                       fontsize=10)
    ax.set_yticks(range(len(feat_cols)))
    ax.set_yticklabels(feat_cols, fontsize=10)
    # Annotate cells
    for i in range(len(feat_cols)):
        for j in range(n_show):
            val = loadings[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    ax.set_title("PCA Loadings — Features × Components\n"
                 "Standardized texture metrics",
                 fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Loading")
    plt.tight_layout()
    out = plot_dir / "pca_loadings.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")

    # --- Plot 12: Biplot colored by epoch ---
    epoch_colors = valid["epoch"].map(
        lambda e: EPOCH_INFO.get(e, {}).get("color", "#888888"))
    sizes = np.clip(np.sqrt(valid["n_pixels"].values) / 3, 8, 120)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(scores[:, 0], scores[:, 1], s=sizes, c=epoch_colors.values,
               edgecolors="black", linewidth=0.3, alpha=0.7, zorder=3)

    # Loading vectors
    scale = np.abs(scores[:, :2]).max() * 0.8
    for i, feat in enumerate(feat_cols):
        ax.annotate(
            "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
        ax.text(loadings[i, 0] * scale * 1.08, loadings[i, 1] * scale * 1.08,
                feat, fontsize=7.5, color="#333333", ha="center", va="center")

    ax.axhline(0, color="gray", linewidth=0.5, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, zorder=1)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA Biplot — Colored by Epoch\n"
                 "Does surface age separate along PC1?",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)

    # Epoch legend (simplified by era)
    era_patches = [
        mpatches.Patch(color="#993311", label="Early Noachian"),
        mpatches.Patch(color="#B35533", label="Middle Noachian"),
        mpatches.Patch(color="#CC7755", label="Late Noachian"),
        mpatches.Patch(color="#6AA84F", label="Early Hesperian"),
        mpatches.Patch(color="#93C47D", label="Late Hesperian"),
        mpatches.Patch(color="#FFD700", label="Early Amazonian"),
        mpatches.Patch(color="#FFE066", label="Middle Amazonian"),
        mpatches.Patch(color="#FFF2B2", label="Late Amazonian"),
    ]
    ax.legend(handles=era_patches, fontsize=7, loc="upper right",
              framealpha=0.9, ncol=2, title="Epoch", title_fontsize=8)
    plt.tight_layout()
    out = plot_dir / "pca_biplot_epoch.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")

    # --- Plot 13: Biplot colored by unit type ---
    fig, ax = plt.subplots(figsize=(9, 7))
    for utype in UNIT_TYPE_ORDER:
        mask = valid["unit_type"].values == utype
        if not mask.any():
            continue
        ax.scatter(scores[mask, 0], scores[mask, 1], s=sizes[mask],
                   c=UNIT_TYPE_COLORS[utype], edgecolors="black", linewidth=0.3,
                   alpha=0.7, zorder=3, label=f"{utype} ({mask.sum()})")

    # Loading vectors (same as above)
    for i, feat in enumerate(feat_cols):
        ax.annotate(
            "", xy=(loadings[i, 0] * scale, loadings[i, 1] * scale),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2))
        ax.text(loadings[i, 0] * scale * 1.08, loadings[i, 1] * scale * 1.08,
                feat, fontsize=7.5, color="#333333", ha="center", va="center")

    ax.axhline(0, color="gray", linewidth=0.5, zorder=1)
    ax.axvline(0, color="gray", linewidth=0.5, zorder=1)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("PCA Biplot — Colored by Unit Type\n"
                 "Do geological types separate orthogonally to age?",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9,
              title="Unit Type", title_fontsize=9)
    plt.tight_layout()
    out = plot_dir / "pca_biplot_type.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    logger.info(f"Saved: {out}")

    # --- Regression / Variance Decomposition ---
    # Predict js_mean from age_mid, unit_type, log10(area)
    reg = valid[["js_mean", "age_mid", "unit_type", "SphArea_km"]].copy()
    reg["log_area"] = np.log10(reg["SphArea_km"].clip(lower=1e-6))
    y = reg["js_mean"].values

    # Design matrices for sequential R² decomposition
    # 1) age alone
    X_age = reg[["age_mid"]].values
    r2_age = _ols_r2(X_age, y)
    # 2) unit_type alone (dummy-coded)
    type_dummies = pd.get_dummies(reg["unit_type"], drop_first=True).values.astype(float)
    r2_type = _ols_r2(type_dummies, y)
    # 3) log_area alone
    X_area = reg[["log_area"]].values
    r2_area = _ols_r2(X_area, y)
    # 4) age + type
    X_age_type = np.hstack([X_age, type_dummies])
    r2_age_type = _ols_r2(X_age_type, y)
    # 5) age + area
    X_age_area = np.hstack([X_age, X_area])
    r2_age_area = _ols_r2(X_age_area, y)
    # 6) type + area
    X_type_area = np.hstack([type_dummies, X_area])
    r2_type_area = _ols_r2(X_type_area, y)
    # 7) full model: age + type + log_area
    X_full = np.hstack([X_age, type_dummies, X_area])
    r2_full = _ols_r2(X_full, y)

    print("\n" + "=" * 80)
    print("VARIANCE DECOMPOSITION: js_mean ~ age_mid + C(unit_type) + log10(SphArea_km)")
    print(f"n = {len(y)} polygons with complete data")
    print("=" * 80)
    print(f"\n{'Factor':<24s} {'R² alone':>10s} {'ΔR² added last':>16s}")
    print("-" * 52)
    print(f"{'age_mid':<24s} {r2_age:>10.4f} {r2_full - r2_type_area:>16.4f}")
    print(f"{'C(unit_type)':<24s} {r2_type:>10.4f} {r2_full - r2_age_area:>16.4f}")
    print(f"{'log10(SphArea_km)':<24s} {r2_area:>10.4f} {r2_full - r2_age_type:>16.4f}")
    print("-" * 52)
    print(f"{'age + type':<24s} {r2_age_type:>10.4f}")
    print(f"{'age + area':<24s} {r2_age_area:>10.4f}")
    print(f"{'type + area':<24s} {r2_type_area:>10.4f}")
    print(f"{'Full model':<24s} {r2_full:>10.4f}")
    print("=" * 80)

    # Interpretation
    print("\nInterpretation:")
    if r2_age_type - r2_age > 0.05:
        print(f"  - Unit type adds {r2_age_type - r2_age:.3f} R² beyond age alone"
              " → type provides independent information")
    else:
        print(f"  - Unit type adds only {r2_age_type - r2_age:.3f} R² beyond age"
              " → largely redundant with age")
    if r2_full - r2_age_type > 0.01:
        print(f"  - Polygon area adds {r2_full - r2_age_type:.3f} R² beyond age+type"
              " → area matters")
    else:
        print(f"  - Polygon area adds only {r2_full - r2_age_type:.3f} R²"
              " → negligible effect")
    print(f"  - Full model explains {r2_full*100:.1f}% of js_mean variance")
    print(f"  - Residual unexplained: {(1 - r2_full)*100:.1f}%\n")


def _ols_r2(X, y):
    """Compute R² for OLS regression of y on X (with intercept)."""
    X_int = np.column_stack([np.ones(len(X)), X])
    beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
    y_pred = X_int @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1 - ss_res / ss_tot


def print_summary_table(gdf):
    """Print comprehensive summary table to stdout."""
    valid = gdf[gdf["n_pixels"] >= 5]

    # Per-epoch table
    print("\n" + "=" * 100)
    print("PER-EPOCH SUMMARY (weighted by n_pixels)")
    print("=" * 100)
    all_results = weighted_epoch_stats(gdf, list(EPOCH_INFO.keys()))
    print(f"{'Epoch':<22s} {'Age':>5s} {'Type':>10s} {'#poly':>6s} {'#px':>10s} "
          f"{'JS mean':>8s} {'JS std':>8s} {'Edge%':>7s} {'E std':>7s}")
    print("-" * 100)
    for r in all_results:
        tag = "specific" if r["is_specific"] else "trans."
        print(f"{r['epoch']:<22s} {r['age_mid']:>5.2f} {tag:>10s} {r['n_polygons']:>6d} "
              f"{r['n_pixels']:>10,d} {r['js_mean']:>8.4f} {r['js_std']:>8.4f} "
              f"{r['edge_mean']:>6.2f}% {r['edge_std']:>6.2f}%")

    # Per-unit-type table
    print("\n" + "=" * 100)
    print("PER-UNIT-TYPE SUMMARY")
    print("=" * 100)
    print(f"{'Unit Type':<14s} {'#poly':>6s} {'#w/stats':>8s} {'JS mean':>9s} "
          f"{'JS med':>8s} {'JS std':>8s} {'Edge%':>8s} {'E med%':>8s}")
    print("-" * 100)
    for utype in UNIT_TYPE_ORDER:
        sub_all = gdf[gdf["unit_type"] == utype]
        sub = valid[valid["unit_type"] == utype]
        if len(sub) == 0:
            print(f"{utype:<14s} {len(sub_all):>6d} {0:>8d}    (no valid stats)")
            continue
        print(f"{utype:<14s} {len(sub_all):>6d} {len(sub):>8d} "
              f"{sub['js_mean'].mean():>9.4f} {sub['js_mean'].median():>8.4f} "
              f"{sub['js_mean'].std():>8.4f} "
              f"{sub['edge_dens'].mean() * 100:>7.2f}% "
              f"{sub['edge_dens'].median() * 100:>7.2f}%")

    print("\n" + "=" * 100)
    print(f"Total polygons: {len(gdf)}")
    print(f"Polygons with valid stats (n_pixels >= 5): {len(valid)}")
    print(f"Polygons without stats: {len(gdf) - len(valid)}")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info(f"Reading GeoPackage: {GPKG_PATH}")
    gdf = gpd.read_file(GPKG_PATH)
    logger.info(f"Loaded {len(gdf)} polygons, columns: {list(gdf.columns)}")

    # Add unit_type column
    gdf["unit_type"] = gdf["Unit"].apply(unit_to_type)
    logger.info(f"Unit type distribution:\n{gdf['unit_type'].value_counts().to_string()}")

    # Save updated GeoPackage with unit_type
    logger.info("Saving updated GeoPackage with unit_type column...")
    gdf.to_file(GPKG_PATH, driver="GPKG")
    logger.info("GeoPackage updated.")

    # Create plots
    PLOT_DIR.mkdir(exist_ok=True)

    plot1_all_epoch_scatter(gdf, PLOT_DIR)
    plot2_per_polygon_all_epochs(gdf, PLOT_DIR)
    plot3_boxplot_js_by_type(gdf, PLOT_DIR)
    plot4_boxplot_edge_by_type(gdf, PLOT_DIR)
    plot5_violin_js_by_type_epoch(gdf, PLOT_DIR)
    plot6_scatter_by_unit_type(gdf, PLOT_DIR)
    plot7_js_std_vs_mean(gdf, PLOT_DIR)
    plot8_js_iqr_vs_age(gdf, PLOT_DIR)
    plot9_js_vs_edge_density(gdf, PLOT_DIR)

    # PCA + variance decomposition
    pca_analysis(gdf, PLOT_DIR)

    # Summary table
    print_summary_table(gdf)

    logger.info("Done — all plots saved to %s", PLOT_DIR)


if __name__ == "__main__":
    main()
