#!/usr/bin/env python3
"""
Compare exponential fits: all epochs vs mAmazonian excluded.
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Data from the zonal analysis (specific sub-epochs only)
data = [
    # (epoch_name, age_mid_Ga, js_mean, edge_density_frac, color, n_pixels)
    ("Early Noachian",   3.95, 0.4488, 0.4255, "#993311", 610536),
    ("Middle Noachian",  3.85, 0.4233, 0.3847, "#B35533", 2955907),
    ("Late Noachian",    3.75, 0.4082, 0.3570, "#CC7755", 236300),
    ("Early Hesperian",  3.55, 0.3792, 0.3078, "#6AA84F", 309462),
    ("Late Hesperian",   3.35, 0.3742, 0.2949, "#93C47D", 957031),
    ("Early Amazonian",  2.40, 0.3533, 0.2392, "#FFD700", 22095),
    ("Middle Amazonian", 1.20, 0.3650, 0.2776, "#FFE066", 152616),
    ("Late Amazonian",   0.30, 0.3227, 0.2298, "#FFF2B2", 317222),
]

names = np.array([d[0] for d in data])
ages = np.array([d[1] for d in data])
js_means = np.array([d[2] for d in data])
edge_dens = np.array([d[3] for d in data]) * 100
colors = [d[4] for d in data]
n_pixels = np.array([d[5] for d in data])
sizes = np.clip(np.sqrt(n_pixels) / 8, 30, 300)
short = [n.replace("Early ", "e").replace("Middle ", "m").replace("Late ", "l")
         for n in names]

# Mask for excluding mAmazonian
exclude_idx = 6  # Middle Amazonian
mask_excl = np.ones(len(data), dtype=bool)
mask_excl[exclude_idx] = False


def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c


def do_fit(x, y, w, p0):
    popt, _ = curve_fit(exp_model, x, y, p0=p0, sigma=1.0/w, maxfev=10000)
    y_pred = exp_model(x, *popt)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot
    return popt, r2, y_pred


# Fits on ALL data
w_all = np.sqrt(n_pixels)
popt_js_all, r2_js_all, js_pred_all = do_fit(ages, js_means, w_all, [0.05, 0.3, 0.3])
popt_ed_all, r2_ed_all, ed_pred_all = do_fit(ages, edge_dens, w_all, [5, 0.3, 20])

# Fits EXCLUDING mAmazonian
ages_ex = ages[mask_excl]
js_ex = js_means[mask_excl]
ed_ex = edge_dens[mask_excl]
w_ex = np.sqrt(n_pixels[mask_excl])
popt_js_ex, r2_js_ex, _ = do_fit(ages_ex, js_ex, w_ex, [0.05, 0.3, 0.3])
popt_ed_ex, r2_ed_ex, _ = do_fit(ages_ex, ed_ex, w_ex, [5, 0.3, 20])

# R² of excluded fit evaluated on ALL points (including mAmazonian)
js_pred_ex_all = exp_model(ages, *popt_js_ex)
ed_pred_ex_all = exp_model(ages, *popt_ed_ex)
ss_res_js_ex_all = np.sum((js_means - js_pred_ex_all)**2)
ss_tot_js_all = np.sum((js_means - js_means.mean())**2)
r2_js_ex_allpts = 1 - ss_res_js_ex_all / ss_tot_js_all
ss_res_ed_ex_all = np.sum((edge_dens - ed_pred_ex_all)**2)
ss_tot_ed_all = np.sum((edge_dens - edge_dens.mean())**2)
r2_ed_ex_allpts = 1 - ss_res_ed_ex_all / ss_tot_ed_all

age_smooth = np.linspace(0, 4.2, 200)

# =====================================================================
# 2x2 comparison plot
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Exponential Fit Comparison: All Epochs vs Excluding Middle Amazonian\n"
             "THEMIS Day IR 100m / Tanaka et al. (2014)",
             fontsize=14, fontweight="bold", y=0.98)

titles = [
    "JS Divergence — All Epochs",
    "JS Divergence — Excl. mAmazonian",
    "Edge Density — All Epochs",
    "Edge Density — Excl. mAmazonian",
]

popts = [popt_js_all, popt_js_ex, popt_ed_all, popt_ed_ex]
r2s = [r2_js_all, r2_js_ex, r2_ed_all, r2_ed_ex]
ydata = [js_means, js_means, edge_dens, edge_dens]
ylabels = ["Mean JS Divergence", "Mean JS Divergence",
           "Edge Density (%)", "Edge Density (%)"]

for idx, ax in enumerate(axes.flat):
    popt = popts[idx]
    r2 = r2s[idx]
    yd = ydata[idx]
    a, b, c = popt
    is_excluded = (idx % 2 == 1)  # right column = excluded

    # Fit curve
    fit_curve = exp_model(age_smooth, *popt)
    ax.plot(age_smooth, fit_curve, "-", color="#444444", linewidth=2.5, zorder=2,
            label=f"$y = {a:.4f}\,e^{{{b:.3f}x}} + {c:.3f}$\n$R^2 = {r2:.4f}$")

    # Data points
    for i in range(len(data)):
        is_mAmaz = (i == exclude_idx)
        if is_excluded and is_mAmaz:
            # Draw mAmazonian as hollow / X marker
            ax.scatter(ages[i], yd[i], s=sizes[i], facecolors="none",
                      edgecolors="red", linewidth=1.5, zorder=5, marker="o")
            ax.scatter(ages[i], yd[i], s=30, c="red", marker="x",
                      linewidth=1.5, zorder=6)
        else:
            ax.scatter(ages[i], yd[i], s=sizes[i], c=colors[i],
                      edgecolors="black", linewidth=0.8, zorder=3)

    # Annotations
    for i, s in enumerate(short):
        # Offset adjustments
        if s == "mNoachian":
            ox, oy = 8, -14
        elif s == "lHesperian":
            ox, oy = 8, -14
        elif s == "lAmazonian":
            ox, oy = 8, -12
        elif s == "mAmazonian" and is_excluded:
            ox, oy = 10, 10
        else:
            ox, oy = 8, 7
        color = "red" if (is_excluded and i == exclude_idx) else "#333333"
        weight = "bold" if (is_excluded and i == exclude_idx) else "normal"
        ax.annotate(s, (ages[i], yd[i]),
                   textcoords="offset points", xytext=(ox, oy),
                   fontsize=7.5, color=color, fontweight=weight)

    ax.set_title(titles[idx], fontsize=11, fontweight="bold")
    ax.set_xlabel("Surface Age (Ga)", fontsize=10)
    ax.set_ylabel(ylabels[idx], fontsize=10)
    ax.invert_xaxis()
    ax.set_xlim(4.3, -0.1)
    ax.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))

plt.tight_layout()
out = "/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/age_vs_edge_expfit_compare.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

# Summary table
print("\n" + "=" * 75)
print(f"{'Metric':<20s} {'All epochs':>25s}  {'Excl. mAmazonian':>25s}")
print("-" * 75)

a, b, c = popt_js_all
a2, b2, c2 = popt_js_ex
print(f"{'JS a':<20s} {a:>25.6f}  {a2:>25.6f}")
print(f"{'JS b':<20s} {b:>25.4f}  {b2:>25.4f}")
print(f"{'JS c':<20s} {c:>25.4f}  {c2:>25.4f}")
print(f"{'JS R²':<20s} {r2_js_all:>25.4f}  {r2_js_ex:>25.4f}")

print()
a, b, c = popt_ed_all
a2, b2, c2 = popt_ed_ex
print(f"{'Edge a':<20s} {a:>25.6f}  {a2:>25.6f}")
print(f"{'Edge b':<20s} {b:>25.4f}  {b2:>25.4f}")
print(f"{'Edge c':<20s} {c:>25.4f}  {c2:>25.4f}")
print(f"{'Edge R²':<20s} {r2_ed_all:>25.4f}  {r2_ed_ex:>25.4f}")
print("=" * 75)

# Residuals for excluded fit on all points
print("\nResiduals for excl-mAmaz fit evaluated on ALL points:")
print(f"\n  JS Divergence (R² on all pts = {r2_js_ex_allpts:.4f}):")
for i, n in enumerate(names):
    flag = " ***" if i == exclude_idx else ""
    print(f"    {n:<20s}  obs={js_means[i]:.4f}  pred={js_pred_ex_all[i]:.4f}  "
          f"resid={js_means[i]-js_pred_ex_all[i]:+.4f}{flag}")

print(f"\n  Edge Density (R² on all pts = {r2_ed_ex_allpts:.4f}):")
for i, n in enumerate(names):
    flag = " ***" if i == exclude_idx else ""
    print(f"    {n:<20s}  obs={edge_dens[i]:.2f}%  pred={ed_pred_ex_all[i]:.2f}%  "
          f"resid={edge_dens[i]-ed_pred_ex_all[i]:+.2f}%{flag}")
