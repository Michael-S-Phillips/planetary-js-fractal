#!/usr/bin/env python3
"""
Exponential fit of geological age vs JS divergence and edge density.
Uses results from the earlier zonal analysis.
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

names = [d[0] for d in data]
ages = np.array([d[1] for d in data])
js_means = np.array([d[2] for d in data])
edge_dens = np.array([d[3] for d in data]) * 100  # percent
colors = [d[4] for d in data]
n_pixels = np.array([d[5] for d in data])

# Marker sizes proportional to sqrt(n_pixels)
sizes = np.clip(np.sqrt(n_pixels) / 8, 30, 300)

# Short labels for annotations
short = [n.replace("Early ", "e").replace("Middle ", "m").replace("Late ", "l")
         for n in names]


def exp_model(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


# --- Fit JS divergence vs age ---
# Weights by sqrt(n_pixels) to give more populated epochs more influence
weights_js = np.sqrt(n_pixels)
popt_js, pcov_js = curve_fit(exp_model, ages, js_means, p0=[0.05, 0.3, 0.3],
                              sigma=1.0/weights_js, maxfev=10000)
a_js, b_js, c_js = popt_js

# R² for JS fit
js_pred = exp_model(ages, *popt_js)
ss_res_js = np.sum((js_means - js_pred)**2)
ss_tot_js = np.sum((js_means - js_means.mean())**2)
r2_js = 1 - ss_res_js / ss_tot_js

# --- Fit edge density vs age ---
weights_ed = np.sqrt(n_pixels)
popt_ed, pcov_ed = curve_fit(exp_model, ages, edge_dens, p0=[5, 0.3, 20],
                              sigma=1.0/weights_ed, maxfev=10000)
a_ed, b_ed, c_ed = popt_ed

# R² for edge density fit
ed_pred = exp_model(ages, *popt_ed)
ss_res_ed = np.sum((edge_dens - ed_pred)**2)
ss_tot_ed = np.sum((edge_dens - edge_dens.mean())**2)
r2_ed = 1 - ss_res_ed / ss_tot_ed

# Smooth curve for plotting
age_smooth = np.linspace(0, 4.2, 200)
js_fit = exp_model(age_smooth, *popt_js)
ed_fit = exp_model(age_smooth, *popt_ed)

# =====================================================================
# Plot
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.suptitle("Mars Geological Age vs Edge Characteristics\n"
             "THEMIS Day IR 100m / Tanaka et al. (2014) geologic units",
             fontsize=13, fontweight="bold", y=0.98)

# --- Panel 1: JS Divergence ---
ax1.plot(age_smooth, js_fit, "-", color="#444444", linewidth=2, zorder=2,
         label=f"$y = {a_js:.3f}\,e^{{{b_js:.3f}x}} + {c_js:.3f}$\n$R^2 = {r2_js:.3f}$")

for i in range(len(data)):
    ax1.scatter(ages[i], js_means[i], s=sizes[i], c=colors[i],
               edgecolors="black", linewidth=0.8, zorder=3)

# Annotations with smart offset
offsets_js = {
    "eNoachian": (8, 8), "mNoachian": (8, -12), "lNoachian": (8, 6),
    "eHesperian": (8, 8), "lHesperian": (8, -12),
    "eAmazonian": (8, 6), "mAmazonian": (8, 8), "lAmazonian": (8, -10),
}
for i, s in enumerate(short):
    ox, oy = offsets_js.get(s, (8, 5))
    ax1.annotate(s, (ages[i], js_means[i]),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5) if abs(ox) > 6 else None)

ax1.set_xlabel("Surface Age (Ga)", fontsize=12)
ax1.set_ylabel("Mean JS Divergence", fontsize=12)
ax1.set_title("JS Divergence vs Age", fontsize=12)
ax1.invert_xaxis()
ax1.set_xlim(4.3, -0.1)
ax1.legend(fontsize=10, loc="lower left", framealpha=0.9)
ax1.grid(alpha=0.25)
ax1.xaxis.set_major_locator(MultipleLocator(0.5))

# --- Panel 2: Edge Density ---
ax2.plot(age_smooth, ed_fit, "-", color="#444444", linewidth=2, zorder=2,
         label=f"$y = {a_ed:.3f}\,e^{{{b_ed:.3f}x}} + {c_ed:.3f}$\n$R^2 = {r2_ed:.3f}$")

for i in range(len(data)):
    ax2.scatter(ages[i], edge_dens[i], s=sizes[i], c=colors[i],
               edgecolors="black", linewidth=0.8, zorder=3)

offsets_ed = {
    "eNoachian": (8, 8), "mNoachian": (8, -12), "lNoachian": (8, 6),
    "eHesperian": (8, 8), "lHesperian": (8, -12),
    "eAmazonian": (8, 6), "mAmazonian": (8, 8), "lAmazonian": (8, -10),
}
for i, s in enumerate(short):
    ox, oy = offsets_ed.get(s, (8, 5))
    ax2.annotate(s, (ages[i], edge_dens[i]),
                textcoords="offset points", xytext=(ox, oy),
                fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5) if abs(ox) > 6 else None)

ax2.set_xlabel("Surface Age (Ga)", fontsize=12)
ax2.set_ylabel("Edge Density (%)", fontsize=12)
ax2.set_title("Edge Density vs Age", fontsize=12)
ax2.invert_xaxis()
ax2.set_xlim(4.3, -0.1)
ax2.legend(fontsize=10, loc="lower left", framealpha=0.9)
ax2.grid(alpha=0.25)
ax2.xaxis.set_major_locator(MultipleLocator(0.5))

# Size legend
for npix, label in [(50000, "50K"), (500000, "500K"), (2000000, "2M")]:
    sz = np.clip(np.sqrt(npix) / 8, 30, 300)
    ax2.scatter([], [], s=sz, c="white", edgecolors="black", linewidth=0.5,
               label=f"n = {label} px")
ax2.legend(fontsize=9, loc="lower left", framealpha=0.9, ncol=1)

plt.tight_layout()
out = "/Volumes/Rohan/Mars_GIS_Data/THEMIS/js_edges/age_vs_edge_expfit.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")

# Print fit parameters
print(f"\nJS Divergence fit: y = {a_js:.4f} * exp({b_js:.4f} * age) + {c_js:.4f}  (R² = {r2_js:.4f})")
print(f"Edge Density fit:  y = {a_ed:.4f} * exp({b_ed:.4f} * age) + {c_ed:.4f}  (R² = {r2_ed:.4f})")

# Residuals
print(f"\nResiduals (JS):")
for i, n in enumerate(names):
    print(f"  {n:<20s}  observed={js_means[i]:.4f}  predicted={js_pred[i]:.4f}  residual={js_means[i]-js_pred[i]:+.4f}")
print(f"\nResiduals (Edge Density %):")
for i, n in enumerate(names):
    print(f"  {n:<20s}  observed={edge_dens[i]:.2f}%  predicted={ed_pred[i]:.2f}%  residual={edge_dens[i]-ed_pred[i]:+.2f}%")
