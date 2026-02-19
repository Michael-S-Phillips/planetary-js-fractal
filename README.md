# planetary-js-fractal

Tools for detecting and characterizing geological surface texture in planetary rasters using Jensen-Shannon (JS) divergence edge detection, multiscale fractal dimension mapping, and unsupervised classification. Developed for Mars but works on any single-band GeoTIFF.

## Pipeline

```
Input raster
    │
    ▼
js_edge_detect.py       → *_js.tif (multi-band JS divergence)
                        → *_edges.tif (binary edge map)
    │
    ▼
js_fractal_classify.py  → *_fractal.tif (fractal dimension map)
                        → *_kmeans_k10.tif (K-Means classification)
                        → *_fractal_classes.tif (binned fractal classes)
    │
    ▼
zonal_stats_geologic.py → tanaka2014_js_stats.gpkg (per-polygon statistics)
    │
    ▼
explore_zonal_stats.py  → plots/ (PCA, scatter, boxplots, etc.)
```

## Installation

```bash
pip install numpy rasterio scipy scikit-learn joblib geopandas matplotlib numba
```

Numba is optional but strongly recommended — without it, edge detection falls back to pure numpy and will be very slow on large rasters.

## Usage

### 1. Edge Detection (`js_edge_detect.py`)

Computes per-pixel JS divergence between opposing half-windows at multiple radii and four directions (N-S, E-W, NE-SW, NW-SE), then applies Otsu thresholding to produce a binary edge map.

Non-uint8 inputs are automatically rescaled to uint8 [1–255] globally (or per-tile with `--local-rescale`). NoData is mapped to 0.

```bash
# Basic usage (uint8 input, nodata from metadata)
python scripts/js_edge_detect.py input.tif -o output/prefix

# Float32 input with NaN nodata
python scripts/js_edge_detect.py thermal_inertia.tif -o out/ti --nodata nan

# int16 DEM with per-tile normalization (recommended for DEMs)
python scripts/js_edge_detect.py dem.tif -o out/dem --nodata -32768 --local-rescale

# Custom radii; skip binary edge map (JS only)
python scripts/js_edge_detect.py input.tif -o out --radii 5 10 20 --js-only

# Resume after a crash (strip-level checkpointing)
python scripts/js_edge_detect.py input.tif -o out --resume-row 42
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--radii` | `5 7 10 14 20` | Window radii in pixels |
| `--bins` | `32` | Histogram bins for JS computation |
| `--tile-size` | `1024` | Processing tile size |
| `--nodata` | auto | NoData value; `nan`, `none`, or a number |
| `--band` | `1` | Band to read from input |
| `--local-rescale` | off | Normalize each tile independently |
| `--single-band` | off | Output mean of all radii (disables fractal mode) |
| `--js-only` | off | Skip Phase 2 (binary edge map) |
| `--no-numba` | off | Force pure-numpy fallback |

**Outputs:**
- `*_js.tif` — float32, one band per radius; band descriptions: `JS_R5`, `JS_R7`, etc.
- `*_edges.tif` — uint8; 0 = no edge, 1 = edge, 255 = nodata

---

### 2. Fractal Analysis & Classification (`js_fractal_classify.py`)

Two modes:

**`edge-density` (default):** Takes a binary edge map, computes multiscale edge density using Summed Area Tables (O(1) per-pixel window queries), and fits `D = 2 + slope(log(density) vs log(R))` per pixel.

**`js-scaling`:** Takes the multi-band JS divergence output, fits `slope(log(JS) vs log(R))` per pixel directly.

Both modes then run MiniBatchKMeans on the multiscale feature vectors and bin the fractal dimension into discrete classes.

```bash
# Edge-density mode (from binary edge map)
python scripts/js_fractal_classify.py output/prefix_edges.tif -o output/prefix

# JS-scaling mode (from multi-band JS divergence)
python scripts/js_fractal_classify.py output/prefix_js.tif -o output/prefix --mode js-scaling

# Custom radii, skip K-Means
python scripts/js_fractal_classify.py edges.tif -o out --radii 8 16 32 64 128 --skip-kmeans
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `edge-density` | `edge-density` or `js-scaling` |
| `--radii` | mode-dependent | Radii for density/scaling computation |
| `--n-clusters` | `10` | K-Means clusters |
| `--n-samples` | `2,000,000` | Pixels to reservoir-sample for K-Means training |
| `--fractal-bin-edges` | mode-dependent | Bin edges for fractal classification |
| `--skip-kmeans` | off | Skip K-Means pass |
| `--skip-fractal-binned` | off | Skip binned classification |

**Outputs:**
- `*_fractal.tif` — float32 fractal dimension (or JS scaling slope)
- `*_kmeans_k{n}.tif` — uint8 K-Means labels (255 = nodata)
- `*_fractal_classes.tif` — uint8 binned fractal classes (255 = nodata)
- `*_kmeans_k{n}_model.joblib` — saved K-Means model

---

### 3. Zonal Statistics (`zonal_stats_geologic.py`)

Computes per-polygon JS divergence and edge density statistics against the Tanaka et al. (2014) global geologic map of Mars (SIM3292). Rasterizes polygon IDs at reduced resolution (~5 km), reads the JS and edge rasters, and saves stats to a GeoPackage.

Paths are currently hardcoded — edit the path variables near the top of `main()` before running.

```bash
python scripts/zonal_stats_geologic.py
```

**Output:** `tanaka2014_js_stats.gpkg` with fields: `js_mean`, `js_std`, `js_median`, `js_p25`, `js_p75`, `edge_dens`, `edge_std`, `n_pixels`, `epoch`, `age_mid`, `unit_type`.

---

### 4. Exploratory Analysis (`explore_zonal_stats.py`)

Reads the GeoPackage produced above and generates a full suite of plots (age vs. texture metrics, boxplots by unit type, PCA biplot, etc.) plus a variance decomposition table printed to stdout.

```bash
python scripts/explore_zonal_stats.py
```

Edit `GPKG_PATH` and `PLOT_DIR` at the top of the script before running.

---

## Technical Notes

**Integral histograms:** JS divergence is computed via integral histograms, giving O(1) per-pixel rectangular window queries regardless of radius size.

**Four-direction edge detection:** At each radius R, JS divergence is computed between opposing half-windows in four orientations (N/S, E/W, NE/SW, NW/SE); the maximum is taken. This makes detection rotation-invariant.

**Smoothing:** After JS computation, a directional median filter (four oriented 3-pixel medians, max with center) is applied to suppress noise while preserving edge sharpness.

**Fractal dimension:** Computed as D = 2 + β where β is the OLS slope of log(density) vs log(R) across the set of radii. Values above 2 indicate increasingly complex, multi-scale texture.

**K-Means training:** Reservoir sampling (Vitter's Algorithm R) is used to collect a fixed-size random sample of feature vectors from arbitrarily large rasters without loading them fully into memory.

**Strip-based tiling:** Both scripts process rasters in horizontal strips with configurable overlap/padding, keeping memory usage bounded regardless of raster size.
