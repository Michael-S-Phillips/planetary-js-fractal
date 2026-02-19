#!/usr/bin/env python3
"""
Multiscale Fractal Analysis & Unsupervised Classification

Two modes:
  edge-density: Takes binary edge map, computes D = 2 + slope(log(density) vs log(R))
                via summed area tables (SATs) for O(1) per-pixel window queries.
  js-scaling:   Takes multi-band JS divergence, computes slope(log(JS) vs log(R))
                per pixel. K-Means on raw JS feature vectors.

Outputs:
  1. Fractal dimension map (single-band float32)
  2. K-Means classification (uint8)
  3. Fractal-binned classification (uint8)

Author: Michael S. Phillips
Date: 2026-02-17
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger("js_fractal_classify")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TILE_SIZE = 1024
DEFAULT_N_CLUSTERS = 10
DEFAULT_N_SAMPLES = 2_000_000
NODATA_UINT8 = 255

# Mode-specific defaults
EDGE_DENSITY_RADII = [8, 16, 32, 64, 128]
EDGE_DENSITY_BIN_EDGES = [1.2, 1.4, 1.6, 1.8]

JS_SCALING_RADII = [5, 7, 10, 14, 20]
JS_SCALING_BIN_EDGES = [-0.5, -0.2, 0.0, 0.2, 0.5]


# ---------------------------------------------------------------------------
# OLS Slope Weights
# ---------------------------------------------------------------------------

def precompute_slope_weights(radii):
    """Precompute OLS weights so slope = dot(weights, log_values).

    For fixed x = log(radii), OLS slope = sum(w_i * y_i) where:
        w_i = (x_i - x_mean) / sum((x_j - x_mean)^2)

    Returns:
        weights: array of shape (n_radii,) for dot product with log(density)
    """
    log_r = np.log(np.array(radii, dtype=np.float64))
    x_mean = log_r.mean()
    x_centered = log_r - x_mean
    ss = np.sum(x_centered ** 2)
    if ss == 0:
        return np.zeros_like(log_r)
    weights = x_centered / ss
    return weights.astype(np.float32)


# ---------------------------------------------------------------------------
# Summed Area Table (SAT) Construction & Density Computation
# ---------------------------------------------------------------------------

def build_sat(tile):
    """Build a summed area table from a 2D array.

    Returns an (h+1, w+1) int64 array with a zero-padded top row and left
    column for boundary-safe queries.
    """
    h, w = tile.shape
    sat = np.zeros((h + 1, w + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(tile.astype(np.int64), axis=0), axis=1)
    return sat


def compute_density_tile(edge_tile, radii, pad):
    """Compute edge density at multiple radii for a padded tile.

    Parameters:
        edge_tile: uint8 array (padded_h, padded_w), values 0/1/255
        radii: list of int window radii
        pad: int, the padding applied around the interior tile

    Returns:
        densities: float32 array (n_radii, tile_h, tile_w) — interior only
    """
    padded_h, padded_w = edge_tile.shape
    tile_h = padded_h - 2 * pad
    tile_w = padded_w - 2 * pad

    # Build SATs
    edge_mask = (edge_tile == 1).astype(np.int64)
    valid_mask = (edge_tile != NODATA_UINT8).astype(np.int64)
    sat_edge = build_sat(edge_mask)
    sat_valid = build_sat(valid_mask)

    n_radii = len(radii)
    densities = np.empty((n_radii, tile_h, tile_w), dtype=np.float32)

    # Interior pixel coordinates (in padded image space)
    rows = np.arange(pad, pad + tile_h)
    cols = np.arange(pad, pad + tile_w)

    for ri, R in enumerate(radii):
        # Window corners: top-left (r1,c1) to bottom-right (r2,c2) inclusive
        # In SAT coordinates (1-indexed due to zero padding):
        r1 = np.clip(rows - R, 0, padded_h - 1)[:, None]      # (tile_h, 1)
        c1 = np.clip(cols - R, 0, padded_w - 1)[None, :]       # (1, tile_w)
        r2 = np.clip(rows + R, 0, padded_h - 1)[:, None] + 1   # (tile_h, 1)
        c2 = np.clip(cols + R, 0, padded_w - 1)[None, :] + 1   # (1, tile_w)

        # SAT query: sum = S[r2,c2] - S[r1,c2] - S[r2,c1] + S[r1,c1]
        edge_count = (sat_edge[r2, c2] - sat_edge[r1, c2]
                      - sat_edge[r2, c1] + sat_edge[r1, c1])
        valid_count = (sat_valid[r2, c2] - sat_valid[r1, c2]
                       - sat_valid[r2, c1] + sat_valid[r1, c1])

        # Density = edge_count / valid_count (0 where no valid pixels)
        density = np.where(valid_count > 0,
                           edge_count.astype(np.float32) / valid_count.astype(np.float32),
                           np.float32(0.0))
        densities[ri] = density

    return densities


def compute_fractal_from_density(densities, weights):
    """Compute fractal dimension from multiscale density vectors.

    Parameters:
        densities: float32 array (n_radii, h, w)
        weights: float32 array (n_radii,) — OLS slope weights for log(R)

    Returns:
        fractal: float32 array (h, w), D = 2 + slope(log(density) vs log(R))
                 NaN where density == 0 at any radius
    """
    n_radii, h, w = densities.shape

    # Valid where density > 0 at all radii
    valid = np.all(densities > 0, axis=0)

    log_density = np.full_like(densities, np.nan)
    for ri in range(n_radii):
        log_density[ri, valid] = np.log(densities[ri, valid])

    # slope = dot(weights, log_density) along radius axis
    slope = np.nansum(log_density * weights[:, np.newaxis, np.newaxis],
                      axis=0).astype(np.float32)
    fractal = np.float32(2.0) + slope
    fractal[~valid] = np.float32(np.nan)

    return fractal


# ---------------------------------------------------------------------------
# JS-Scaling Mode: per-pixel log(JS) vs log(R) regression
# ---------------------------------------------------------------------------

def _fractal_tile_js(js_bands, weights):
    """Compute fractal dimension for a tile from multi-band JS divergence.

    Parameters:
        js_bands: (n_radii, h, w) float32, JS divergence per radius
        weights: (n_radii,) float32, OLS slope weights

    Returns:
        fractal: (h, w) float32, slope of log(JS) vs log(R)
    """
    n_radii, h, w = js_bands.shape

    # Valid only where all bands are > 0 and not NaN
    valid = np.ones((h, w), dtype=bool)
    for ri in range(n_radii):
        band = js_bands[ri]
        valid &= np.isfinite(band) & (band > 0)

    log_js = np.full_like(js_bands, np.nan)
    for ri in range(n_radii):
        log_js[ri, valid] = np.log(js_bands[ri, valid])

    fractal = np.nansum(log_js * weights[:, np.newaxis, np.newaxis],
                        axis=0).astype(np.float32)
    fractal[~valid] = np.float32(np.nan)
    return fractal


def run_pass1_js(input_path, fractal_path, radii, tile_size, n_samples,
                 compress="lzw"):
    """Compute fractal map from multi-band JS divergence + reservoir sample.

    Per-pixel OLS regression of log(JS) vs log(R). No padding needed.

    Returns:
        samples: (n_collected, n_bands) float32 array of JS feature vectors
    """
    weights = precompute_slope_weights(radii)
    rng = np.random.default_rng(42)

    logger.info(f"JS-scaling mode: OLS weights for radii {radii}: {weights}")

    with rasterio.open(input_path) as src:
        img_h = src.height
        img_w = src.width
        n_bands = src.count

        if n_bands != len(radii):
            logger.error(f"Input has {n_bands} bands but {len(radii)} radii "
                         f"specified")
            sys.exit(1)

        logger.info(f"Input: {img_w} x {img_h}, {n_bands} bands")

        profile = {
            "driver": "GTiff",
            "width": img_w,
            "height": img_h,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "predictor": 3,
            "nodata": None,
            "bigtiff": "YES",
        }
        out_ds = rasterio.open(fractal_path, "w", **profile)
        out_ds.set_band_description(1, "fractal_dimension")

        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        tiles_done = 0
        t_start = time.time()

        reservoir = np.empty((n_samples, n_bands), dtype=np.float32)
        n_seen = 0
        n_collected = 0

        for tr in range(n_tile_rows):
            row_start = tr * tile_size
            row_end = min(row_start + tile_size, img_h)
            bh = row_end - row_start

            for tc in range(n_tile_cols):
                col_start = tc * tile_size
                col_end = min(col_start + tile_size, img_w)
                bw = col_end - col_start

                window = Window(col_start, row_start, bw, bh)
                js_bands = src.read(window=window).astype(np.float32)

                fractal = _fractal_tile_js(js_bands, weights)
                out_ds.write(fractal, 1, window=window)

                # Reservoir-sample raw JS feature vectors
                pixels = js_bands.reshape(n_bands, -1).T  # (bh*bw, n_bands)
                valid_mask = np.all(np.isfinite(pixels) & (pixels > 0),
                                   axis=1)
                valid_pixels = pixels[valid_mask]
                n_valid = valid_pixels.shape[0]

                for i in range(n_valid):
                    n_seen += 1
                    if n_collected < n_samples:
                        reservoir[n_collected] = valid_pixels[i]
                        n_collected += 1
                    else:
                        j = rng.integers(0, n_seen)
                        if j < n_samples:
                            reservoir[j] = valid_pixels[i]

                tiles_done += 1
                if tiles_done % 200 == 0 or tiles_done == total_tiles:
                    elapsed = time.time() - t_start
                    rate = tiles_done / elapsed if elapsed > 0 else 0
                    eta = (total_tiles - tiles_done) / rate if rate > 0 else 0
                    logger.info(
                        f"  Pass1 tile {tiles_done}/{total_tiles} "
                        f"({100*tiles_done/total_tiles:.1f}%) "
                        f"[{rate:.1f} tiles/s, ETA {eta/60:.1f}m] "
                        f"samples={n_collected:,}/{n_seen:,}")

        out_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Pass 1 complete: {elapsed/60:.1f}m, output: {fractal_path}")
        logger.info(f"Sampled {n_collected:,} feature vectors from "
                    f"{n_seen:,} valid pixels")

    return reservoir[:n_collected]


def run_pass2_js(input_path, kmeans_path, n_clusters, samples, radii,
                 tile_size, compress="lzw"):
    """K-Means classification on multi-band JS divergence feature vectors."""
    from sklearn.cluster import MiniBatchKMeans
    import joblib

    n_bands = len(radii)

    if samples.shape[0] < n_clusters:
        logger.error(f"Only {samples.shape[0]} valid samples, "
                     f"need at least {n_clusters}")
        sys.exit(1)

    logger.info(f"Pass 2: Fitting MiniBatchKMeans (k={n_clusters}, "
                f"n_samples={samples.shape[0]:,})")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(10000, samples.shape[0]),
        random_state=42,
        n_init=3,
    )
    kmeans.fit(samples)
    logger.info(f"K-Means fit in {time.time()-t0:.1f}s, "
                f"inertia={kmeans.inertia_:.2f}")

    model_path = kmeans_path.replace(".tif", "_model.joblib")
    joblib.dump(kmeans, model_path)
    logger.info(f"Model saved: {model_path}")

    logger.info("Pass 2: Predicting classes across full image")
    with rasterio.open(input_path) as src:
        img_h = src.height
        img_w = src.width

        profile = {
            "driver": "GTiff",
            "width": img_w,
            "height": img_h,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "predictor": 2,
            "nodata": NODATA_UINT8,
            "bigtiff": "YES",
        }
        out_ds = rasterio.open(kmeans_path, "w", **profile)
        out_ds.set_band_description(1, f"kmeans_k{n_clusters}")

        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        tiles_done = 0
        t_start = time.time()

        for tr in range(n_tile_rows):
            row_start = tr * tile_size
            row_end = min(row_start + tile_size, img_h)
            bh = row_end - row_start

            for tc in range(n_tile_cols):
                col_start = tc * tile_size
                col_end = min(col_start + tile_size, img_w)
                bw = col_end - col_start

                window = Window(col_start, row_start, bw, bh)
                tile = src.read(window=window).astype(np.float32)
                pixels = tile.reshape(n_bands, -1).T

                valid_mask = np.all(np.isfinite(pixels) & (pixels > 0),
                                   axis=1)
                labels = np.full(pixels.shape[0], NODATA_UINT8, dtype=np.uint8)
                if np.any(valid_mask):
                    labels[valid_mask] = kmeans.predict(
                        pixels[valid_mask]).astype(np.uint8)

                out_ds.write(labels.reshape(bh, bw), 1, window=window)

                tiles_done += 1
                if tiles_done % 200 == 0 or tiles_done == total_tiles:
                    elapsed = time.time() - t_start
                    rate = tiles_done / elapsed if elapsed > 0 else 0
                    eta = (total_tiles - tiles_done) / rate if rate > 0 else 0
                    logger.info(
                        f"  Pass2 tile {tiles_done}/{total_tiles} "
                        f"({100*tiles_done/total_tiles:.1f}%) "
                        f"[{rate:.1f} tiles/s, ETA {eta/60:.1f}m]")

        out_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Pass 2 complete: {elapsed/60:.1f}m, output: {kmeans_path}")


# ---------------------------------------------------------------------------
# Edge-Density Mode: Pass 1 — Fractal Dimension Map + Reservoir Sampling
# ---------------------------------------------------------------------------

def run_pass1(input_path, fractal_path, radii, tile_size, n_samples,
              compress="lzw"):
    """Compute fractal dimension map from binary edge map.

    Strip-based tiling with padding. Also reservoir-samples density vectors
    for K-Means training.

    Returns:
        samples: (n_collected, n_radii) float32 array of density vectors
    """
    weights = precompute_slope_weights(radii)
    pad = max(radii)
    n_radii = len(radii)
    rng = np.random.default_rng(42)

    logger.info(f"OLS weights for radii {radii}: {weights}")
    logger.info(f"Padding: {pad} pixels")

    with rasterio.open(input_path) as src:
        img_h = src.height
        img_w = src.width

        if src.count != 1:
            logger.error(f"Expected single-band edge map, got {src.count} bands")
            sys.exit(1)

        logger.info(f"Input: {img_w} x {img_h}, 1 band")

        # Create fractal output
        profile = {
            "driver": "GTiff",
            "width": img_w,
            "height": img_h,
            "count": 1,
            "dtype": "float32",
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "predictor": 3,
            "nodata": None,
            "bigtiff": "YES",
        }
        out_ds = rasterio.open(fractal_path, "w", **profile)
        out_ds.set_band_description(1, "fractal_dimension")

        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        tiles_done = 0
        t_start = time.time()

        # Reservoir sampling state
        reservoir = np.empty((n_samples, n_radii), dtype=np.float32)
        n_seen = 0
        n_collected = 0

        for strip_row in range(n_tile_rows):
            row_start = strip_row * tile_size
            row_end = min(row_start + tile_size, img_h)
            actual_tile_h = row_end - row_start

            # Padded row range for reading
            read_row_start = max(row_start - pad, 0)
            read_row_end = min(row_end + pad, img_h)
            pad_top_actual = row_start - read_row_start
            pad_bottom_actual = read_row_end - row_end

            # Read the full-width strip with vertical padding
            strip_window = Window(0, read_row_start, img_w,
                                  read_row_end - read_row_start)
            strip_data = src.read(1, window=strip_window)  # (strip_h, img_w)

            # Extra padding at image boundaries (constant=0 for binary edges)
            extra_pad_top = pad - pad_top_actual
            extra_pad_bottom = pad - pad_bottom_actual
            if extra_pad_top > 0 or extra_pad_bottom > 0:
                strip_data = np.pad(
                    strip_data,
                    ((max(extra_pad_top, 0), max(extra_pad_bottom, 0)),
                     (0, 0)),
                    mode="constant", constant_values=0
                )

            for tile_col in range(n_tile_cols):
                col_start = tile_col * tile_size
                col_end = min(col_start + tile_size, img_w)
                actual_tile_w = col_end - col_start

                # Padded column range
                read_col_start = max(col_start - pad, 0)
                read_col_end = min(col_end + pad, img_w)
                pad_left_actual = col_start - read_col_start
                pad_right_actual = read_col_end - col_end

                # Extract padded tile from strip
                tile_data = strip_data[:, read_col_start:read_col_end]

                # Extra horizontal padding at boundaries
                extra_pad_left = pad - pad_left_actual
                extra_pad_right = pad - pad_right_actual
                if extra_pad_left > 0 or extra_pad_right > 0:
                    tile_data = np.pad(
                        tile_data,
                        ((0, 0),
                         (max(extra_pad_left, 0), max(extra_pad_right, 0))),
                        mode="constant", constant_values=0
                    )

                # Compute densities (n_radii, actual_tile_h, actual_tile_w)
                densities = compute_density_tile(tile_data, radii, pad)
                # Trim to actual tile size
                densities = densities[:, :actual_tile_h, :actual_tile_w]

                # Compute fractal dimension
                fractal = compute_fractal_from_density(densities, weights)

                # Write fractal tile
                out_window = Window(col_start, row_start,
                                    actual_tile_w, actual_tile_h)
                out_ds.write(fractal, 1, window=out_window)

                # Reservoir-sample density vectors where valid
                valid = np.all(densities > 0, axis=0)
                if np.any(valid):
                    # Reshape to (n_valid, n_radii)
                    valid_densities = densities[:, valid].T  # (n_valid, n_radii)
                    n_valid = valid_densities.shape[0]

                    for i in range(n_valid):
                        n_seen += 1
                        if n_collected < n_samples:
                            reservoir[n_collected] = valid_densities[i]
                            n_collected += 1
                        else:
                            j = rng.integers(0, n_seen)
                            if j < n_samples:
                                reservoir[j] = valid_densities[i]

                tiles_done += 1
                if tiles_done % 200 == 0 or tiles_done == total_tiles:
                    elapsed = time.time() - t_start
                    rate = tiles_done / elapsed if elapsed > 0 else 0
                    eta = (total_tiles - tiles_done) / rate if rate > 0 else 0
                    logger.info(
                        f"  Pass1 tile {tiles_done}/{total_tiles} "
                        f"({100*tiles_done/total_tiles:.1f}%) "
                        f"[{rate:.1f} tiles/s, ETA {eta/60:.1f}m] "
                        f"samples={n_collected:,}/{n_seen:,}")

            # Free strip memory
            del strip_data

        out_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Pass 1 complete: {elapsed/60:.1f}m, output: {fractal_path}")
        logger.info(f"Sampled {n_collected:,} density vectors from "
                    f"{n_seen:,} valid pixels")

    return reservoir[:n_collected]


# ---------------------------------------------------------------------------
# Pass 2: K-Means Classification
# ---------------------------------------------------------------------------

def run_pass2_classify(input_path, kmeans_path, n_clusters, samples, radii,
                       tile_size, compress="lzw"):
    """Fit K-Means on sampled density vectors, then classify full image.

    Re-reads the edge map and recomputes density vectors tile-by-tile to
    predict cluster labels.
    """
    from sklearn.cluster import MiniBatchKMeans
    import joblib

    n_radii = len(radii)
    pad = max(radii)

    if samples.shape[0] < n_clusters:
        logger.error(f"Only {samples.shape[0]} valid samples, "
                     f"need at least {n_clusters}")
        sys.exit(1)

    # Fit K-Means
    logger.info(f"Pass 2: Fitting MiniBatchKMeans (k={n_clusters}, "
                f"n_samples={samples.shape[0]:,})")
    t0 = time.time()
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=min(10000, samples.shape[0]),
        random_state=42,
        n_init=3,
    )
    kmeans.fit(samples)
    logger.info(f"K-Means fit in {time.time()-t0:.1f}s, "
                f"inertia={kmeans.inertia_:.2f}")

    # Save model
    model_path = kmeans_path.replace(".tif", "_model.joblib")
    joblib.dump(kmeans, model_path)
    logger.info(f"Model saved: {model_path}")

    # Predict across full image
    logger.info("Pass 2: Predicting classes across full image")
    with rasterio.open(input_path) as src:
        img_h = src.height
        img_w = src.width

        profile = {
            "driver": "GTiff",
            "width": img_w,
            "height": img_h,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "predictor": 2,
            "nodata": NODATA_UINT8,
            "bigtiff": "YES",
        }
        out_ds = rasterio.open(kmeans_path, "w", **profile)
        out_ds.set_band_description(1, f"kmeans_k{n_clusters}")

        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        tiles_done = 0
        t_start = time.time()

        for strip_row in range(n_tile_rows):
            row_start = strip_row * tile_size
            row_end = min(row_start + tile_size, img_h)
            actual_tile_h = row_end - row_start

            read_row_start = max(row_start - pad, 0)
            read_row_end = min(row_end + pad, img_h)
            pad_top_actual = row_start - read_row_start
            pad_bottom_actual = read_row_end - row_end

            strip_window = Window(0, read_row_start, img_w,
                                  read_row_end - read_row_start)
            strip_data = src.read(1, window=strip_window)

            extra_pad_top = pad - pad_top_actual
            extra_pad_bottom = pad - pad_bottom_actual
            if extra_pad_top > 0 or extra_pad_bottom > 0:
                strip_data = np.pad(
                    strip_data,
                    ((max(extra_pad_top, 0), max(extra_pad_bottom, 0)),
                     (0, 0)),
                    mode="constant", constant_values=0
                )

            for tile_col in range(n_tile_cols):
                col_start = tile_col * tile_size
                col_end = min(col_start + tile_size, img_w)
                actual_tile_w = col_end - col_start

                read_col_start = max(col_start - pad, 0)
                read_col_end = min(col_end + pad, img_w)
                pad_left_actual = col_start - read_col_start
                pad_right_actual = read_col_end - col_end

                tile_data = strip_data[:, read_col_start:read_col_end]

                extra_pad_left = pad - pad_left_actual
                extra_pad_right = pad - pad_right_actual
                if extra_pad_left > 0 or extra_pad_right > 0:
                    tile_data = np.pad(
                        tile_data,
                        ((0, 0),
                         (max(extra_pad_left, 0), max(extra_pad_right, 0))),
                        mode="constant", constant_values=0
                    )

                densities = compute_density_tile(tile_data, radii, pad)
                densities = densities[:, :actual_tile_h, :actual_tile_w]

                # Valid where density > 0 at all radii
                valid = np.all(densities > 0, axis=0)
                labels = np.full((actual_tile_h, actual_tile_w),
                                 NODATA_UINT8, dtype=np.uint8)

                if np.any(valid):
                    # Reshape valid pixels to (n_valid, n_radii)
                    valid_densities = densities[:, valid].T
                    labels[valid] = kmeans.predict(
                        valid_densities).astype(np.uint8)

                out_window = Window(col_start, row_start,
                                    actual_tile_w, actual_tile_h)
                out_ds.write(labels, 1, window=out_window)

                tiles_done += 1
                if tiles_done % 200 == 0 or tiles_done == total_tiles:
                    elapsed = time.time() - t_start
                    rate = tiles_done / elapsed if elapsed > 0 else 0
                    eta = (total_tiles - tiles_done) / rate if rate > 0 else 0
                    logger.info(
                        f"  Pass2 tile {tiles_done}/{total_tiles} "
                        f"({100*tiles_done/total_tiles:.1f}%) "
                        f"[{rate:.1f} tiles/s, ETA {eta/60:.1f}m]")

            del strip_data

        out_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Pass 2 complete: {elapsed/60:.1f}m, output: {kmeans_path}")


# ---------------------------------------------------------------------------
# Fractal-Binned Classification
# ---------------------------------------------------------------------------

def run_fractal_binned(fractal_path, output_path, bin_edges, tile_size,
                       compress="lzw"):
    """Classify fractal dimension into discrete bins.

    Parameters:
        fractal_path: path to single-band fractal dimension GeoTIFF
        output_path: output uint8 classification GeoTIFF
        bin_edges: list of bin boundary values (n edges -> n+1 classes)
        tile_size: processing tile size
    """
    n_classes = len(bin_edges) + 1
    logger.info(f"Fractal binning: {len(bin_edges)} edges -> {n_classes} classes")
    logger.info(f"Bin edges: {bin_edges}")

    with rasterio.open(fractal_path) as src:
        img_h = src.height
        img_w = src.width

        profile = {
            "driver": "GTiff",
            "width": img_w,
            "height": img_h,
            "count": 1,
            "dtype": "uint8",
            "crs": src.crs,
            "transform": src.transform,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": compress,
            "predictor": 2,
            "nodata": NODATA_UINT8,
            "bigtiff": "YES",
        }
        out_ds = rasterio.open(output_path, "w", **profile)
        out_ds.set_band_description(1, "fractal_class")

        edges_arr = np.array(bin_edges, dtype=np.float64)

        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        tiles_done = 0
        t_start = time.time()

        for tr in range(n_tile_rows):
            row_start = tr * tile_size
            row_end = min(row_start + tile_size, img_h)
            bh = row_end - row_start

            for tc in range(n_tile_cols):
                col_start = tc * tile_size
                col_end = min(col_start + tile_size, img_w)
                bw = col_end - col_start

                window = Window(col_start, row_start, bw, bh)
                fractal = src.read(1, window=window)

                valid = np.isfinite(fractal)
                classes = np.full((bh, bw), NODATA_UINT8, dtype=np.uint8)

                if np.any(valid):
                    binned = np.digitize(fractal[valid],
                                         edges_arr).astype(np.uint8)
                    binned = np.clip(binned, 0, n_classes - 1)
                    classes[valid] = binned

                out_ds.write(classes, 1, window=window)

                tiles_done += 1
                if tiles_done % 500 == 0 or tiles_done == total_tiles:
                    elapsed = time.time() - t_start
                    rate = tiles_done / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  Fractal-bin tile {tiles_done}/{total_tiles} "
                        f"({100*tiles_done/total_tiles:.1f}%)")

        out_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Fractal binning complete: {elapsed:.1f}s, "
                    f"output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Multiscale Fractal Analysis & Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Edge-density mode (default): fractal from binary edge map
  python js_fractal_classify.py edge_map.tif -o output_prefix

  # JS-scaling mode: fractal from multi-band JS divergence
  python js_fractal_classify.py js_5band.tif -o out --mode js-scaling
"""
    )
    parser.add_argument("input",
                        help="Input GeoTIFF: binary edge map (edge-density) "
                             "or multi-band JS divergence (js-scaling)")
    parser.add_argument("-o", "--output-prefix", required=True,
                        help="Output prefix for result files")
    parser.add_argument("--mode", choices=["edge-density", "js-scaling"],
                        default="edge-density",
                        help="Analysis mode (default: edge-density)")
    parser.add_argument("--radii", type=int, nargs="+", default=None,
                        help="Radii (default: mode-dependent)")
    parser.add_argument("--n-clusters", type=int, default=DEFAULT_N_CLUSTERS,
                        help=f"Number of K-Means clusters "
                             f"(default: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Number of pixels to sample for K-Means "
                             f"(default: {DEFAULT_N_SAMPLES:,})")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help=f"Tile size for processing "
                             f"(default: {DEFAULT_TILE_SIZE})")
    parser.add_argument("--fractal-bin-edges", type=float, nargs="+",
                        default=None,
                        help="Bin edges for fractal classification "
                             "(default: mode-dependent)")
    parser.add_argument("--skip-kmeans", action="store_true",
                        help="Skip K-Means classification")
    parser.add_argument("--skip-fractal-binned", action="store_true",
                        help="Skip fractal-binned classification")
    parser.add_argument("--compress", default="lzw",
                        choices=["lzw", "deflate", "zstd", "none"],
                        help="GeoTIFF compression (default: lzw)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Mode-dependent defaults
    mode = args.mode
    if args.radii is None:
        radii = (EDGE_DENSITY_RADII if mode == "edge-density"
                 else JS_SCALING_RADII)
    else:
        radii = args.radii
    if args.fractal_bin_edges is None:
        bin_edges = (EDGE_DENSITY_BIN_EDGES if mode == "edge-density"
                     else JS_SCALING_BIN_EDGES)
    else:
        bin_edges = args.fractal_bin_edges

    output_prefix = args.output_prefix
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    compress = args.compress if args.compress != "none" else None

    fractal_path = f"{output_prefix}_fractal.tif"
    kmeans_path = f"{output_prefix}_kmeans_k{args.n_clusters}.tif"
    fractal_class_path = f"{output_prefix}_fractal_classes.tif"

    logger.info(f"Mode: {mode}, radii: {radii}, bin_edges: {bin_edges}")

    # Pass 1: Fractal dimension + reservoir sampling
    logger.info("=" * 60)
    logger.info("Pass 1: Computing fractal dimension map")
    logger.info("=" * 60)
    if mode == "edge-density":
        samples = run_pass1(str(input_path), fractal_path, radii,
                            args.tile_size, args.n_samples, compress=compress)
    else:
        samples = run_pass1_js(str(input_path), fractal_path, radii,
                               args.tile_size, args.n_samples,
                               compress=compress)

    # Pass 2: K-Means classification
    if not args.skip_kmeans:
        logger.info("=" * 60)
        logger.info("Pass 2: K-Means classification")
        logger.info("=" * 60)
        if mode == "edge-density":
            run_pass2_classify(str(input_path), kmeans_path, args.n_clusters,
                               samples, radii, args.tile_size,
                               compress=compress)
        else:
            run_pass2_js(str(input_path), kmeans_path, args.n_clusters,
                         samples, radii, args.tile_size, compress=compress)

    # Fractal-binned classification
    if not args.skip_fractal_binned:
        logger.info("=" * 60)
        logger.info("Fractal-binned classification")
        logger.info("=" * 60)
        run_fractal_binned(fractal_path, fractal_class_path,
                           bin_edges, args.tile_size, compress=compress)

    logger.info("Done.")


if __name__ == "__main__":
    main()
