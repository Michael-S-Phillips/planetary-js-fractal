#!/usr/bin/env python3
"""
JS Divergence Edge Detection for Planetary Rasters

Detects geological edges at configurable spatial scales using Jensen-Shannon
divergence between opposing rectangular half-windows at multiple radii and
directions, accelerated by integral histograms.

Supports arbitrary single-band rasters (any dtype, any nodata convention).
Non-uint8 inputs are automatically rescaled to uint8 [1-255] for histogram
binning, with nodata mapped to 0.  The rescaling preserves relative ordering
and does not affect JS divergence results.

Outputs either a multi-band GeoTIFF (one band per radius, default) or a single-band
averaged GeoTIFF (--single-band). Multi-band output enables fractal dimension
analysis via log(JS) vs log(R) scaling.

Inspired by the JS fractal analysis approach of Phillips/Loke/Chisholm.

Author: Michael S. Phillips
Date: 2026-02-12
"""

import argparse
import json
import logging
import sys
import time
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from scipy.ndimage import median_filter

# Try to import numba; fall back to pure numpy if unavailable
try:
    from numba import njit, prange, types
    from numba.typed import Dict as NumbaDict
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

logger = logging.getLogger("js_edge_detect")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RADII = [5, 7, 10, 14, 20]
DEFAULT_BINS = 32
DEFAULT_TILE_SIZE = 1024
MIN_VALID_PIXELS = 16  # minimum pixels in a half-window to compute JS
OTSU_HIST_BINS = 10000  # bins for global JS histogram used in Otsu
_NODATA_UINT8 = 0   # internal uint8 marker: 0 = nodata after rescaling
NODATA_JS = np.float32(np.nan)
_NAN32 = np.float32(np.nan)  # numba-compatible NaN constant
NODATA_EDGE = 255


# ---------------------------------------------------------------------------
# Input rescaling helpers
# ---------------------------------------------------------------------------

def _prescan_range(src, band, nodata_val):
    """Pre-scan raster to find valid data [min, max] for rescaling to uint8.

    Reads full raster in blocks; fast compared to the JS computation itself.
    """
    logger.info("Pre-scanning raster to determine data range...")
    t0 = time.time()
    data_min = np.inf
    data_max = -np.inf
    block_size = 4096

    for row_off in range(0, src.height, block_size):
        h = min(block_size, src.height - row_off)
        for col_off in range(0, src.width, block_size):
            w = min(block_size, src.width - col_off)
            window = Window(col_off, row_off, w, h)
            data = src.read(band, window=window)

            # Build valid mask (exclude nodata + non-finite)
            if np.issubdtype(data.dtype, np.floating):
                mask = np.isfinite(data)
            else:
                mask = np.ones(data.shape, dtype=bool)

            if nodata_val is not None:
                if np.isfinite(nodata_val):
                    mask &= (data != nodata_val)
                # else nodata is NaN — already excluded by isfinite above

            valid = data[mask]
            if valid.size > 0:
                data_min = min(data_min, float(valid.min()))
                data_max = max(data_max, float(valid.max()))

    elapsed = time.time() - t0
    logger.info(f"Data range: [{data_min}, {data_max}] (scanned in {elapsed:.1f}s)")
    return float(data_min), float(data_max)


def _rescale_to_uint8(tile, nodata_val, data_min, data_max):
    """Rescale a tile from native dtype to uint8 [1, 255] with nodata -> 0.

    Parameters:
        tile: 2-D array in native dtype
        nodata_val: nodata value (float, NaN, or None)
        data_min, data_max: global data range from _prescan_range

    Returns:
        uint8 array with 0 = nodata, 1-255 = valid rescaled data
    """
    out = np.zeros(tile.shape, dtype=np.uint8)

    # Build valid mask
    if np.issubdtype(tile.dtype, np.floating):
        valid = np.isfinite(tile)
    else:
        valid = np.ones(tile.shape, dtype=bool)

    if nodata_val is not None:
        if np.isfinite(nodata_val):
            valid &= (tile != nodata_val)

    if data_max > data_min:
        scaled = ((tile[valid].astype(np.float64) - data_min)
                  / (data_max - data_min) * 254.0 + 1.0)
        out[valid] = np.clip(np.round(scaled), 1, 255).astype(np.uint8)
    elif np.isfinite(data_min):
        out[valid] = 128  # constant data: arbitrary middle value

    return out


def _local_range(tile, nodata_val):
    """Compute [min, max] of valid pixels in a tile for local rescaling."""
    if np.issubdtype(tile.dtype, np.floating):
        mask = np.isfinite(tile)
    else:
        mask = np.ones(tile.shape, dtype=bool)
    if nodata_val is not None and np.isfinite(float(nodata_val)):
        mask &= (tile != nodata_val)
    valid = tile[mask]
    if valid.size == 0:
        return 0.0, 0.0
    return float(valid.min()), float(valid.max())


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

if HAS_NUMBA:

    @njit(cache=True)
    def _bin_tile(tile_uint8, n_bins):
        """Bin uint8 tile into n_bins. NoData (0) -> -1, values 1-255 -> 0..(n_bins-1)."""
        rows, cols = tile_uint8.shape
        out = np.empty((rows, cols), dtype=np.int8)
        bin_width = 256.0 / n_bins
        for r in range(rows):
            for c in range(cols):
                v = tile_uint8[r, c]
                if v == 0:
                    out[r, c] = -1
                else:
                    b = int(v / bin_width)
                    if b >= n_bins:
                        b = n_bins - 1
                    out[r, c] = b
        return out

    @njit(cache=True)
    def _build_integral_histogram(binned, n_bins):
        """Build integral histogram: shape (n_bins, rows+1, cols+1), dtype int32.

        ihist[b, r, c] = count of bin b in binned[0:r, 0:c].
        NoData pixels (bin == -1) are excluded from all bins.
        """
        rows, cols = binned.shape
        ihist = np.zeros((n_bins, rows + 1, cols + 1), dtype=np.int32)
        for b in range(n_bins):
            for r in range(rows):
                row_sum = np.int32(0)
                for c in range(cols):
                    if binned[r, c] == b:
                        row_sum += 1
                    ihist[b, r + 1, c + 1] = ihist[b, r, c + 1] + row_sum
        return ihist

    @njit(cache=True, inline="always")
    def _rect_hist(ihist, n_bins, r0, c0, r1, c1):
        """Query integral histogram for rectangle [r0, r1) x [c0, c1).
        Returns histogram array of length n_bins and total count."""
        hist = np.empty(n_bins, dtype=np.float64)
        total = np.int32(0)
        for b in range(n_bins):
            val = (ihist[b, r1, c1] - ihist[b, r0, c1]
                   - ihist[b, r1, c0] + ihist[b, r0, c0])
            hist[b] = val
            total += val
        return hist, total

    @njit(cache=True, inline="always")
    def _entropy_from_hist(hist, total):
        """Shannon entropy in bits from unnormalized histogram with known total."""
        if total <= 0:
            return 0.0
        inv_total = 1.0 / total
        ent = 0.0
        for i in range(hist.shape[0]):
            if hist[i] > 0:
                p = hist[i] * inv_total
                ent -= p * (math.log(p) / math.log(2.0))
        return ent

    @njit(cache=True, inline="always")
    def _js_div(hist_a, total_a, hist_b, total_b, n_bins):
        """JS divergence between two unnormalized histograms."""
        if total_a < MIN_VALID_PIXELS or total_b < MIN_VALID_PIXELS:
            return 0.0
        # Compute mixture histogram (unnormalized by sum of counts)
        m = np.empty(n_bins, dtype=np.float64)
        total_m = total_a + total_b
        for i in range(n_bins):
            m[i] = hist_a[i] + hist_b[i]
        h_m = _entropy_from_hist(m, total_m)
        h_a = _entropy_from_hist(hist_a, total_a)
        h_b = _entropy_from_hist(hist_b, total_b)
        # JS = H(M) - 0.5*(H(A) + H(B))  where M = 0.5*(P+Q)
        # With unnormalized counts: H(M) is entropy of mixture distribution
        # But we need the weighted version: JS = H(w_a*P + w_b*Q) - w_a*H(P) - w_b*H(Q)
        # where w_a = total_a/(total_a+total_b), w_b = total_b/(total_a+total_b)
        w_a = total_a / total_m
        w_b = total_b / total_m
        js = h_m - w_a * h_a - w_b * h_b
        if js < 0.0:
            js = 0.0
        if js > 1.0:
            js = 1.0
        return js

    @njit(parallel=True, cache=True)
    def _compute_js_tile(ihist, n_bins, pad, tile_h, tile_w, radii, padded_h, padded_w, nodata_mask):
        """Compute JS divergence for all pixels in a tile.

        Parameters:
            ihist: integral histogram (n_bins, padded_h+1, padded_w+1)
            n_bins: number of histogram bins
            pad: padding size (max_radius + 1)
            tile_h, tile_w: output tile dimensions
            radii: array of radii to evaluate
            padded_h, padded_w: padded tile dimensions
            nodata_mask: boolean (padded_h, padded_w), True where center is nodata

        Returns:
            js_out: (n_radii, tile_h, tile_w) float32 array of per-radius JS divergence
        """
        n_radii = radii.shape[0]
        js_out = np.zeros((n_radii, tile_h, tile_w), dtype=np.float32)

        for row in prange(tile_h):
            for col in range(tile_w):
                pr = row + pad  # position in padded array
                pc = col + pad

                # Skip nodata center pixels
                if nodata_mask[pr, pc]:
                    for ri in range(n_radii):
                        js_out[ri, row, col] = _NAN32
                    continue

                for ri in range(n_radii):
                    R = radii[ri]
                    max_js = 0.0

                    # --- Direction 1: N-S (top R rows vs bottom R rows, width 2R+1) ---
                    # Top half: rows [pr-R, pr), cols [pc-R, pc+R+1)
                    r0_a = max(pr - R, 0)
                    r1_a = pr
                    c0_a = max(pc - R, 0)
                    c1_a = min(pc + R + 1, padded_w)
                    # Bottom half: rows [pr+1, pr+R+1), cols [pc-R, pc+R+1)
                    r0_b = pr + 1
                    r1_b = min(pr + R + 1, padded_h)
                    c0_b = c0_a
                    c1_b = c1_a

                    if r1_a > r0_a and r1_b > r0_b and c1_a > c0_a:
                        ha, ta = _rect_hist(ihist, n_bins, r0_a, c0_a, r1_a, c1_a)
                        hb, tb = _rect_hist(ihist, n_bins, r0_b, c0_b, r1_b, c1_b)
                        js_val = _js_div(ha, ta, hb, tb, n_bins)
                        if js_val > max_js:
                            max_js = js_val

                    # --- Direction 2: E-W (left R cols vs right R cols, height 2R+1) ---
                    # Left half: rows [pr-R, pr+R+1), cols [pc-R, pc)
                    r0_a = max(pr - R, 0)
                    r1_a = min(pr + R + 1, padded_h)
                    c0_a = max(pc - R, 0)
                    c1_a = pc
                    # Right half: rows [pr-R, pr+R+1), cols [pc+1, pc+R+1)
                    r0_b = r0_a
                    r1_b = r1_a
                    c0_b = pc + 1
                    c1_b = min(pc + R + 1, padded_w)

                    if c1_a > c0_a and c1_b > c0_b and r1_a > r0_a:
                        ha, ta = _rect_hist(ihist, n_bins, r0_a, c0_a, r1_a, c1_a)
                        hb, tb = _rect_hist(ihist, n_bins, r0_b, c0_b, r1_b, c1_b)
                        js_val = _js_div(ha, ta, hb, tb, n_bins)
                        if js_val > max_js:
                            max_js = js_val

                    # --- Direction 3: NE-SW (upper-right vs lower-left, each (R+1)x(R+1)) ---
                    # Upper-right: rows [pr-R, pr+1), cols [pc, pc+R+1)
                    r0_a = max(pr - R, 0)
                    r1_a = pr + 1
                    c0_a = pc
                    c1_a = min(pc + R + 1, padded_w)
                    # Lower-left: rows [pr, pr+R+1), cols [pc-R, pc+1)
                    r0_b = pr
                    r1_b = min(pr + R + 1, padded_h)
                    c0_b = max(pc - R, 0)
                    c1_b = pc + 1

                    if r1_a > r0_a and c1_a > c0_a and r1_b > r0_b and c1_b > c0_b:
                        ha, ta = _rect_hist(ihist, n_bins, r0_a, c0_a, r1_a, c1_a)
                        hb, tb = _rect_hist(ihist, n_bins, r0_b, c0_b, r1_b, c1_b)
                        js_val = _js_div(ha, ta, hb, tb, n_bins)
                        if js_val > max_js:
                            max_js = js_val

                    # --- Direction 4: NW-SE (upper-left vs lower-right, each (R+1)x(R+1)) ---
                    # Upper-left: rows [pr-R, pr+1), cols [pc-R, pc+1)
                    r0_a = max(pr - R, 0)
                    r1_a = pr + 1
                    c0_a = max(pc - R, 0)
                    c1_a = pc + 1
                    # Lower-right: rows [pr, pr+R+1), cols [pc, pc+R+1)
                    r0_b = pr
                    r1_b = min(pr + R + 1, padded_h)
                    c0_b = pc
                    c1_b = min(pc + R + 1, padded_w)

                    if r1_a > r0_a and c1_a > c0_a and r1_b > r0_b and c1_b > c0_b:
                        ha, ta = _rect_hist(ihist, n_bins, r0_a, c0_a, r1_a, c1_a)
                        hb, tb = _rect_hist(ihist, n_bins, r0_b, c0_b, r1_b, c1_b)
                        js_val = _js_div(ha, ta, hb, tb, n_bins)
                        if js_val > max_js:
                            max_js = js_val

                    js_out[ri, row, col] = np.float32(max_js)

        return js_out


# ---------------------------------------------------------------------------
# Pure numpy fallback (no numba)
# ---------------------------------------------------------------------------

def _bin_tile_numpy(tile_uint8, n_bins):
    """Bin uint8 tile. NoData (0) -> -1, values 1-255 -> bin index."""
    out = np.full(tile_uint8.shape, -1, dtype=np.int8)
    valid = tile_uint8 > 0
    bin_width = 256.0 / n_bins
    out[valid] = np.clip((tile_uint8[valid].astype(np.float32) / bin_width).astype(np.int8),
                         0, n_bins - 1)
    return out


def _build_integral_histogram_numpy(binned, n_bins):
    """Build integral histogram using numpy cumulative sums."""
    rows, cols = binned.shape
    ihist = np.zeros((n_bins, rows + 1, cols + 1), dtype=np.int32)
    for b in range(n_bins):
        layer = (binned == b).astype(np.int32)
        ihist[b, 1:, 1:] = np.cumsum(np.cumsum(layer, axis=0), axis=1)
    return ihist


def _rect_hist_numpy(ihist, n_bins, r0, c0, r1, c1):
    """Query integral histogram for a rectangle."""
    hist = np.empty(n_bins, dtype=np.float64)
    for b in range(n_bins):
        hist[b] = (ihist[b, r1, c1] - ihist[b, r0, c1]
                   - ihist[b, r1, c0] + ihist[b, r0, c0])
    total = int(hist.sum())
    return hist, total


def _entropy_numpy(hist, total):
    """Shannon entropy from unnormalized histogram."""
    if total <= 0:
        return 0.0
    p = hist[hist > 0] / total
    return -np.sum(p * np.log2(p))


def _js_div_numpy(hist_a, total_a, hist_b, total_b, n_bins):
    """JS divergence between two unnormalized histograms (numpy)."""
    if total_a < MIN_VALID_PIXELS or total_b < MIN_VALID_PIXELS:
        return 0.0
    total_m = total_a + total_b
    m = hist_a + hist_b
    h_m = _entropy_numpy(m, total_m)
    h_a = _entropy_numpy(hist_a, total_a)
    h_b = _entropy_numpy(hist_b, total_b)
    w_a = total_a / total_m
    w_b = total_b / total_m
    js = h_m - w_a * h_a - w_b * h_b
    return np.clip(js, 0.0, 1.0)


def _compute_js_tile_numpy(ihist, n_bins, pad, tile_h, tile_w, radii, padded_h, padded_w, nodata_mask):
    """Pure numpy fallback for JS computation (much slower).

    Returns:
        js_out: (n_radii, tile_h, tile_w) float32 array of per-radius JS divergence
    """
    n_radii = len(radii)
    js_out = np.zeros((n_radii, tile_h, tile_w), dtype=np.float32)

    for row in range(tile_h):
        for col in range(tile_w):
            pr = row + pad
            pc = col + pad

            if nodata_mask[pr, pc]:
                for ri in range(n_radii):
                    js_out[ri, row, col] = _NAN32
                continue

            for ri, R in enumerate(radii):
                max_js = 0.0

                # Direction 1: N-S
                r0a, r1a = max(pr - R, 0), pr
                c0a, c1a = max(pc - R, 0), min(pc + R + 1, padded_w)
                r0b, r1b = pr + 1, min(pr + R + 1, padded_h)
                if r1a > r0a and r1b > r0b and c1a > c0a:
                    ha, ta = _rect_hist_numpy(ihist, n_bins, r0a, c0a, r1a, c1a)
                    hb, tb = _rect_hist_numpy(ihist, n_bins, r0b, c0a, r1b, c1a)
                    js_val = _js_div_numpy(ha, ta, hb, tb, n_bins)
                    max_js = max(max_js, js_val)

                # Direction 2: E-W
                r0a, r1a = max(pr - R, 0), min(pr + R + 1, padded_h)
                c0a, c1a = max(pc - R, 0), pc
                c0b, c1b = pc + 1, min(pc + R + 1, padded_w)
                if c1a > c0a and c1b > c0b and r1a > r0a:
                    ha, ta = _rect_hist_numpy(ihist, n_bins, r0a, c0a, r1a, c1a)
                    hb, tb = _rect_hist_numpy(ihist, n_bins, r0a, c0b, r1a, c1b)
                    js_val = _js_div_numpy(ha, ta, hb, tb, n_bins)
                    max_js = max(max_js, js_val)

                # Direction 3: NE-SW
                r0a, r1a = max(pr - R, 0), pr + 1
                c0a, c1a = pc, min(pc + R + 1, padded_w)
                r0b, r1b = pr, min(pr + R + 1, padded_h)
                c0b, c1b = max(pc - R, 0), pc + 1
                if r1a > r0a and c1a > c0a and r1b > r0b and c1b > c0b:
                    ha, ta = _rect_hist_numpy(ihist, n_bins, r0a, c0a, r1a, c1a)
                    hb, tb = _rect_hist_numpy(ihist, n_bins, r0b, c0b, r1b, c1b)
                    js_val = _js_div_numpy(ha, ta, hb, tb, n_bins)
                    max_js = max(max_js, js_val)

                # Direction 4: NW-SE
                r0a, r1a = max(pr - R, 0), pr + 1
                c0a, c1a = max(pc - R, 0), pc + 1
                r0b, r1b = pr, min(pr + R + 1, padded_h)
                c0b, c1b = pc, min(pc + R + 1, padded_w)
                if r1a > r0a and c1a > c0a and r1b > r0b and c1b > c0b:
                    ha, ta = _rect_hist_numpy(ihist, n_bins, r0a, c0a, r1a, c1a)
                    hb, tb = _rect_hist_numpy(ihist, n_bins, r0b, c0b, r1b, c1b)
                    js_val = _js_div_numpy(ha, ta, hb, tb, n_bins)
                    max_js = max(max_js, js_val)

                js_out[ri, row, col] = np.float32(max_js)

    return js_out


# ---------------------------------------------------------------------------
# Smoothing (directional median, matching reference smooth_divergence_matrix)
# ---------------------------------------------------------------------------

def smooth_divergence(js_tile):
    """Apply directional median smoothing: 4 oriented 3-pixel medians, take max with center.

    Handles both 2D (h, w) and 3D (n_radii, h, w) input. For 3D input, each
    band is smoothed independently.

    Directions: horizontal, vertical, diagonal, anti-diagonal.
    For each pixel, compute the median of the 3 pixels along each direction,
    then take the max of those 4 medians and the center pixel value.
    This preserves strong edges while smoothing noise.
    """
    if js_tile.size == 0:
        return js_tile

    # Handle 3D input: smooth each band independently
    if js_tile.ndim == 3:
        out = np.empty_like(js_tile)
        for i in range(js_tile.shape[0]):
            out[i] = smooth_divergence(js_tile[i])
        return out

    # 2D path
    # Replace NaN with 0 for filtering, then restore
    nan_mask = np.isnan(js_tile)
    work = np.where(nan_mask, 0.0, js_tile).astype(np.float32)

    # Pad with reflect for border handling
    padded = np.pad(work, 1, mode="reflect")
    h, w = work.shape

    # Extract directional 3-pixel medians using shifted arrays
    # Horizontal: (r, c-1), (r, c), (r, c+1)
    horiz = np.median(np.stack([padded[1:-1, 0:-2], padded[1:-1, 1:-1], padded[1:-1, 2:]], axis=0), axis=0)
    # Vertical: (r-1, c), (r, c), (r+1, c)
    vert = np.median(np.stack([padded[0:-2, 1:-1], padded[1:-1, 1:-1], padded[2:, 1:-1]], axis=0), axis=0)
    # Diagonal (NW-SE): (r-1, c-1), (r, c), (r+1, c+1)
    diag = np.median(np.stack([padded[0:-2, 0:-2], padded[1:-1, 1:-1], padded[2:, 2:]], axis=0), axis=0)
    # Anti-diagonal (NE-SW): (r-1, c+1), (r, c), (r+1, c-1)
    adiag = np.median(np.stack([padded[0:-2, 2:], padded[1:-1, 1:-1], padded[2:, 0:-2]], axis=0), axis=0)

    # Max of center and all directional medians
    result = np.maximum.reduce([work, horiz, vert, diag, adiag])

    # Restore NaN
    result[nan_mask] = _NAN32
    return result


# ---------------------------------------------------------------------------
# Otsu thresholding from accumulated histogram
# ---------------------------------------------------------------------------

def otsu_threshold(hist_counts, bin_edges):
    """Compute Otsu threshold from a precomputed histogram.

    Parameters:
        hist_counts: array of counts per bin
        bin_edges: array of bin edges (len = len(hist_counts) + 1)

    Returns:
        threshold: optimal threshold value
    """
    # Use bin centers as values
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalize to probability
    total = hist_counts.sum()
    if total == 0:
        return 0.0
    p = hist_counts.astype(np.float64) / total

    # Cumulative sums
    omega = np.cumsum(p)
    mu = np.cumsum(p * bin_centers)
    mu_total = mu[-1]

    # Between-class variance
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b_sq = (mu_total * omega - mu) ** 2 / (omega * (1 - omega))

    sigma_b_sq = np.nan_to_num(sigma_b_sq, nan=0.0)

    # Find threshold that maximizes between-class variance
    idx = np.argmax(sigma_b_sq)
    return bin_centers[idx]


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def create_output_tif(path, width, height, transform, crs, dtype, nodata,
                      compress="lzw", count=1, descriptions=None):
    """Create an output GeoTIFF with tiled layout.

    Parameters:
        count: number of bands (default 1)
        descriptions: optional list of band description strings
    """
    predictor = 3 if np.issubdtype(np.dtype(dtype), np.floating) else 2
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "compress": compress,
        "predictor": predictor,
        "nodata": nodata,
        "bigtiff": "YES",
    }
    ds = rasterio.open(path, "w", **profile)
    if descriptions:
        for i, desc in enumerate(descriptions):
            ds.set_band_description(i + 1, desc)
    return ds


def process_tile(tile_data, radii, n_bins, pad, use_numba):
    """Process a single tile: bin -> integral histogram -> JS divergence -> smooth.

    Parameters:
        tile_data: uint8 array (padded_h, padded_w) including overlap
        radii: list of radii
        n_bins: number of histogram bins
        pad: padding size
        use_numba: whether to use numba acceleration

    Returns:
        js_smooth: float32 array (n_radii, tile_h, tile_w) of smoothed JS divergence
    """
    n_radii = len(radii)
    padded_h, padded_w = tile_data.shape
    tile_h = padded_h - 2 * pad
    tile_w = padded_w - 2 * pad

    if tile_h <= 0 or tile_w <= 0:
        th = max(tile_h, 0)
        tw = max(tile_w, 0)
        return np.full((n_radii, th, tw), _NAN32, dtype=np.float32)

    # Early exit: if all center pixels are nodata, skip expensive computation
    if not np.any(tile_data[pad:pad + tile_h, pad:pad + tile_w]):
        return np.full((n_radii, tile_h, tile_w), _NAN32, dtype=np.float32)

    # Create nodata mask for padded tile
    nodata_mask = (tile_data == _NODATA_UINT8)

    if use_numba and HAS_NUMBA:
        binned = _bin_tile(tile_data, n_bins)
        ihist = _build_integral_histogram(binned, n_bins)
        radii_arr = np.array(radii, dtype=np.int32)
        js_raw = _compute_js_tile(ihist, n_bins, pad, tile_h, tile_w,
                                  radii_arr, padded_h, padded_w, nodata_mask)
    else:
        binned = _bin_tile_numpy(tile_data, n_bins)
        ihist = _build_integral_histogram_numpy(binned, n_bins)
        js_raw = _compute_js_tile_numpy(ihist, n_bins, pad, tile_h, tile_w,
                                        radii, padded_h, padded_w, nodata_mask)

    # Apply directional median smoothing (handles 3D)
    js_smooth = smooth_divergence(js_raw)
    return js_smooth


def _save_progress(progress_path, strip_row, hist_counts):
    """Save progress checkpoint after completing a strip."""
    data = {
        "completed_strip": strip_row,
        "hist_counts": hist_counts.tolist(),
    }
    # Write to temp then rename for atomicity
    tmp_path = str(progress_path) + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f)
    Path(tmp_path).rename(progress_path)


def _load_progress(progress_path):
    """Load progress from checkpoint file.

    Returns:
        (completed_strip, hist_counts) or (None, None) if no file exists
    """
    if not Path(progress_path).exists():
        return None, None
    with open(progress_path) as f:
        data = json.load(f)
    return data["completed_strip"], np.array(data["hist_counts"], dtype=np.int64)


def run_phase1(input_path, output_js_path, radii, n_bins, tile_size, use_numba,
               resume_row=0, compress="lzw", single_band=False,
               band=1, nodata="auto", local_rescale=False):
    """Phase 1: Compute JS divergence map.

    Parameters:
        single_band: if True, average radii into 1 band (legacy behavior)
        band: which raster band to read (default 1)
        nodata: "auto" to read from metadata, None for no nodata,
                or a float value to use as nodata

    Returns:
        global_hist_counts: accumulated histogram of JS values for Otsu
        global_hist_edges: bin edges for the histogram
    """
    pad = max(radii) + 1  # padding needed on each side
    n_radii = len(radii)
    n_bands = 1 if single_band else n_radii
    band_descriptions = None
    if not single_band:
        band_descriptions = [f"JS_R{r}" for r in radii]

    progress_path = f"{output_js_path}.progress.json"

    with rasterio.open(input_path) as src:
        img_h = src.height
        img_w = src.width
        crs = src.crs
        transform = src.transform

        # Resolve nodata value
        if nodata == "auto":
            nodata_val = src.nodata
        else:
            nodata_val = nodata  # None = no nodata; float = specific value

        # Determine whether input needs rescaling to uint8
        src_dtype = src.dtypes[band - 1]
        needs_rescale = not (src_dtype == "uint8"
                             and (nodata_val is None or nodata_val == 0))

        data_min = data_max = None
        if needs_rescale and not local_rescale:
            data_min, data_max = _prescan_range(src, band, nodata_val)
            if not np.isfinite(data_min) or not np.isfinite(data_max):
                logger.error("No valid data found in raster — cannot determine range")
                sys.exit(1)

        logger.info(f"Input: {img_w} x {img_h}, dtype={src_dtype}, band={band}, "
                     f"CRS={crs}, NoData={nodata_val}")
        if needs_rescale and not local_rescale:
            logger.info(f"Rescaling [{data_min}, {data_max}] -> uint8 [1, 255]")
        elif needs_rescale and local_rescale:
            logger.info("Local rescaling enabled: each tile normalized independently to uint8 [1, 255]")
        logger.info(f"Radii: {radii}, Bins: {n_bins}, Tile: {tile_size}, Pad: {pad}")
        logger.info(f"Output bands: {n_bands} ({'single-band averaged' if single_band else 'per-radius'})")
        logger.info(f"Using {'numba' if (use_numba and HAS_NUMBA) else 'numpy (slow!)'} backend")

        # Calculate grid dimensions
        n_tile_rows = math.ceil(img_h / tile_size)
        n_tile_cols = math.ceil(img_w / tile_size)
        total_tiles = n_tile_rows * n_tile_cols
        logger.info(f"Grid: {n_tile_cols} x {n_tile_rows} = {total_tiles} tiles")

        # Global histogram accumulator for Otsu
        global_hist_counts = np.zeros(OTSU_HIST_BINS, dtype=np.int64)
        global_hist_edges = np.linspace(0.0, 1.0, OTSU_HIST_BINS + 1)

        # Handle resume: open existing file in r+ mode, restore histogram
        if resume_row > 0:
            saved_strip, saved_hist = _load_progress(progress_path)
            if saved_hist is not None:
                global_hist_counts = saved_hist
                logger.info(f"Restored histogram from progress file (strip {saved_strip})")
            js_ds = rasterio.open(output_js_path, "r+")
            logger.info(f"Resuming from strip row {resume_row}")
        else:
            js_ds = create_output_tif(
                output_js_path, img_w, img_h, transform, crs,
                "float32", nodata=None, compress=compress,
                count=n_bands, descriptions=band_descriptions
            )

        tiles_done = 0
        t_start = time.time()

        # Track strips: read full-width horizontal strips with overlap
        for strip_row in range(n_tile_rows):
            # Row range for this strip (in image coordinates)
            row_start = strip_row * tile_size
            row_end = min(row_start + tile_size, img_h)
            actual_tile_h = row_end - row_start

            # Padded row range for reading
            read_row_start = max(row_start - pad, 0)
            read_row_end = min(row_end + pad, img_h)
            pad_top = row_start - read_row_start
            pad_bottom = read_row_end - row_end

            # Skip if resuming
            if strip_row < resume_row:
                tiles_done += n_tile_cols
                continue

            # Read the full-width strip (with vertical padding)
            strip_window = Window(0, read_row_start, img_w, read_row_end - read_row_start)
            strip_data = src.read(band, window=strip_window)
            if needs_rescale and not local_rescale:
                strip_data = _rescale_to_uint8(strip_data, nodata_val,
                                               data_min, data_max)
            strip_h_read = strip_data.shape[0]

            logger.debug(f"Strip {strip_row}: rows [{row_start}, {row_end}), "
                        f"read [{read_row_start}, {read_row_end}), "
                        f"shape {strip_data.shape}")

            # Process tiles within this strip
            for tile_col in range(n_tile_cols):
                col_start = tile_col * tile_size
                col_end = min(col_start + tile_size, img_w)
                actual_tile_w = col_end - col_start

                # Padded column range
                read_col_start = max(col_start - pad, 0)
                read_col_end = min(col_end + pad, img_w)
                pad_left = col_start - read_col_start
                pad_right = read_col_end - col_end

                # Extract padded tile from strip
                tile_data = strip_data[:, read_col_start:read_col_end]

                # If padding extends beyond image, reflect-pad
                needs_extra_pad = False
                extra_pad_top = 0
                extra_pad_bottom = 0
                extra_pad_left = 0
                extra_pad_right = 0

                if row_start - pad < 0:
                    extra_pad_top = pad - pad_top
                    needs_extra_pad = True
                if row_end + pad > img_h:
                    extra_pad_bottom = pad - pad_bottom
                    needs_extra_pad = True
                if col_start - pad < 0:
                    extra_pad_left = pad - pad_left
                    needs_extra_pad = True
                if col_end + pad > img_w:
                    extra_pad_right = pad - pad_right
                    needs_extra_pad = True

                if needs_extra_pad:
                    tile_data = np.pad(tile_data,
                                       ((extra_pad_top, extra_pad_bottom),
                                        (extra_pad_left, extra_pad_right)),
                                       mode="reflect")
                    # Adjust padding offsets
                    pad_top += extra_pad_top
                    pad_left += extra_pad_left

                # Local rescale: normalize this tile independently to uint8
                if needs_rescale and local_rescale:
                    local_min, local_max = _local_range(tile_data, nodata_val)
                    tile_data = _rescale_to_uint8(tile_data, nodata_val,
                                                  local_min, local_max)

                # Process tile — returns (n_radii, tile_h, tile_w)
                js_result = process_tile(tile_data, radii, n_bins, pad, use_numba)

                # Trim to actual tile size (in case of rounding)
                js_result = js_result[:, :actual_tile_h, :actual_tile_w]

                # Write to output
                out_window = Window(col_start, row_start, actual_tile_w, actual_tile_h)
                if single_band:
                    # Average across radii for single-band output
                    js_mean = np.nanmean(js_result, axis=0).astype(np.float32)
                    js_ds.write(js_mean, 1, window=out_window)
                    hist_data = js_mean
                else:
                    # Write each radius band separately
                    for band_idx in range(n_radii):
                        js_ds.write(js_result[band_idx], band_idx + 1, window=out_window)
                    # Compute mean across radii for Otsu histogram
                    hist_data = np.nanmean(js_result, axis=0).astype(np.float32)

                # Accumulate global histogram (exclude NaN)
                valid_js = hist_data[~np.isnan(hist_data)]
                if valid_js.size > 0:
                    counts, _ = np.histogram(valid_js, bins=global_hist_edges)
                    global_hist_counts += counts

                tiles_done += 1
                elapsed = time.time() - t_start
                rate = tiles_done / elapsed if elapsed > 0 else 0
                eta = (total_tiles - tiles_done) / rate if rate > 0 else 0
                if tiles_done % 50 == 0 or tiles_done == total_tiles:
                    logger.info(f"  Tile {tiles_done}/{total_tiles} "
                               f"({100*tiles_done/total_tiles:.1f}%) "
                               f"[{rate:.1f} tiles/s, ETA {eta/3600:.1f}h] "
                               f"strip={strip_row} col={tile_col}")

            # Free strip memory
            del strip_data

            # Save progress after each strip
            _save_progress(progress_path, strip_row, global_hist_counts)

        js_ds.close()
        elapsed_total = time.time() - t_start
        logger.info(f"Phase 1 complete: {elapsed_total/3600:.2f}h, "
                    f"output: {output_js_path}")

        # Clean up progress file on successful completion
        if Path(progress_path).exists():
            Path(progress_path).unlink()

    return global_hist_counts, global_hist_edges


def run_phase2(js_path, edge_path, global_hist_counts, global_hist_edges, compress="lzw"):
    """Phase 2: Apply Otsu threshold to produce binary edge map.

    Auto-detects band count from input. If multi-band, computes nanmean
    across bands before thresholding.
    """
    threshold = otsu_threshold(global_hist_counts, global_hist_edges)
    logger.info(f"Otsu threshold: {threshold:.6f}")

    with rasterio.open(js_path) as src:
        img_h = src.height
        img_w = src.width
        n_bands = src.count

        if n_bands > 1:
            logger.info(f"Multi-band input ({n_bands} bands), will average for thresholding")

        edge_ds = create_output_tif(
            edge_path, img_w, img_h, src.transform, src.crs,
            "uint8", nodata=NODATA_EDGE, compress=compress
        )

        # Process in blocks matching the JS output tiling
        block_size = 1024
        n_block_rows = math.ceil(img_h / block_size)
        n_block_cols = math.ceil(img_w / block_size)
        total_blocks = n_block_rows * n_block_cols
        blocks_done = 0
        t_start = time.time()

        for br in range(n_block_rows):
            row_start = br * block_size
            row_end = min(row_start + block_size, img_h)
            bh = row_end - row_start

            for bc in range(n_block_cols):
                col_start = bc * block_size
                col_end = min(col_start + block_size, img_w)
                bw = col_end - col_start

                window = Window(col_start, row_start, bw, bh)

                if n_bands == 1:
                    js_data = src.read(1, window=window)
                else:
                    # Read all bands and compute mean
                    all_bands = src.read(window=window)  # (n_bands, bh, bw)
                    js_data = np.nanmean(all_bands, axis=0).astype(np.float32)

                # Apply threshold
                edge_data = np.zeros_like(js_data, dtype=np.uint8)
                valid = ~np.isnan(js_data)
                edge_data[valid & (js_data >= threshold)] = 1
                edge_data[~valid] = NODATA_EDGE

                edge_ds.write(edge_data, 1, window=window)

                blocks_done += 1
                if blocks_done % 200 == 0 or blocks_done == total_blocks:
                    elapsed = time.time() - t_start
                    logger.info(f"  Edge block {blocks_done}/{total_blocks} "
                               f"({100*blocks_done/total_blocks:.1f}%)")

        edge_ds.close()
        elapsed = time.time() - t_start
        logger.info(f"Phase 2 complete: {elapsed:.1f}s, threshold={threshold:.6f}, "
                    f"output: {edge_path}")

    return threshold


# ---------------------------------------------------------------------------
# Numba warmup
# ---------------------------------------------------------------------------

def warmup_numba():
    """Trigger numba JIT compilation on small dummy data."""
    if not HAS_NUMBA:
        return
    logger.info("Warming up numba JIT compilation...")
    t0 = time.time()
    dummy = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    dummy[0, 0] = 0  # include a nodata pixel
    binned = _bin_tile(dummy, 32)
    ihist = _build_integral_histogram(binned, 32)
    radii = np.array([5], dtype=np.int32)
    nodata_mask = (dummy == 0)
    # Returns (n_radii, tile_h, tile_w) now
    _compute_js_tile(ihist, 32, 6, 52, 52, radii, 64, 64, nodata_mask)
    logger.info(f"Numba warmup done in {time.time()-t0:.1f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="JS Divergence Edge Detection for planetary rasters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # THEMIS uint8 mosaic (legacy default behavior)
  python js_edge_detect.py themis_day_100m.tif -o output_prefix

  # Thermal inertia (float32, NaN nodata)
  python js_edge_detect.py thermal_inertia.tif -o ti_prefix --nodata nan

  # Elevation (int16, -9999 nodata, band 1)
  python js_edge_detect.py mola_dem.tif -o dem_prefix --nodata -9999 --band 1

  # Legacy single-band averaged output
  python js_edge_detect.py input.tif -o output_prefix --single-band

  # Custom radii and resume from strip 50
  python js_edge_detect.py input.tif -o out --radii 5 10 20 --resume-row 50

  # JS divergence only (skip edge map)
  python js_edge_detect.py input.tif -o out --js-only
"""
    )
    parser.add_argument("input", help="Input GeoTIFF path")
    parser.add_argument("-o", "--output-prefix", required=True,
                        help="Output prefix (produces <prefix>_js.tif and <prefix>_edges.tif)")
    parser.add_argument("--radii", type=int, nargs="+", default=DEFAULT_RADII,
                        help=f"Radii in pixels (default: {DEFAULT_RADII})")
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS,
                        help=f"Number of histogram bins (default: {DEFAULT_BINS})")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE,
                        help=f"Tile size in pixels (default: {DEFAULT_TILE_SIZE})")
    parser.add_argument("--resume-row", type=int, default=0,
                        help="Resume from this strip row (for crash recovery)")
    parser.add_argument("--local-rescale", action="store_true",
                        help="Normalize each tile independently to uint8 instead of globally. "
                             "Recommended for DEMs and other data with large global range but "
                             "small local variation (prevents flat areas collapsing to one bin).")
    parser.add_argument("--no-numba", action="store_true",
                        help="Disable numba acceleration (use pure numpy)")
    parser.add_argument("--js-only", action="store_true",
                        help="Only compute JS divergence (skip binary edge map)")
    parser.add_argument("--single-band", action="store_true",
                        help="Output single band (mean of all radii) instead of per-radius bands")
    parser.add_argument("--band", type=int, default=1,
                        help="Band number to read from input raster (default: 1)")
    parser.add_argument("--nodata", type=str, default=None, metavar="VALUE",
                        help="NoData value override. Use 'nan' for NaN, 'none' to treat all "
                             "pixels as valid. Default: read from raster metadata. For uint8 "
                             "input with nodata=0, no rescaling is performed (backward compat)")
    parser.add_argument("--compress", default="lzw",
                        choices=["lzw", "deflate", "zstd", "none"],
                        help="GeoTIFF compression (default: lzw)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging level (default: INFO)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    output_prefix = args.output_prefix
    js_path = f"{output_prefix}_js.tif"
    edge_path = f"{output_prefix}_edges.tif"

    # Ensure output directory exists
    Path(js_path).parent.mkdir(parents=True, exist_ok=True)

    use_numba = (not args.no_numba) and HAS_NUMBA
    if not args.no_numba and not HAS_NUMBA:
        logger.warning("Numba not available, falling back to pure numpy (will be very slow)")

    compress = args.compress if args.compress != "none" else None

    # Resolve --nodata argument
    if args.nodata is None:
        nodata = "auto"
    elif args.nodata.lower() == "none":
        nodata = None
    elif args.nodata.lower() == "nan":
        nodata = float("nan")
    else:
        nodata = float(args.nodata)

    # Warmup numba
    if use_numba:
        warmup_numba()

    # Phase 1: JS divergence
    logger.info("=" * 60)
    logger.info("PHASE 1: Computing JS divergence map")
    logger.info("=" * 60)
    global_hist_counts, global_hist_edges = run_phase1(
        str(input_path), js_path, args.radii, args.bins, args.tile_size,
        use_numba, resume_row=args.resume_row, compress=compress,
        single_band=args.single_band, band=args.band, nodata=nodata,
        local_rescale=args.local_rescale,
    )

    # Phase 2: Binary edges
    if not args.js_only:
        logger.info("=" * 60)
        logger.info("PHASE 2: Computing binary edge map")
        logger.info("=" * 60)
        threshold = run_phase2(js_path, edge_path, global_hist_counts,
                              global_hist_edges, compress=compress)
    else:
        logger.info("Skipping Phase 2 (--js-only)")

    logger.info("Done.")


if __name__ == "__main__":
    main()
