"""Raster I/O helpers for SuperDove 8-band scenes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .bands import BandPair, SUPERDOVE_BAND_INDEX


def read_pair(
    scene_path: str | Path,
    pair: BandPair,
    window: Window | None = None,
    index_map: dict[str, int] = SUPERDOVE_BAND_INDEX,
    scale_factor: float = 10000.0,
) -> np.ndarray:
    """Read a band pair from a SuperDove scene as a float32 array (2, H, W).

    Values are divided by ``scale_factor`` (10000 is the Planet surface
    reflectance convention). Returns the array, not metadata — use
    :func:`read_scene_meta` if you need the affine/CRS.
    """
    ia, ib = pair.indices(index_map)
    with rasterio.open(scene_path) as src:
        a = src.read(ia, window=window).astype(np.float32)
        b = src.read(ib, window=window).astype(np.float32)
    out = np.stack([a, b], axis=0) / scale_factor
    return out


def read_scene_meta(scene_path: str | Path) -> dict:
    """Return a metadata dict (transform, crs, width, height, nodata)."""
    with rasterio.open(scene_path) as src:
        return {
            "transform": src.transform,
            "crs": src.crs,
            "width": src.width,
            "height": src.height,
            "nodata": src.nodata,
            "count": src.count,
            "dtype": src.dtypes[0],
        }


def write_rgb_geotiff(
    out_path: str | Path,
    rgb: np.ndarray,
    transform,
    crs,
    scale_factor: float = 10000.0,
    dtype: str = "uint16",
    compress: str = "deflate",
) -> None:
    """Write a (3, H, W) float RGB array as a tiled COG-friendly GeoTIFF.

    ``rgb`` is expected in [0, 1] (reflectance-like); it is multiplied by
    ``scale_factor`` and cast to ``dtype`` on write. Use a smaller transform
    pixel size to reflect the super-resolved grid.
    """
    assert rgb.ndim == 3 and rgb.shape[0] == 3, f"expected (3,H,W), got {rgb.shape}"
    if dtype == "uint16":
        arr = np.clip(rgb * scale_factor, 0, 65535).astype(np.uint16)
    elif dtype == "uint8":
        arr = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        arr = rgb.astype(dtype)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[1],
        "width": arr.shape[2],
        "count": 3,
        "dtype": arr.dtype,
        "transform": transform,
        "crs": crs,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "compress": compress,
        "predictor": 2,
        "interleave": "pixel",
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)


def scaled_transform(transform, scale: int):
    """Return a new affine transform with pixel size divided by ``scale``."""
    return transform * transform.scale(1.0 / scale, 1.0 / scale)
