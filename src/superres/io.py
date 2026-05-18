"""Raster I/O helpers for SuperDove 8-band scenes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .bands import BandPair, SUPERDOVE_BAND_INDEX
from .metadata import SceneCalibration


def read_pair(
    scene_path: str | Path,
    pair: BandPair,
    calibration: SceneCalibration,
    window: Window | None = None,
    index_map: dict[str, int] = SUPERDOVE_BAND_INDEX,
) -> np.ndarray:
    """Read a band pair as a float32 (2, H, W) array in TOA reflectance [0, 1].

    ``calibration`` provides per-band DN→reflectance coefficients. For TOAR
    products construct it from the Planet XML sidecar; for SR products use
    :meth:`SceneCalibration.surface_reflectance`.
    """
    ia, ib = pair.indices(index_map)
    with rasterio.open(scene_path) as src:
        a = src.read(ia, window=window).astype(np.float32) * calibration.coefficient(ia)
        b = src.read(ib, window=window).astype(np.float32) * calibration.coefficient(ib)
    return np.stack([a, b], axis=0)


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
    output_scale: float = 10000.0,
    dtype: str = "uint16",
    compress: str = "deflate",
) -> None:
    """Write a (3, H, W) reflectance array as a tiled GeoTIFF.

    ``rgb`` is expected in [0, 1]; it is multiplied by ``output_scale`` and
    cast to ``dtype`` on write. The default 0-10000 ↔ 0-1 mapping matches
    Planet's SR convention so the output is interchangeable with their
    downstream tooling.
    """
    assert rgb.ndim == 3 and rgb.shape[0] == 3, f"expected (3,H,W), got {rgb.shape}"
    if dtype == "uint16":
        arr = np.clip(rgb * output_scale, 0, 65535).astype(np.uint16)
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
