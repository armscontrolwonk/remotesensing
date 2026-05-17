"""Smoke tests: verify shapes/dtypes flow end-to-end without real Planet data."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from superres.bands import DEFAULT_PAIRS
from superres.data import SuperDoveWaldDataset, WaldConfig
from superres.io import read_pair, scaled_transform, write_rgb_geotiff
from superres.model import build_default_model
from superres.registration import coarse_align, estimate_offset


def _synth_scene(path, height=256, width=256, n_bands=8, seed=0):
    rng = np.random.default_rng(seed)
    # Spectrally smooth: each band is a slightly shifted version of a base image.
    base = rng.integers(2000, 6000, size=(height, width), dtype=np.uint16)
    data = np.empty((n_bands, height, width), dtype=np.uint16)
    for b in range(n_bands):
        shifted = np.roll(base, shift=b, axis=1)
        noise = rng.integers(-200, 200, size=base.shape)
        data[b] = np.clip(shifted.astype(int) + noise, 0, 65535).astype(np.uint16)
    transform = from_origin(0, 0, 3.0, 3.0)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width, count=n_bands,
        dtype="uint16", transform=transform, crs="EPSG:32610",
    ) as dst:
        dst.write(data)


def test_read_pair(tmp_path):
    scene = tmp_path / "scene.tif"
    _synth_scene(scene)
    pair = DEFAULT_PAIRS[0]
    arr = read_pair(scene, pair)
    assert arr.shape == (2, 256, 256)
    assert arr.dtype == np.float32
    assert 0 < arr.mean() < 1


def test_dataset_yields_tensors(tmp_path):
    scenes = [tmp_path / f"s{i}.tif" for i in range(2)]
    for i, p in enumerate(scenes):
        _synth_scene(p, seed=i)
    pair = DEFAULT_PAIRS[1]
    cfg = WaldConfig(chip_size=64, scale=2, chips_per_scene=4, augment=True)
    ds = SuperDoveWaldDataset(scenes, pair=pair, config=cfg)
    assert len(ds) == 8
    lr, hr = ds[0]
    assert lr.shape == (2, 32, 32)
    assert hr.shape == (1, 64, 64)
    assert lr.dtype == np.float32
    assert hr.dtype == np.float32


def test_model_forward():
    model = build_default_model(scale=2).eval()
    x = torch.randn(1, 2, 32, 32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 1, 64, 64)


def test_registration_recovers_known_offset():
    from scipy.ndimage import shift as nd_shift

    rng = np.random.default_rng(42)
    ref = rng.normal(size=(128, 128)).astype(np.float32)
    true_dy, true_dx = 0.7, -1.3
    moving = nd_shift(ref, shift=(true_dy, true_dx), order=3, mode="reflect")
    est_dy, est_dx = estimate_offset(ref, moving, upsample_factor=20)
    # phase_cross_correlation returns the shift to apply to moving → reference,
    # which is the negative of the shift we applied above.
    assert abs(est_dy + true_dy) < 0.15
    assert abs(est_dx + true_dx) < 0.15


def test_coarse_align_runs():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(64, 64)).astype(np.float32)
    b = np.roll(a, shift=(1, 2), axis=(0, 1))
    pair = np.stack([a, b], axis=0)
    aligned, residual = coarse_align(pair)
    assert aligned.shape == pair.shape
    assert abs(residual[0]) < 0.5
    assert abs(residual[1]) < 0.5


def test_write_rgb_roundtrip(tmp_path):
    rgb = np.random.default_rng(0).random((3, 32, 32)).astype(np.float32)
    transform = from_origin(0, 0, 3.0, 3.0)
    sr_transform = scaled_transform(transform, 2)
    out = tmp_path / "rgb.tif"
    write_rgb_geotiff(out, rgb, sr_transform, "EPSG:32610")
    with rasterio.open(out) as src:
        assert src.count == 3
        assert src.width == 32 and src.height == 32
        assert abs(src.transform.a - 1.5) < 1e-6  # halved pixel size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
