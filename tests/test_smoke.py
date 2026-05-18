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
from superres.metadata import (
    SceneCalibration,
    find_planet_metadata,
    resolve_calibration,
)
from superres.model import build_default_model
from superres.registration import coarse_align, estimate_offset


# Realistic per-band reflectanceCoefficient values for a Planet 8-band TOAR scene.
_TOAR_COEFFS = {
    1: 2.05e-05,  # Coastal Blue
    2: 1.99e-05,  # Blue
    3: 2.10e-05,  # Green I
    4: 2.07e-05,  # Green
    5: 2.21e-05,  # Yellow
    6: 2.35e-05,  # Red
    7: 2.48e-05,  # Red Edge
    8: 2.60e-05,  # NIR
}


def _synth_scene(path, height=256, width=256, n_bands=8, seed=0):
    """Write a synthetic 8-band uint16 GeoTIFF that roughly mimics Planet DN scaling."""
    rng = np.random.default_rng(seed)
    # DN values roughly in [2000, 30000] so that after multiplying by ~2e-5 we
    # land in a plausible TOA reflectance range.
    base = rng.integers(5000, 25000, size=(height, width), dtype=np.uint16)
    data = np.empty((n_bands, height, width), dtype=np.uint16)
    for b in range(n_bands):
        shifted = np.roll(base, shift=b, axis=1)
        noise = rng.integers(-500, 500, size=base.shape)
        data[b] = np.clip(shifted.astype(int) + noise, 0, 65535).astype(np.uint16)
    transform = from_origin(0, 0, 3.0, 3.0)
    with rasterio.open(
        path, "w",
        driver="GTiff", height=height, width=width, count=n_bands,
        dtype="uint16", transform=transform, crs="EPSG:32610",
    ) as dst:
        dst.write(data)


def _synth_planet_xml(path, coeffs=_TOAR_COEFFS):
    """Write a minimal Planet-style XML sidecar with per-band reflectance coefficients."""
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<ps:EarthObservation xmlns:ps="http://schemas.planet.com/ps/v1/planet_product_metadata">',
        "  <eop:resultOf xmlns:eop=\"http://earth.esa.int/eop\">",
        "    <ps:EarthObservationResult>",
        "      <eop:product>",
        "        <eop:ProductInformation>",
    ]
    for band_num, coeff in sorted(coeffs.items()):
        parts += [
            "          <ps:bandSpecificMetadata>",
            f"            <ps:bandNumber>{band_num}</ps:bandNumber>",
            f"            <ps:reflectanceCoefficient>{coeff:.6e}</ps:reflectanceCoefficient>",
            "          </ps:bandSpecificMetadata>",
        ]
    parts += [
        "        </eop:ProductInformation>",
        "      </eop:product>",
        "    </ps:EarthObservationResult>",
        "  </eop:resultOf>",
        "</ps:EarthObservation>",
    ]
    path.write_text("\n".join(parts))


def _synth_toar_scene(tmp_path, name="scene", **kwargs):
    """Write paired TIF + XML sidecar in TOAR convention."""
    tif = tmp_path / f"{name}.tif"
    xml = tmp_path / f"{name}_metadata.xml"
    _synth_scene(tif, **kwargs)
    _synth_planet_xml(xml)
    return tif


def test_parse_planet_xml(tmp_path):
    xml = tmp_path / "x_metadata.xml"
    _synth_planet_xml(xml)
    cal = SceneCalibration.from_planet_xml(xml)
    assert cal.reflectance_coefficients == _TOAR_COEFFS
    assert abs(cal.coefficient(1) - 2.05e-05) < 1e-10


def test_find_planet_metadata(tmp_path):
    scene = _synth_toar_scene(tmp_path)
    found = find_planet_metadata(scene)
    assert found is not None
    assert found.name == "scene_metadata.xml"


def test_resolve_calibration_toar_and_sr(tmp_path):
    scene = _synth_toar_scene(tmp_path)
    toar_cal = resolve_calibration(scene, product="toar")
    assert abs(toar_cal.coefficient(2) - 1.99e-05) < 1e-10

    sr_cal = resolve_calibration(scene, product="sr")
    assert sr_cal.coefficient(2) == 1.0 / 10000.0


def test_resolve_toar_fails_without_xml(tmp_path):
    scene = tmp_path / "lonely.tif"
    _synth_scene(scene)
    with pytest.raises(FileNotFoundError):
        resolve_calibration(scene, product="toar")


def test_read_pair_applies_coefficients(tmp_path):
    scene = _synth_toar_scene(tmp_path)
    pair = DEFAULT_PAIRS[0]  # Coastal Blue + Blue → indices (1, 2)
    cal = resolve_calibration(scene, product="toar")
    arr = read_pair(scene, pair, cal)
    assert arr.shape == (2, 256, 256)
    assert arr.dtype == np.float32
    # Reflectance values should land in a plausible TOA range.
    assert 0.05 < arr.mean() < 0.7
    assert arr.max() < 1.5


def test_dataset_yields_tensors(tmp_path):
    scenes = [_synth_toar_scene(tmp_path, name=f"s{i}", seed=i) for i in range(2)]
    pair = DEFAULT_PAIRS[1]
    cfg = WaldConfig(chip_size=64, scale=2, chips_per_scene=4, augment=True, product="toar")
    ds = SuperDoveWaldDataset(scenes, pair=pair, config=cfg)
    assert len(ds) == 8
    lr, hr = ds[0]
    assert lr.shape == (2, 32, 32)
    assert hr.shape == (1, 64, 64)
    assert lr.dtype == np.float32
    assert hr.dtype == np.float32
    # Reflectance-scale values
    assert lr.max() < 1.5 and hr.max() < 1.5


def test_dataset_sr_product_works(tmp_path):
    # SR product: no XML sidecar required.
    scenes = [tmp_path / f"sr{i}.tif" for i in range(2)]
    for i, p in enumerate(scenes):
        _synth_scene(p, seed=i)
    cfg = WaldConfig(chip_size=64, scale=2, chips_per_scene=2, product="sr")
    ds = SuperDoveWaldDataset(scenes, pair=DEFAULT_PAIRS[2], config=cfg)
    lr, hr = ds[0]
    assert lr.shape == (2, 32, 32)
    assert hr.shape == (1, 64, 64)


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
