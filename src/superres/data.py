"""Wald's-protocol dataset for per-pair super-resolution training.

Given a directory of real SuperDove 8-band scenes, this dataset emits
(LR-pair, HR-target) chips for one configured RGB pair. The LR input is the
two bands downsampled by ``scale`` using a Gaussian anti-alias filter; the HR
target is the *average* of those two bands at native resolution. Training on
the average pushes the network toward a fused output rather than just one of
the two source bands.

Each scene's per-band reflectance coefficients are loaded from its Planet
XML sidecar (``product="toar"``) or assumed to be 1/10000 (``product="sr"``),
so all chips reach the network in a uniform TOA-reflectance [0, 1] frame
regardless of scene acquisition geometry.

Important: we simulate the real-world per-band sub-pixel offset by jittering
band B relative to band A with a small random shift each sample. Without this
the network only ever sees perfectly co-registered pairs at training time and
then sees offset pairs at inference, hurting generalization.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, shift as nd_shift
from torch.utils.data import Dataset

from .bands import BandPair, SUPERDOVE_BAND_INDEX
from .metadata import SceneCalibration, resolve_calibration


@dataclass
class WaldConfig:
    chip_size: int = 128  # HR chip size in pixels
    scale: int = 2
    product: str = "toar"  # "toar" parses XML sidecar; "sr" uses 1/10000
    blur_sigma: float = 1.0  # Gaussian sigma applied before downsampling
    max_subpixel_jitter: float = 0.5  # ± fraction of a pixel for band B
    chips_per_scene: int = 64
    augment: bool = True


class SuperDoveWaldDataset(Dataset):
    """On-the-fly Wald's protocol degradation."""

    def __init__(
        self,
        scene_paths: list[str | Path],
        pair: BandPair,
        config: WaldConfig | None = None,
        index_map: dict[str, int] = SUPERDOVE_BAND_INDEX,
        seed: int = 0,
    ):
        if not scene_paths:
            raise ValueError("scene_paths is empty")
        self.scene_paths = [Path(p) for p in scene_paths]
        self.pair = pair
        self.cfg = config or WaldConfig()
        self.index_map = index_map
        self.rng = random.Random(seed)
        self._scene_dims = self._index_scene_dims()
        # Eagerly resolve per-scene calibration so multi-worker DataLoaders
        # don't each re-parse XML on every worker process.
        self._calibrations: list[SceneCalibration] = [
            resolve_calibration(p, product=self.cfg.product) for p in self.scene_paths
        ]

    def _index_scene_dims(self) -> list[tuple[int, int]]:
        dims = []
        for p in self.scene_paths:
            with rasterio.open(p) as src:
                dims.append((src.height, src.width))
        return dims

    def __len__(self) -> int:
        return len(self.scene_paths) * self.cfg.chips_per_scene

    def _sample_window(self, scene_idx: int) -> Window:
        h, w = self._scene_dims[scene_idx]
        size = self.cfg.chip_size
        if h < size or w < size:
            raise ValueError(f"scene {self.scene_paths[scene_idx]} smaller than chip size")
        row = self.rng.randint(0, h - size)
        col = self.rng.randint(0, w - size)
        return Window(col_off=col, row_off=row, width=size, height=size)

    def _read_pair(self, scene_idx: int, window: Window) -> np.ndarray:
        ia, ib = self.pair.indices(self.index_map)
        cal = self._calibrations[scene_idx]
        with rasterio.open(self.scene_paths[scene_idx]) as src:
            a = src.read(ia, window=window).astype(np.float32) * cal.coefficient(ia)
            b = src.read(ib, window=window).astype(np.float32) * cal.coefficient(ib)
        return np.stack([a, b], axis=0)

    def _degrade(self, hr_pair: np.ndarray) -> np.ndarray:
        """Wald's protocol: blur + decimate, with per-band jitter on band B."""
        cfg = self.cfg
        # Random sub-pixel jitter applied to band B only, simulating the
        # SuperDove inter-band offset that survives L1B correction.
        if cfg.max_subpixel_jitter > 0:
            jy = self.rng.uniform(-cfg.max_subpixel_jitter, cfg.max_subpixel_jitter)
            jx = self.rng.uniform(-cfg.max_subpixel_jitter, cfg.max_subpixel_jitter)
            band_b = nd_shift(hr_pair[1], shift=(jy, jx), order=3, mode="reflect")
        else:
            band_b = hr_pair[1]

        a_blur = gaussian_filter(hr_pair[0], sigma=cfg.blur_sigma)
        b_blur = gaussian_filter(band_b, sigma=cfg.blur_sigma)
        a_lr = a_blur[:: cfg.scale, :: cfg.scale]
        b_lr = b_blur[:: cfg.scale, :: cfg.scale]
        return np.stack([a_lr, b_lr], axis=0)

    def _augment(self, hr: np.ndarray, lr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Random 90-degree rotation + flips. Same op on both LR and HR.
        k = self.rng.randint(0, 3)
        if k:
            hr = np.rot90(hr, k=k, axes=(-2, -1)).copy()
            lr = np.rot90(lr, k=k, axes=(-2, -1)).copy()
        if self.rng.random() < 0.5:
            hr = np.flip(hr, axis=-1).copy()
            lr = np.flip(lr, axis=-1).copy()
        return hr, lr

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        scene_idx = idx % len(self.scene_paths)
        for _ in range(8):  # retry on invalid chips (all-zero/nodata)
            window = self._sample_window(scene_idx)
            hr_pair = self._read_pair(scene_idx, window)
            if np.isfinite(hr_pair).all() and hr_pair.max() > 0.01:
                break
        else:
            raise RuntimeError(f"could not find valid chip in {self.scene_paths[scene_idx]}")

        lr_pair = self._degrade(hr_pair)
        hr_target = ((hr_pair[0] + hr_pair[1]) * 0.5)[None]  # (1, H, W)

        if self.cfg.augment:
            hr_target, lr_pair = self._augment(hr_target, lr_pair)

        return lr_pair.astype(np.float32), hr_target.astype(np.float32)
