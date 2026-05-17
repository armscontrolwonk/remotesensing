"""End-to-end inference: 8-band SuperDove scene → super-resolved RGB GeoTIFF.

Loads one trained model per RGB pair, tiles the input scene with overlap,
runs each model, stitches with Hann-window blending to suppress tile seams,
then writes a GeoTIFF on the upscaled grid.

Usage:
    python -m superres.infer \\
        --scene path/to/8band.tif \\
        --blue checkpoints/blue/best.pt \\
        --green checkpoints/green/best.pt \\
        --red checkpoints/red/best.pt \\
        --out out_rgb.tif
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from .bands import DEFAULT_PAIRS, RED_EDGE_PAIRS, BandPair, SUPERDOVE_BAND_INDEX
from .io import scaled_transform, write_rgb_geotiff
from .model import build_default_model
from .registration import coarse_align


def _hann_window_2d(size: int) -> np.ndarray:
    w = np.hanning(size).astype(np.float32)
    return np.outer(w, w)


def _load_model(ckpt_path: Path, device: torch.device, scale: int) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_default_model(scale=scale).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _read_scene_bands(scene_path: Path, pair: BandPair, scale_factor: float) -> np.ndarray:
    ia, ib = pair.indices()
    with rasterio.open(scene_path) as src:
        a = src.read(ia).astype(np.float32)
        b = src.read(ib).astype(np.float32)
    return np.stack([a, b], axis=0) / scale_factor


def _tile_infer(
    pair_arr: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    scale: int,
    tile: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    """Tile-based inference with Hann-window blending. Returns (H*scale, W*scale)."""
    _, h, w = pair_arr.shape
    out_h, out_w = h * scale, w * scale
    out = np.zeros((out_h, out_w), dtype=np.float32)
    weight = np.zeros_like(out)

    stride = tile - overlap
    win = _hann_window_2d(tile * scale)

    ys = list(range(0, max(1, h - tile + 1), stride))
    xs = list(range(0, max(1, w - tile + 1), stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                chip = pair_arr[:, y : y + tile, x : x + tile]
                # Pad if at the edge (shouldn't happen given ys/xs above, but safe)
                if chip.shape[1] != tile or chip.shape[2] != tile:
                    pad_h = tile - chip.shape[1]
                    pad_w = tile - chip.shape[2]
                    chip = np.pad(chip, ((0, 0), (0, pad_h), (0, pad_w)), mode="reflect")
                tens = torch.from_numpy(chip).unsqueeze(0).to(device)
                pred = model(tens).squeeze().cpu().numpy()
                oy, ox = y * scale, x * scale
                out[oy : oy + tile * scale, ox : ox + tile * scale] += pred * win
                weight[oy : oy + tile * scale, ox : ox + tile * scale] += win

    out = out / np.clip(weight, 1e-6, None)
    return out


def _histogram_match(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Per-channel CDF matching, src and ref are (3, H, W) in [0, 1]."""
    matched = np.empty_like(src)
    for c in range(src.shape[0]):
        s = src[c].ravel()
        r = ref[c].ravel()
        s_sorted_idx = np.argsort(s)
        s_sorted = s[s_sorted_idx]
        r_sorted = np.sort(r)
        # Map s percentiles to r percentiles by interp
        new_vals = np.interp(
            np.linspace(0, 1, s.size),
            np.linspace(0, 1, r_sorted.size),
            r_sorted,
        )
        out = np.empty_like(s)
        out[s_sorted_idx] = new_vals
        matched[c] = out.reshape(src[c].shape)
    return matched


def infer(
    scene_path: Path,
    ckpts: dict[str, Path],
    out_path: Path,
    use_red_edge: bool = False,
    scale: int = 2,
    tile: int = 256,
    overlap: int = 32,
    scale_factor: float = 10000.0,
    align: bool = True,
    color_match: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    dev = torch.device(device)
    pairs = RED_EDGE_PAIRS if use_red_edge else DEFAULT_PAIRS

    with rasterio.open(scene_path) as src:
        transform = src.transform
        crs = src.crs

    rgb_channels: list[np.ndarray] = []
    naive_rgb_channels: list[np.ndarray] = []
    for pair in pairs:
        if pair.name not in ckpts:
            raise ValueError(f"missing checkpoint for {pair.name!r}")
        pair_arr = _read_scene_bands(scene_path, pair, scale_factor)
        if align:
            pair_arr, residual = coarse_align(pair_arr)
            print(f"[{pair.name}] residual sub-pixel offset: {residual}")
        naive_rgb_channels.append((pair_arr[0] + pair_arr[1]) * 0.5)

        model = _load_model(ckpts[pair.name], dev, scale=scale)
        sr = _tile_infer(pair_arr, model, dev, scale=scale, tile=tile, overlap=overlap)
        rgb_channels.append(sr)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    rgb = np.stack(rgb_channels, axis=0).clip(0, None)
    if color_match:
        naive_rgb = np.stack(naive_rgb_channels, axis=0)
        # Upsample naive RGB to SR grid for reference distribution (nearest is fine)
        ref = np.repeat(np.repeat(naive_rgb, scale, axis=1), scale, axis=2)
        rgb = _histogram_match(rgb, ref)

    sr_transform = scaled_transform(transform, scale)
    write_rgb_geotiff(out_path, rgb, sr_transform, crs, scale_factor=scale_factor)
    print(f"wrote {out_path} ({rgb.shape[1]}x{rgb.shape[2]})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--blue", required=True, type=Path)
    ap.add_argument("--green", required=True, type=Path)
    ap.add_argument("--red", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--red-edge", action="store_true", help="use Red+RedEdge instead of Red+Yellow")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--no-align", action="store_true")
    ap.add_argument("--no-color-match", action="store_true")
    args = ap.parse_args()

    infer(
        scene_path=args.scene,
        ckpts={"blue": args.blue, "green": args.green, "red": args.red},
        out_path=args.out,
        use_red_edge=args.red_edge,
        scale=args.scale,
        tile=args.tile,
        overlap=args.overlap,
        align=not args.no_align,
        color_match=not args.no_color_match,
    )


if __name__ == "__main__":
    main()
