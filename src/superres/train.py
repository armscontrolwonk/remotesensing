"""Train one super-resolution model for a single RGB band pair.

Run three times — once per pair — to produce a full RGB pipeline. Pair name
is passed via ``--pair`` and selects from :mod:`superres.bands`.

Usage:
    python -m superres.train --config configs/default.yaml --pair blue
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .bands import DEFAULT_PAIRS, RED_EDGE_PAIRS, BandPair
from .data import SuperDoveWaldDataset, WaldConfig
from .losses import SRLoss
from .model import build_default_model


def _resolve_pair(name: str, use_red_edge: bool) -> BandPair:
    pairs = RED_EDGE_PAIRS if use_red_edge else DEFAULT_PAIRS
    for p in pairs:
        if p.name == name:
            return p
    raise ValueError(f"unknown pair {name!r}; choose from {[p.name for p in pairs]}")


def _gather_scenes(data_root: str | Path, glob: str) -> list[Path]:
    paths = sorted(Path(data_root).rglob(glob))
    if not paths:
        raise FileNotFoundError(f"no scenes matching {glob!r} under {data_root}")
    return paths


def train(cfg: dict, pair_name: str) -> None:
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    pair = _resolve_pair(pair_name, cfg.get("use_red_edge", False))
    out_dir = Path(cfg["checkpoint_dir"]) / pair.name
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = _gather_scenes(cfg["data_root"], cfg.get("scene_glob", "*.tif"))
    wald_cfg = WaldConfig(**cfg.get("wald", {}))
    dataset = SuperDoveWaldDataset(scenes, pair=pair, config=wald_cfg, seed=cfg.get("seed", 0))

    n_val = max(1, int(len(dataset) * cfg.get("val_split", 0.05)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(cfg.get("seed", 0))
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    model = build_default_model(scale=wald_cfg.scale).to(device)
    loss_fn = SRLoss(
        l1_weight=cfg.get("l1_weight", 1.0),
        lpips_weight=cfg.get("lpips_weight", 0.1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        betas=(0.9, 0.99),
        weight_decay=cfg.get("weight_decay", 0.0),
    )
    epochs = cfg.get("epochs", 50)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * steps_per_epoch), eta_min=cfg.get("lr_min", 1e-6)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and cfg.get("amp", True))

    best_val = math.inf
    print(f"[{pair.name}] train scenes={len(scenes)} chips/epoch={len(train_set)} device={device}")
    for epoch in range(epochs):
        model.train()
        running = {"total": 0.0, "n": 0}
        t0 = time.time()
        for lr_pair, hr_target in tqdm(train_loader, desc=f"ep{epoch+1}/{epochs}"):
            lr_pair = lr_pair.to(device, non_blocking=True)
            hr_target = hr_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                pred = model(lr_pair)
                losses = loss_fn(pred, hr_target)
            scaler.scale(losses["total"]).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running["total"] += losses["total"].item() * lr_pair.size(0)
            running["n"] += lr_pair.size(0)

        # Validation
        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for lr_pair, hr_target in val_loader:
                lr_pair = lr_pair.to(device)
                hr_target = hr_target.to(device)
                pred = model(lr_pair)
                losses = loss_fn(pred, hr_target)
                val_loss += losses["total"].item() * lr_pair.size(0)
                val_n += lr_pair.size(0)
        val_loss /= max(1, val_n)
        train_loss = running["total"] / max(1, running["n"])
        dt = time.time() - t0
        print(f"[{pair.name}] epoch={epoch+1} train={train_loss:.4f} val={val_loss:.4f} ({dt:.0f}s)")

        ckpt = {
            "model_state": model.state_dict(),
            "config": cfg,
            "pair": asdict(pair) if hasattr(pair, "__dataclass_fields__") else pair.__dict__,
            "wald": asdict(wald_cfg),
            "epoch": epoch + 1,
            "val_loss": val_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, out_dir / "best.pt")
            print(f"[{pair.name}] new best val={val_loss:.4f} → best.pt")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--pair", required=True, choices=["blue", "green", "red"])
    args = ap.parse_args()
    with args.config.open() as f:
        cfg = yaml.safe_load(f)
    train(cfg, args.pair)


if __name__ == "__main__":
    main()
