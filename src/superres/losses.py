"""Loss bundle for super-resolution training.

The default mix is L1 + LPIPS (perceptual). LPIPS gives the visual sharpness
the user wants; pure L1 looks soft. A GAN term can be layered on later — see
notes in README.
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:
    import lpips  # type: ignore
    _HAS_LPIPS = True
except ImportError:  # pragma: no cover
    _HAS_LPIPS = False


class SRLoss(nn.Module):
    """L1 + LPIPS for single-channel SR outputs.

    LPIPS expects 3-channel RGB in [-1, 1], so we tile our single-channel
    output across RGB and rescale before feeding it through.
    """

    def __init__(
        self,
        l1_weight: float = 1.0,
        lpips_weight: float = 0.1,
        lpips_net: str = "vgg",
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.lpips_weight = lpips_weight
        self.l1 = nn.L1Loss()
        if lpips_weight > 0:
            if not _HAS_LPIPS:
                raise RuntimeError("lpips not installed; pip install lpips or set lpips_weight=0")
            self.lpips = lpips.LPIPS(net=lpips_net)
            for p in self.lpips.parameters():
                p.requires_grad = False
        else:
            self.lpips = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        loss_l1 = self.l1(pred, target)
        total = self.l1_weight * loss_l1
        out = {"l1": loss_l1.detach(), "total": total}
        if self.lpips is not None:
            pred_rgb = pred.repeat(1, 3, 1, 1).clamp(0, 1) * 2 - 1
            target_rgb = target.repeat(1, 3, 1, 1).clamp(0, 1) * 2 - 1
            loss_lp = self.lpips(pred_rgb, target_rgb).mean()
            total = total + self.lpips_weight * loss_lp
            out["lpips"] = loss_lp.detach()
            out["total"] = total
        return out
