"""RRDBNet backbone adapted for 2-channel input / 1-channel output SR.

This is the Real-ESRGAN / ESRGAN generator architecture (Wang et al., 2018),
minimal self-contained implementation so we don't take a hard dependency on
basicsr. Default at 2x upscale to match the band-pair fusion target; switch
to 4x by raising ``scale``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseBlock(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + x5 * self.residual_scale


class _RRDB(nn.Module):
    def __init__(self, nf: int = 64, gc: int = 32, residual_scale: float = 0.2):
        super().__init__()
        self.db1 = _DenseBlock(nf, gc, residual_scale)
        self.db2 = _DenseBlock(nf, gc, residual_scale)
        self.db3 = _DenseBlock(nf, gc, residual_scale)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + out * self.residual_scale


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        num_features: int = 64,
        num_blocks: int = 16,
        growth_channels: int = 32,
        scale: int = 2,
    ):
        super().__init__()
        if scale not in (1, 2, 4):
            raise ValueError(f"scale must be 1, 2, or 4 (got {scale})")
        self.scale = scale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, padding=1)
        self.body = nn.Sequential(
            *[_RRDB(num_features, growth_channels) for _ in range(num_blocks)]
        )
        self.conv_body = nn.Conv2d(num_features, num_features, 3, padding=1)

        upsample_layers: list[nn.Module] = []
        for _ in range(int(torch.log2(torch.tensor(scale)).item())) if scale > 1 else []:
            upsample_layers += [
                nn.Conv2d(num_features, num_features, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.upsample = nn.Sequential(*upsample_layers) if upsample_layers else nn.Identity()

        self.conv_hr = nn.Conv2d(num_features, num_features, 3, padding=1)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        body = self.conv_body(self.body(feat))
        feat = feat + body
        if self.scale > 1:
            # nearest-upsample inside the upsample stack, mirroring Real-ESRGAN
            n_steps = int(torch.log2(torch.tensor(self.scale)).item())
            for i in range(n_steps):
                feat = F.interpolate(feat, scale_factor=2, mode="nearest")
                # apply (conv, act) pair from self.upsample
                conv = self.upsample[2 * i]
                act = self.upsample[2 * i + 1]
                feat = act(conv(feat))
        feat = self.act(self.conv_hr(feat))
        return self.conv_last(feat)


def build_default_model(scale: int = 2) -> RRDBNet:
    return RRDBNet(in_channels=2, out_channels=1, num_blocks=16, scale=scale)
