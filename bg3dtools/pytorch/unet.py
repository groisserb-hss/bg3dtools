#!/usr/bin/env python3
"""Residual‑U‑Net (stride‑conv down, transposed‑conv up)

* Replaces the classic DoubleConv with a lightweight **HalfResidualConv**
  block (residual skip around the *second* convolution only).
* Drops max‑pool and bilinear upsample: **all scaling is learned** with
  stride‑2 convolutions in the encoder and `ConvTranspose2d` in the decoder.
* Depth and base width remain configurable via `depth` and `base_channels`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 1.  HalfResidualConv – the new atomic unit
# ---------------------------------------------------------------------------

class HalfResidualConv(nn.Module):
    """Conv‑BN‑ReLU → (Conv‑BN + skip) → ReLU.

    Parameters
    ----------
    in_channels : int — input channel count.
    out_channels: int — output channel count **and** the channel count inside
                       the residual unit.  If `in_channels != out_channels`
                       the first convolution handles the change; the residual
                       skip starts *after* that conversion, so channels match.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        p = (k - 1) // 2

        # Conv‑BN‑ReLU that can change channel dimension or keep it.
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Second Conv‑BN (no activation) operating on `out_channels`.
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.prep(x)
        return self.relu(x + self.conv(x))

# ---------------------------------------------------------------------------
# 2.  Down & Up blocks using strided / transposed convolutions
# ---------------------------------------------------------------------------


class Down(nn.Module):
    """Spatial down‑scaling via stride‑2 Conv, then HalfResidualConv."""

    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            HalfResidualConv(out_channels, out_channels, k=k),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.down(x)


class Up(nn.Module):
    """Transposed‑conv upsample then HalfResidualConv after concatenation."""

    def __init__(self, in_channels: int, out_channels: int, k: int) -> None:
        super().__init__()
        # Up‑scaling: halve channels, double spatial dims
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # After concatenation we have 2×out_channels → compress back
        self.conv = HalfResidualConv(out_channels * 2, out_channels, k=k)

    def forward(self, x_high: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x_high = self.up(x_high)

        # Pad if necessary (rare with power‑of‑2 inputs but keeps things robust)
        diff_y = x_skip.size(2) - x_high.size(2)
        diff_x = x_skip.size(3) - x_high.size(3)
        x_high = F.pad(x_high, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x_skip, x_high], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1×1 convolution with sigmoid activation."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return torch.sigmoid(self.conv(x))

# ---------------------------------------------------------------------------
# 3.  Full Residual‑U‑Net
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """U‑Net with residual blocks and learned down/up sampling."""

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        *,
        channels: tuple = (8, 16, 32, 32),
        kernel_sizes: tuple = (7, 5, 3, 3)
    ) -> None:
        super().__init__()
        self.depth = len(channels)

        # Encoder ----------------------------------------------------------------
        self.inc = HalfResidualConv(n_channels, channels[0], k=kernel_sizes[0])

        self.downs = nn.ModuleList()
        for d in range(self.depth - 1):
            self.downs.append(Down(channels[d], channels[d + 1], k=kernel_sizes[d]))

        # Decoder ----------------------------------------------------------------
        self.ups = nn.ModuleList()
        for d in range(self.depth - 1, 0, -1):
            self.ups.append(Up(channels[d], channels[d - 1], k=kernel_sizes[d - 1]))

        # Output conv ------------------------------------------------------------
        self.outc = OutConv(channels[0], n_classes)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_, target_=None):
        x = torch.stack(input_, dim=0)

        # Encoder with skip collection
        skips: list[torch.Tensor] = []
        x = self.inc(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)

        # Decoder with skip fusion
        for up in self.ups:
            x = up(x, skips.pop())

        y = self.outc(x)

        if target_ is None:
            return y
        target = torch.stack([t["masks"] for t in target_], dim=0)
        return self.get_loss(y, target)

    # ------------------------------------------------------------------
    @staticmethod
    def get_loss(prediction: torch.Tensor, target: torch.Tensor):
        err = (prediction - target) ** 2
        return {"l2": err.sum() / err.shape[0]}

# ---------------------------------------------------------------------------
# End of file
# ---------------------------------------------------------------------------
