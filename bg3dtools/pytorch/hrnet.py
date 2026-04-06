#!/usr/bin/env python3
"""
Fully annotated, **comment‑heavy** re‑printing of a minimal High‑Resolution
Network (HRNet) implementation.

The goal of this version is **pedagogical clarity** – we do *not* touch a single
executable line from the original submission, but we surround every logical
unit with rich explanatory comments, docstrings, and inline notes so that a
newcomer can follow the reasoning behind each architectural choice **line by
line**.

-------------------------------------------------------------------------
Paper reference
-------------------------------------------------------------------------
* **Deep High‑Resolution Representation Learning for Visual Recognition**
  (Wang *et al.*, 2020 – arXiv:1908.07919)

-------------------------------------------------------------------------
Authorship & edit history
-------------------------------------------------------------------------
* Original concise code: **Shuchen Du** (date unknown in snippet)
* Comment expansion: ChatGPT (OpenAI o3) – 2025‑05‑01

> 💡 **Disclaimer**
> These comments highlight potential pitfalls (marked “NOTE”), common PyTorch
> idioms, layer interactions, and HRNet design philosophy.  They do **not** fix
> bugs or change behaviour. If you copy‑paste this file the network will run
> identically to the original.
"""

# ---------------------------------------------------------------------------
# Standard lib & PyTorch imports
# ---------------------------------------------------------------------------
import torch
from torch import nn

# ---------------------------------------------------------------------------
#  Global constants & debug aids
# ---------------------------------------------------------------------------
BN_MOMENTUM = 0.1  # BatchNorm momentum used across all layers (from HRNet paper)

# Torch helper that raises helpful stack traces when autograd detects weird
# operations (slower – enable only during development).
torch.autograd.set_detect_anomaly(True)

# ===========================================================================
#                           1.  PRIMITIVE BUILDING BLOCKS
# ===========================================================================

class Conv(nn.Module):
    """Helper layer: *Conv2d → BatchNorm2d → (optional) ReLU*.

    Parameters
    ----------
    in_ch, out_ch : int
        Input / output channel counts.
    kernel_size   : int, default 3
        Convolutional kernel size.
    stride        : int, default 1
        Stride for the convolution.
    relued        : bool, default True
        Whether to append a ReLU non‑linearity.  Some blocks omit the final
        activation to stay compatible with residual connections.
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, relued=True):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2  # SAME‑padding for odd kernels

        # Sequential container keeps order Conv → BN
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=BN_MOMENTUM)
        )
        self.relu = nn.ReLU()
        self.relued = relued  # store flag so forward() can skip activation

    def forward(self, x):
        x = self.conv_bn(x)
        if self.relued:
            x = self.relu(x)
        return x

# ---------------------------------------------------------------------------
#  Residual blocks – BasicBlock & Bottleneck
# ---------------------------------------------------------------------------

class BasicBlock(nn.Module):
    """Shallow *basic* residual block (≈ ResNet‑18/34 design).

    Architecture:
        ┌────────┐                  ┌────────────────────┐
    x ─▶│ Conv‑BN │─ ReLU ─ Conv‑BN │  + skip connection │─ ReLU ─▶ out
        └────────┘                  └────────────────────┘

    Note
    ----
    * The residual addition requires the input and output tensors to share the
      same shape.  If channel counts change inside the block the residual path
      must be adapted accordingly – the original implementation here **assumes
      they do not change**, which is fine for HRNet stages where each branch
      keeps a fixed width.
    """

    def __init__(self, in_ch, out_ch):
        super(BasicBlock, self).__init__()

        # ⚠️  Important: The second Conv still consumes `in_ch` – *not* out_ch.
        # If the first conv changes the channel dimension this will break.
        # HRNet chooses widths so that `in_ch == out_ch` within a branch.
        self.conv = nn.Sequential(
            Conv(in_ch, out_ch),
            Conv(in_ch, out_ch, relued=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x  # preserve input for skip connection
        x = self.conv(x)
        x = x + identity
        return self.relu(x)


class Bottleneck(nn.Module):
    """Deeper *bottleneck* residual block (≈ ResNet‑50/101 design).

    Uses the 1×1 → 3×3 → 1×1 channel squeezing pattern.
    """

    expansion = 4  # output channels = `out_ch * expansion`

    def __init__(self, in_ch, out_ch, downsampling=None):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            Conv(in_ch, out_ch, kernel_size=1),        # 1×1 reduce
            Conv(out_ch, out_ch),                      # 3×3
            Conv(out_ch, out_ch * self.expansion,      # 1×1 expand
                 kernel_size=1, relued=False)
        )
        self.relu = nn.ReLU()
        self.downsampling = downsampling  # optional 1×1 Conv to match channels

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if self.downsampling:
            identity = self.downsampling(identity)
        # -----------------------------------------------------------------------------------------------------------------
        # NOTE Potential bug: the original code writes `x = + identity` (unary +)
        # which simply makes `x` a *positive* copy of itself and discards `x`!
        # It *should* be `x += identity` or `x = x + identity`. We leave it as‑is
        # to honour the “no code modifications” rule.
        # -----------------------------------------------------------------------------------------------------------------
        x = + identity
        return self.relu(x)

# ---------------------------------------------------------------------------
#          2.  Helper modules for scale adaptation (Up/Down samplers)
# ---------------------------------------------------------------------------

class UpSampling(nn.Module):
    """Nearest/bilinear up‑sampling followed by a 1×1 Conv to reduce channels.

    Mirrors the fusion strategy in the original HRNet where lower‑resolution
    branches are upsampled and aligned before being summed into higher‑res
    ones.
    """

    def __init__(self, ch, up_factor):
        super(UpSampling, self).__init__()

        # We first expand spatial dims (bilinear) then compress channels so the
        # receiving branch doesn’t explode in width.
        self.up_sampling = nn.Sequential(
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False),
            Conv(ch, ch // up_factor, 1, relued=False)
        )

    def forward(self, x):
        return self.up_sampling(x)


class DownSampling(nn.Module):
    """Repeated stride‑2 Conv blocks to move high‑res features → low‑res space."""

    def __init__(self, ch, num_samplings):
        super(DownSampling, self).__init__()
        convs = []
        for i in range(num_samplings):
            # Add ReLU after every conv *except* the last (mirrors HRNet fuse op)
            relued = True if i < num_samplings - 1 else False
            convs.append(Conv(ch, ch * 2, 3, 2, relued=relued))
            ch = 2 * ch  # double channels after each down‑scaling
        self.down_sampling = nn.Sequential(*convs)

    def forward(self, x):
        return self.down_sampling(x)

# ---------------------------------------------------------------------------
#                    3.  HRNet *stage* containing multi‑resolution branches
# ---------------------------------------------------------------------------

class HRBlock(nn.Module):
    """One *stage* of HRNet consisting of parallel branches and fusion layers.

    Parameters
    ----------
    ch : int
        Base channel width of the highest‑resolution branch.
    index : int
        Number of parallel branches in this stage (resolution count).
    last_stage : bool
        Whether this is the final stage of the whole network (affects fusion).
    block : nn.Module
        Residual block class to use (e.g. `BasicBlock`).
    num_conv_block_per_list : int, default 4
        How many residual blocks to chain **within each branch**.
    """

    def __init__(self, ch, index, last_stage, block, num_conv_block_per_list=4):
        super(HRBlock, self).__init__()
        self.index = index
        self.last_stage = last_stage
        self.num_conv_block_per_list = num_conv_block_per_list
        self.relu = nn.ReLU()

        # ------------------------------------------------------------------
        # a) Parallel *intra‑branch* stacks (each keeps its own resolution)
        # ------------------------------------------------------------------
        self.parallel_conv_lists = nn.ModuleList()
        for i in range(index):
            ch_i = ch * 2 ** i  # width doubles as resolution halves
            conv_list = []
            for j in range(num_conv_block_per_list):
                conv_list.append(block(ch_i, ch_i))
            self.parallel_conv_lists.append(nn.Sequential(*conv_list))

        # ------------------------------------------------------------------
        # b) *Inter‑branch* up‑sampling paths (low‑res → high‑res)
        # ------------------------------------------------------------------
        self.up_conv_lists = nn.ModuleList()
        for i in range(index - 1):
            conv_list = nn.ModuleList()
            for j in range(i + 1, index):
                up_factor = 2 ** (j - i)
                ch_j = ch * 2 ** j
                conv_list.append(UpSampling(ch_j, up_factor))
            self.up_conv_lists.append(conv_list)

        # ------------------------------------------------------------------
        # c) *Inter‑branch* down‑sampling paths (high‑res → low‑res)
        # ------------------------------------------------------------------
        self.down_conv_lists = nn.ModuleList()
        for i in range(1, index if last_stage else index + 1):
            conv_list = nn.ModuleList()
            for j in range(i):
                ch_j = ch * 2 ** j
                conv_list.append(DownSampling(ch_j, i - j))
            self.down_conv_lists.append(conv_list)

    # --------------------------- forward pass --------------------------------
    def forward(self, x_list):
        # Pass each input through its corresponding same‑resolution branch
        parallel_res_list = []
        for i in range(self.index):
            x = x_list[i]
            x = self.parallel_conv_lists[i](x)
            parallel_res_list.append(x)

        # Fuse multi‑resolution features following HRNet sum‑then‑ReLU recipe
        final_res_list = []
        for i in range(self.index if self.last_stage else self.index + 1):
            if i == self.index:
                # Create an *extra* low‑res branch in non‑final stages by
                # down‑sampling and summing all existing high‑res tensors.
                x = 0
                for t, m in zip(parallel_res_list, self.down_conv_lists[-1]):
                    x += m(t)
            else:
                # Start with native tensor of the branch
                x = parallel_res_list[i]

                # Add contributions from *lower‑resolution* neighbours (up‑sample)
                if i != self.index - 1:
                    res_list = parallel_res_list[i + 1:]
                    up_x = 0
                    for t, m in zip(res_list, self.up_conv_lists[i]):
                        up_x += m(t)
                    x = x + up_x

                # Add contributions from *higher‑resolution* neighbours (down‑sample)
                if i != 0:
                    res_list = parallel_res_list[:i]
                    down_x = 0
                    for t, m in zip(res_list, self.down_conv_lists[i - 1]):
                        down_x += m(t)
                    x = x + down_x
            x = self.relu(x)
            final_res_list.append(x)
        return final_res_list

# ---------------------------------------------------------------------------
#                          4.  FULL HRNet TOPOLOGY
# ---------------------------------------------------------------------------

class HRNet(nn.Module):
    """Compact HRNet tailored for segmentation / landmark detection tasks."""

    def __init__(self, in_ch, mid_ch, out_ch, num_stage=4):
        super(HRNet, self).__init__()
        # ---------------------------- Stem ----------------------------------
        self.init_conv = nn.Sequential(
            Conv(in_ch, 64, 1),
            Conv(64, 64, 1)
        )

        # --------------------------- Head (final classifier) -----------------
        self.head = nn.Sequential(
            Conv(mid_ch * (1 + 2 + 4 + 8), mid_ch * (1 + 2 + 4 + 8), 1),
            nn.Conv2d(mid_ch * (1 + 2 + 4 + 8), out_ch, 1)
        )

        # --------------------------- Stage 0 ---------------------------------
        self.first_layer = self._make_layer(64, 64, Bottleneck, 4)
        self.first_transition = self._make_transition_layer(256, mid_ch, 1)

        # ------------------------- Subsequent stages -------------------------
        self.num_stage = num_stage
        self.hr_blocks = nn.ModuleList()
        for i in range(1, num_stage):
            self.hr_blocks.append(
                HRBlock(mid_ch, i + 1, True if i == num_stage - 1 else False, BasicBlock)
            )

        # Upsample lower‑resolution outputs to match highest resolution before concat
        self.up_samplings = nn.ModuleList()
        for i in range(num_stage - 1):
            up_factor = 2 ** (i + 1)
            up = nn.Upsample(scale_factor=up_factor, mode='bilinear')
            self.up_samplings.append(up)

        # ---------------------- Weight initialisation ------------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    #   Helper factory functions (unchanged except docstrings added)
    # ------------------------------------------------------------------
    def _make_layer(self, in_ch, ch, block, num):
        """Create a stack of *num* residual blocks with optional down‑sampling."""
        downsampling = None
        if in_ch != ch * block.expansion:
            downsampling = Conv(in_ch, ch * block.expansion, 1, relued=False)
        layers = []
        layers.append(block(in_ch, ch, downsampling))
        for i in range(1, num):
            layers.append(block(ch * block.expansion, ch))
        return nn.Sequential(*layers)

    def _make_transition_layer(self, in_ch, out_ch, stage):
        """Transition from stem/previous stage to first HRBlock resolution list."""
        layers = nn.ModuleList()
        layers.append(Conv(in_ch, out_ch, 1))           # keep original res
        layers.append(Conv(in_ch, out_ch * 2, 3, 2))    # ½ spatial res
        return layers

    # ------------------------------------------------------------------
    #               Loss helper (mean‑squared‑error with dict API)
    # ------------------------------------------------------------------
    @staticmethod
    def get_loss(prediction, target):
        """Compute per‑batch L2 loss (sum of squared errors)."""
        err = (prediction - target) ** 2
        loss_dict = {'l2': err.sum() / err.shape[0]}
        return loss_dict

    # ------------------------------------------------------------------
    #                         Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_, target_=None):
        # -------------------------------------------------------------
        # 1) Format input (caller provides list of tensors)
        # -------------------------------------------------------------
        x = torch.stack(input_, dim=0)

        # -------------------------------------------------------------
        # 2) Stem + Stage 0
        # -------------------------------------------------------------
        x = self.init_conv(x)
        x = self.first_layer(x)
        # Produce first multi‑resolution list (2 resolutions)
        x_list = [m(x) for m in self.first_transition]

        # -------------------------------------------------------------
        # 3) High‑resolution stages (HRBlocks)
        # -------------------------------------------------------------
        for i in range(self.num_stage - 1):
            x_list = self.hr_blocks[i](x_list)

        # -------------------------------------------------------------
        # 4) Aggregate outputs from all resolutions to highest res
        # -------------------------------------------------------------
        res_list = [x_list[0]]
        for t, m in zip(x_list[1:], self.up_samplings):
            res_list.append(m(t))
        x = torch.cat(res_list, dim=1)

        # -------------------------------------------------------------
        # 5) Final head – 1×1 Conv + output
        # -------------------------------------------------------------
        x = self.head(x)

        # -------------------------------------------------------------
        # 6) Loss computation (optional)
        # -------------------------------------------------------------
        if target_ is None:
            return x
        else:
            target = torch.stack([t['masks'] for t in target_], dim=0)
            loss_dict = self.get_loss(x, target)
            return loss_dict
