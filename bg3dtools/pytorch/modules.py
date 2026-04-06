"""
PyTorch neural network modules.

This module provides reusable neural network building blocks for deep learning.
"""

import torch
from torch import nn, Tensor


class LinearResBlock(nn.Module):
    """
    Residual block with linear layers.

    A two-layer MLP with batch normalization and LeakyReLU activations,
    combined with a skip connection.

    Parameters
    ----------
    dimensions : int
        Number of input and output features.
    """
    def __init__(self, dimensions: int) -> None:
        nn.Module.__init__(self)
        self.block = nn.Sequential(
            nn.Linear(dimensions, dimensions),
            nn.BatchNorm1d(dimensions),
            nn.LeakyReLU(),
            nn.Linear(dimensions, dimensions),
            nn.BatchNorm1d(dimensions),
            nn.LeakyReLU())

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)

