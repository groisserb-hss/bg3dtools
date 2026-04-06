"""
Object detection utilities for PyTorch.

This package provides training and evaluation utilities for object detection
models, including COCO dataset handling and evaluation metrics.

Functions
---------
train_one_epoch
    Train model for one epoch.
evaluate
    Evaluate model on COCO-style dataset.

Classes
-------
CocoEvaluator
    COCO evaluation wrapper.
"""

from .engine import train_one_epoch, evaluate
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset

__all__ = [
    "train_one_epoch",
    "evaluate",
    "CocoEvaluator",
    "get_coco_api_from_dataset",
]
