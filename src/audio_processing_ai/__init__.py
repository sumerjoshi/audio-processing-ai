"""
Audio Processing AI Package

A deep learning package for analyzing audio files and detecting AI-generated content.
"""

__version__ = "0.1.0"
__author__ = "Audio Processing AI Team"

from .dataset import AIAudioDataset
from .model.pretrained.dual_head_cnn14 import DualHeadCnn14Simple

__all__ = [
    "AIAudioDataset",
    "DualHeadCnn14Simple",
]
