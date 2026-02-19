"""Layout detection base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Dict
from PIL import Image

if TYPE_CHECKING:
    from glmocr.config import LayoutConfig


class BaseLayoutDetector(ABC):
    """Base class for layout detectors.

    Defines a unified interface for layout detection.
    """

    def __init__(self, config: "LayoutConfig"):
        """Initialize.

        Args:
            config: LayoutConfig instance.
        """
        self.config = config

    @abstractmethod
    def process(self, images: List[Image.Image]) -> List[List[Dict]]:
        """Process images (unified interface).

        Batch-detect layout regions.

        Args:
            images: List of PIL Images.

        Returns:
            List[List[Dict]]: detections per image.
                Each detection contains:
                - index: region index
                - label: original label
                - score: confidence
                - bbox_2d: normalized coordinates [x1, y1, x2, y2] (0-1000)
        """

    @abstractmethod
    def start(self):
        """Start the detector (e.g., start worker processes)."""

    @abstractmethod
    def stop(self):
        """Stop the detector (e.g., stop worker processes)."""
