"""Base post-processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Dict

if TYPE_CHECKING:
    from glmocr.config import ResultFormatterConfig


class BasePostProcessor:
    """Base post-processor.

    Returns recognition results with a stable ordering. Subclasses can implement
    richer formatting logic.
    """

    def __init__(self, config: "ResultFormatterConfig"):
        """Initialize.

        Args:
            config: ResultFormatterConfig instance.
        """
        self.config = config

    def process(self, results: List[Dict]) -> List[Dict]:
        """Process recognition results.

        Args:
            results: Recognition results.

        Returns:
            List[Dict]: Processed results.
        """
        # Ensure stable ordering by index
        sorted_results = sorted(results, key=lambda x: x.get("index", 0))

        # Additional post-processing can be added here, e.g.:
        # - merge adjacent text blocks
        # - filter low-confidence results
        # - output formatting

        return sorted_results
