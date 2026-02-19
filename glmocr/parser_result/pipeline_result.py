"""Pipeline result with layout visualization support."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Optional, Union

from glmocr.utils.logging import get_logger

from .base import BaseParserResult

logger = get_logger(__name__)


class PipelineResult(BaseParserResult):
    """Pipeline result for one input unit (one image or one PDF).

    Supports saving JSON, Markdown, and optional layout visualization.
    """

    def __init__(
        self,
        json_result: Union[str, dict, list],
        markdown_result: Optional[str],
        original_images: List[str],
        layout_vis_dir: Optional[str] = None,
        layout_image_indices: Optional[List[int]] = None,
    ):
        """Initialize.

        Args:
            json_result: JSON result (string, dict, or list).
            markdown_result: Markdown result.
            original_images: Original image paths for this unit.
            layout_vis_dir: Temp dir with layout_page{N}.jpg (optional).
            layout_image_indices: Indices of layout pages belonging to this unit;
                None means all files in layout_vis_dir belong to this unit.
        """
        super().__init__(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=original_images,
        )
        self.layout_vis_dir = layout_vis_dir
        self.layout_image_indices = layout_image_indices
        self._layout_vis_saved = False

    def save(
        self,
        output_dir: Union[str, Path] = "./results",
        save_layout_visualization: bool = True,
    ) -> None:
        """Save JSON, Markdown, and optionally layout visualization."""
        self._save_json_and_markdown(output_dir)

        if (
            not save_layout_visualization
            or not self.layout_vis_dir
            or self._layout_vis_saved
        ):
            return

        temp_layout_path = Path(self.layout_vis_dir)
        if not temp_layout_path.exists():
            return

        if self.original_images:
            stem = Path(self.original_images[0]).stem
            target_dir = Path(output_dir).absolute() / stem / "layout_vis"
        else:
            target_dir = Path(output_dir).absolute() / "result" / "layout_vis"

        target_dir.mkdir(parents=True, exist_ok=True)

        if self.layout_image_indices is not None:
            layout_files = []
            for idx in self.layout_image_indices:
                for ext in (".jpg", ".png"):
                    p = temp_layout_path / f"layout_page{idx}{ext}"
                    if p.exists():
                        layout_files.append(p)
                        break
        else:
            layout_files = sorted(temp_layout_path.glob("layout_page*.jpg"))
            layout_files.extend(sorted(temp_layout_path.glob("layout_page*.png")))

        stem = Path(self.original_images[0]).stem if self.original_images else "result"
        for layout_file in layout_files:
            m = re.match(
                r"layout_page(\d+)\.(jpg|png)$",
                layout_file.name,
                re.IGNORECASE,
            )
            if m:
                idx_str, ext = m.group(1), m.group(2).lower()
            else:
                idx_str, ext = "0", layout_file.suffix.lstrip(".") or "jpg"
            new_name = (
                f"{stem}.{ext}"
                if len(layout_files) == 1
                else f"{stem}_page{idx_str}.{ext}"
            )
            target_file = target_dir / new_name
            shutil.move(str(layout_file), str(target_file))

        if self.layout_image_indices is None:
            try:
                temp_layout_path.rmdir()
            except Exception:
                pass

        self._layout_vis_saved = True
        logger.debug("Layout visualization saved to %s", target_dir)
