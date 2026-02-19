"""Markdown processing utilities."""

import re
import ast
from pathlib import Path
from typing import List, Tuple

from PIL import Image
from glmocr.utils.image_utils import (
    crop_image_region,
    pdf_to_images_pil,
    PYPDFIUM2_AVAILABLE,
)
from glmocr.utils.logging import get_logger

logger = get_logger(__name__)


def extract_image_refs(markdown_text: str) -> List[Tuple[int, List[int], str]]:
    """Extract image references from Markdown.

    Args:
        markdown_text: Markdown text.

    Returns:
        List of (page_idx, bbox, original_tag).
    """
    # Pattern: ![](page=0,bbox=[57, 199, 884, 444])
    pattern = r"!\[\]\(page=(\d+),bbox=(\[[\d,\s]+\])\)"
    matches = re.finditer(pattern, markdown_text)

    image_refs = []
    for match in matches:
        page_idx = int(match.group(1))
        bbox_str = match.group(2)
        # Parse bbox string safely
        try:
            bbox = ast.literal_eval(bbox_str)
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox format: {bbox_str}")
        except (ValueError, SyntaxError) as e:
            logger.warning("Cannot parse bbox %s: %s", bbox_str, e)
            continue
        original_tag = match.group(0)
        image_refs.append((page_idx, bbox, original_tag))

    return image_refs


def crop_and_replace_images(
    markdown_text: str,
    original_images: List[str],
    output_dir: Path,
    image_prefix: str = "image",
) -> Tuple[str, List[str]]:
    """Crop referenced image regions and replace Markdown tags.

    Args:
        markdown_text: Source Markdown.
        original_images: Original image paths.
        output_dir: Output directory.
        image_prefix: Filename prefix for cropped images.

    Returns:
        (updated_markdown, saved_image_paths)
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract image references
    image_refs = extract_image_refs(markdown_text)

    if not image_refs:
        # No image references
        return markdown_text, []

    # Load originals (supports PDFs)
    loaded_images = []
    for img_path in original_images:
        path = Path(img_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            # PDF: convert to images (pypdfium2 only)
            if not PYPDFIUM2_AVAILABLE:
                raise RuntimeError(
                    "PDF support requires pypdfium2. Install: pip install pypdfium2"
                )
            try:
                pdf_images = pdf_to_images_pil(
                    img_path, dpi=200, max_width_or_height=3500
                )
                loaded_images.extend(pdf_images)
            except Exception as e:
                raise RuntimeError(f"Failed to convert PDF to images: {e}") from e
        else:
            # Normal image file
            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            loaded_images.append(img)

    # Process each reference
    result_markdown = markdown_text
    saved_image_paths = []

    for idx, (page_idx, bbox, original_tag) in enumerate(image_refs):
        # Validate page index
        if page_idx < 0 or page_idx >= len(loaded_images):
            logger.warning(
                "page_idx %d out of range (total %d images), skipping",
                page_idx,
                len(loaded_images),
            )
            continue

        # Crop from original
        original_image = loaded_images[page_idx]
        try:
            cropped_image = crop_image_region(original_image, bbox)

            # Output filename format: image_page0_idx0.jpg
            image_filename = f"{image_prefix}_page{page_idx}_idx{idx}.jpg"
            image_path = output_dir / image_filename

            # Save cropped image
            cropped_image.save(image_path, quality=95)
            saved_image_paths.append(str(image_path))

            # Replace Markdown image tag with a relative path (imgs/filename)
            relative_path = f"imgs/{image_filename}"
            new_tag = f"![Image {page_idx}-{idx}]({relative_path})"
            result_markdown = result_markdown.replace(original_tag, new_tag, 1)

        except Exception as e:
            logger.warning(
                "Failed to crop image (page=%d, bbox=%s): %s", page_idx, bbox, e
            )
            # Keep original tag on failure
            continue

    return result_markdown, saved_image_paths
