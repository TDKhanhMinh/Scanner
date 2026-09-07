"""Atomic single-page PDF export primitive for attendance sheets.

Given a processed document image (NumPy array, SingleScanResult, or PIL Image),
this module renders a standards-compliant single-page PDF, saves it to a unique
temporary file in the target directory, and atomically commits it using `os.replace`.
Failure to save cleanly unlinks the temporary file without corrupting any existing target.
"""

import os
import uuid
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import Field

from attendance_scanner.contracts import BaseContract, PdfWriteError
from attendance_scanner.pipeline.orchestrator import SingleScanResult


class PdfExportConfig(BaseContract):
    """Configuration options for single-page PDF export."""

    dpi: float = Field(default=300.0, gt=10.0, le=1200.0)
    quality: int = Field(default=95, ge=1, le=100)
    optimize: bool = True


class PdfExportResult(BaseContract):
    """Result metadata for a committed single-page PDF file."""

    pdf_path: str
    file_size_bytes: int
    page_count: int = 1
    width_px: int
    height_px: int
    mode: str


def _prepare_pil_image(
    image: Union[np.ndarray, SingleScanResult, Image.Image],
) -> Image.Image:
    """Extract and format input into a PIL Image with proper color channels."""
    if isinstance(image, SingleScanResult):
        arr = image.image
    elif isinstance(image, Image.Image):
        return image
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. Expected np.ndarray, "
            "SingleScanResult, or PIL.Image."
        )

    if arr.dtype != np.uint8:
        raise ValueError(f"NumPy image array must have dtype=uint8, got {arr.dtype}")

    if arr.ndim == 2:
        # Grayscale or B&W single-channel
        return Image.fromarray(arr, mode="L")

    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            return Image.fromarray(arr[:, :, 0], mode="L")
        if channels == 3:
            # OpenCV BGR -> Pillow RGB
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb, mode="RGB")
        if channels == 4:
            # BGRA -> RGB
            rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            return Image.fromarray(rgb, mode="RGB")
        raise ValueError(f"Unsupported number of channels in 3D array: {channels}")

    raise ValueError(f"Unsupported NumPy array dimensions: {arr.ndim} (shape {arr.shape})")


def export_single_page_pdf(
    image: Union[np.ndarray, SingleScanResult, Image.Image],
    target_path: Union[str, Path],
    *,
    config: Optional[PdfExportConfig] = None,
) -> PdfExportResult:
    """Render a processed image into a single-page PDF and commit atomically.

    Guarantees:
    - Lazy creation of parent target directory.
    - Temporary file created in the exact same directory for atomic rename on the same filesystem.
    - Successful atomic replacement of existing PDF at target_path.
    - Best-effort cleanup of temporary file upon failure, leaving target_path untouched.
    - Exactly 1 PDF page.

    Args:
        image: Processed image (uint8 NumPy array, SingleScanResult, or PIL Image).
        target_path: Destination filesystem path for the final committed PDF.
        config: Optional PDF export configuration (DPI, quality, optimize).

    Returns:
        PdfExportResult describing the written PDF.

    Raises:
        PdfWriteError: If image conversion, PDF encoding, or filesystem commit fails.
    """
    cfg = config or PdfExportConfig()
    dest = Path(target_path).resolve()

    try:
        pil_img = _prepare_pil_image(image)
    except Exception as exc:
        raise PdfWriteError(str(dest), f"Image preparation failed: {exc}") from exc

    # 1. Lazily create destination parent directory
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise PdfWriteError(str(dest), f"Failed to create parent directory: {exc}") from exc

    # 2. Allocate unique temporary file in the same directory for atomic commit
    temp_id = uuid.uuid4().hex[:12]
    temp_file = dest.parent / f".{dest.name}.tmp.{temp_id}"

    try:
        # 3. Save PDF using Pillow's native PDF writer
        pil_img.save(
            temp_file,
            format="PDF",
            resolution=cfg.dpi,
            quality=cfg.quality,
            optimize=cfg.optimize,
            title=dest.stem,
        )

        # 4. Verify temporary file was created and has non-zero size
        if not temp_file.exists():
            raise PdfWriteError(str(dest), "Temporary PDF file was not created on disk")
        stat = temp_file.stat()
        if stat.st_size == 0:
            raise PdfWriteError(str(dest), "Generated temporary PDF file is 0 bytes")

        # 5. Atomically replace target path
        os.replace(temp_file, dest)

    except Exception as exc:
        # Best-effort cleanup of temporary file
        try:
            if temp_file.exists():
                temp_file.unlink()
        except OSError:
            pass

        if isinstance(exc, PdfWriteError):
            raise
        raise PdfWriteError(str(dest), f"Failed to commit PDF: {exc}") from exc

    final_size = dest.stat().st_size
    w_px, h_px = pil_img.size

    return PdfExportResult(
        pdf_path=str(dest),
        file_size_bytes=final_size,
        page_count=1,
        width_px=w_px,
        height_px=h_px,
        mode=pil_img.mode,
    )
