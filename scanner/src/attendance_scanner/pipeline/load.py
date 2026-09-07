"""Safe image loader with EXIF orientation normalization and alpha channel flattening.

This module is the first stage in the document scanning pipeline. It reads source images
using Pillow, applies EXIF orientation transpose before any Computer Vision processing,
flattens transparent pixels onto a pure white background, and converts the resulting
RGB image into an OpenCV-compatible BGR NumPy array.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from attendance_scanner.contracts import BaseContract, ImageDecodeError

# Supported image file extensions for the scanner pipeline
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class LoadedImageMetadata(BaseContract):
    """Structured metadata describing the loaded image before and after normalization."""

    original_width: int
    original_height: int
    width: int
    height: int
    original_mode: str
    normalized_mode: str = "BGR"
    exif_orientation: Optional[int] = None
    has_transparency: bool = False


@dataclass
class LoadedImage:
    """Container holding the OpenCV BGR image array and associated normalization metadata."""

    path: Path
    image: np.ndarray  # Shape: (height, width, 3), dtype: np.uint8, color order: BGR
    metadata: LoadedImageMetadata

    @property
    def width(self) -> int:
        """Normalized image width in pixels."""
        return self.metadata.width

    @property
    def height(self) -> int:
        """Normalized image height in pixels."""
        return self.metadata.height

    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape (height, width, channels)."""
        return self.image.shape

    @property
    def channels(self) -> int:
        """Number of color channels (guaranteed 3 for BGR)."""
        return 3 if self.image.ndim == 3 else 1


def load_image(file_path: Union[str, Path]) -> LoadedImage:
    """Safely load an image from disk, normalize orientation, composite alpha, and convert to BGR.

    Pipeline sequence:
    1. Validate path existence, file type, and non-empty size.
    2. Read image with Pillow inside a managed context (closing file handles immediately).
    3. Extract raw EXIF orientation tag (0x0112) if present.
    4. Apply `ImageOps.exif_transpose` to orient pixels upright according to camera sensor data.
    5. Composite alpha channels (RGBA, LA, transparent palette) onto a pure white background.
    6. Convert from RGB to OpenCV BGR array (`np.ndarray`).
    7. Wrap any decode, I/O, or corruption errors in typed `ImageDecodeError`.

    Args:
        file_path: Absolute or relative path to the image file.

    Returns:
        LoadedImage with normalized OpenCV BGR array and metadata.

    Raises:
        ImageDecodeError: When the file does not exist, is not an image, is empty,
                          is corrupted, or cannot be decoded.
    """
    path = Path(file_path)

    # 1. Validate file existence and extension
    if not path.exists() or not path.is_file():
        raise ImageDecodeError(str(path), "File does not exist or is not a regular file")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ImageDecodeError(
            str(path),
            f"Unsupported image extension '{path.suffix}'. Expected one of: "
            f"{', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}",
        )

    # Check for empty file
    try:
        if path.stat().st_size == 0:
            raise ImageDecodeError(str(path), "Image file is empty (0 bytes)")
    except OSError as stat_err:
        raise ImageDecodeError(str(path), f"Cannot stat image file: {stat_err}") from stat_err

    # 2. Decode image with Pillow
    try:
        with Image.open(path) as raw_img:
            # Force Pillow to decode pixel data into memory so the file handle can close
            raw_img.load()

            orig_w, orig_h = raw_img.size
            orig_mode = raw_img.mode

            # 3. Read EXIF orientation tag (tag 0x0112) before transposition
            raw_orientation: Optional[int] = None
            try:
                exif = raw_img.getexif()
                if exif:
                    val = exif.get(0x0112)
                    if isinstance(val, int) and 1 <= val <= 8:
                        raw_orientation = val
            except Exception:
                # Non-fatal: ignore unparseable EXIF metadata
                raw_orientation = None

            # 4. Apply EXIF orientation transpose before any CV processing
            # ImageOps.exif_transpose automatically rotates / flips the image upright
            transposed_img = ImageOps.exif_transpose(raw_img)
            if transposed_img is None:
                transposed_img = raw_img.copy()

            # 5. Normalize alpha channel onto solid white background
            # Document scanning rationale:
            # Attendance sheets and document scans are printed on white paper.
            # Transparent pixels (e.g. from transparent PNG/WEBP signatures or scans)
            # are composited onto pure white (255, 255, 255) background so that document
            # edge detection, adaptive thresholding, and PDF rendering treat transparent
            # areas as clean paper instead of black voids or arbitrary noise.
            has_transparency = False

            if transposed_img.mode in ("RGBA", "LA") or (
                transposed_img.mode == "P" and "transparency" in transposed_img.info
            ):
                has_transparency = True
                rgba = transposed_img.convert("RGBA")
                white_canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                # Alpha composite: layer image over white background
                composite = Image.alpha_composite(white_canvas, rgba)
                rgb_img = composite.convert("RGB")
            elif transposed_img.mode == "RGB":
                rgb_img = transposed_img
            else:
                # Modes like "L" (grayscale), "1" (monochrome), "CMYK", etc.
                rgb_img = transposed_img.convert("RGB")

            norm_w, norm_h = rgb_img.size

            # 6. Convert PIL RGB image to OpenCV BGR NumPy array
            rgb_arr = np.asarray(rgb_img, dtype=np.uint8)
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    except ImageDecodeError:
        raise
    except Exception as exc:
        raise ImageDecodeError(str(path), f"Failed to decode image data: {exc}") from exc

    metadata = LoadedImageMetadata(
        original_width=orig_w,
        original_height=orig_h,
        width=norm_w,
        height=norm_h,
        original_mode=orig_mode,
        normalized_mode="BGR",
        exif_orientation=raw_orientation,
        has_transparency=has_transparency,
    )

    return LoadedImage(
        path=path,
        image=bgr_arr,
        metadata=metadata,
    )
