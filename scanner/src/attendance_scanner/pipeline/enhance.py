"""Scan enhancement filters for document images (Gray, B&W, and Color Enhanced).

This module is the fourth stage of the document scanning pipeline. Operating as pure
in-memory functions on OpenCV images (without filesystem I/O), it applies contrast
enhancement, noise reduction, and binarization according to the selected ScanMode:
1. Gray (default): Grayscale + CLAHE + edge-preserving bilateral denoise + mild unsharp mask.
   Preserves faint pen strokes and table grid lines.
2. B&W: Grayscale + background normalization + Gaussian adaptive thresholding.
   Crisp binary output for clean dark text (warning: faint handwriting may lose fidelity).
3. Color Enhanced: LAB color-space CLAHE strictly on Luminance (L) + unsharp mask.
   Preserves stamps and colored signatures without color cast distortion.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field, field_validator, model_validator

from attendance_scanner.contracts import BaseContract, ScanMode
from attendance_scanner.pipeline.load import LoadedImage
from attendance_scanner.pipeline.perspective import WarpedDocument


class EnhancementConfig(BaseContract):
    """Configurable parameters and threshold bounds for scan enhancement filters."""

    # Gray mode parameters
    gray_clahe_clip_limit: float = Field(default=2.0, gt=0.0, le=20.0)
    gray_clahe_tile_grid: Tuple[int, int] = (8, 8)
    gray_denoise_d: int = Field(default=5, ge=1, le=15)
    gray_sharpen_amount: float = Field(default=0.3, ge=0.0, le=2.0)
    gray_sharpen_sigma: float = Field(default=1.0, gt=0.0, le=5.0)

    # B&W mode parameters
    bw_adaptive_block_size: int = Field(default=21, ge=3, le=101)
    bw_adaptive_c: int = Field(default=10, ge=-30, le=50)
    bw_bg_normalize: bool = True
    bw_bg_kernel_size: int = Field(default=25, ge=3, le=101)

    # Color Enhanced mode parameters
    color_clahe_clip_limit: float = Field(default=1.5, gt=0.0, le=20.0)
    color_clahe_tile_grid: Tuple[int, int] = (8, 8)
    color_sharpen_amount: float = Field(default=0.2, ge=0.0, le=2.0)
    color_sharpen_sigma: float = Field(default=1.0, gt=0.0, le=5.0)

    @field_validator("gray_clahe_tile_grid", "color_clahe_tile_grid")
    @classmethod
    def validate_tile_grid(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure CLAHE tile grid dimensions are positive integers > 0."""
        if len(v) != 2 or v[0] <= 0 or v[1] <= 0:
            raise ValueError(f"CLAHE tile grid dimensions must be positive integers > 0, got {v}")
        return v

    @model_validator(mode="after")
    def validate_odd_kernel_sizes(self) -> "EnhancementConfig":
        """Ensure block and kernel sizes are odd numbers for OpenCV requirements."""
        if self.bw_adaptive_block_size % 2 == 0:
            self.bw_adaptive_block_size += 1
        if self.bw_bg_kernel_size % 2 == 0:
            self.bw_bg_kernel_size += 1
        return self


def _extract_bgr_array(image: Union[np.ndarray, WarpedDocument, LoadedImage]) -> np.ndarray:
    """Extract raw uint8 NumPy array from ndarray, WarpedDocument, or LoadedImage."""
    if isinstance(image, WarpedDocument):
        arr = image.image
    elif isinstance(image, LoadedImage):
        arr = image.image
    elif isinstance(image, np.ndarray):
        arr = image
    else:
        raise ValueError(f"Expected np.ndarray, WarpedDocument, or LoadedImage, got {type(image)}")

    if arr.dtype != np.uint8 or arr.ndim not in (2, 3):
        raise ValueError(
            f"Expected uint8 array with 2 or 3 dimensions, got dtype={arr.dtype}, ndim={arr.ndim}"
        )

    if arr.ndim == 3 and arr.shape[2] != 3:
        raise ValueError(f"Expected 3 color channels for 3D array, got shape {arr.shape}")

    return arr


def enhance_gray(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Enhance image in Grayscale mode (recommended default for attendance documents).

    Pipeline:
    1. Grayscale conversion.
    2. CLAHE (Contrast-Limited Adaptive Histogram Equalization) to balance page lighting.
    3. Bilateral filter for edge-preserving noise reduction.
    4. Mild unsharp masking to enhance text and table borders without heavy halos.

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        config: Optional configuration thresholds.

    Returns:
        1-channel 2D NumPy array of shape (height, width) with dtype uint8.
    """
    if config is None:
        config = EnhancementConfig()

    bgr = _extract_bgr_array(image)

    # Convert to single-channel grayscale
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return gray

    # 1. CLAHE adaptive contrast enhancement
    clahe = cv2.createCLAHE(
        clipLimit=config.gray_clahe_clip_limit,
        tileGridSize=config.gray_clahe_tile_grid,
    )
    equalized = clahe.apply(gray)

    # 2. Bilateral filtering for edge-preserving denoising
    # Using small spatial sigma preserves fine handwriting while smoothing paper grain
    denoised = cv2.bilateralFilter(
        equalized,
        d=config.gray_denoise_d,
        sigmaColor=25.0,
        sigmaSpace=25.0,
    )

    # 3. Mild unsharp mask sharpening
    if config.gray_sharpen_amount > 0.0:
        blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=config.gray_sharpen_sigma)
        sharpened = cv2.addWeighted(
            denoised,
            1.0 + config.gray_sharpen_amount,
            blurred,
            -config.gray_sharpen_amount,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    return denoised


def enhance_bw(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Enhance image in B&W binary mode (clean text documents with dark ink).

    Document Scanning Note:
    B&W mode applies high-contrast adaptive binarization which maximizes text crispness
    on printed tables and dark ink. However, faint ballpoint pen marks, light pencil notes,
    or faded signatures may be eroded. Gray mode is recommended when faint handwriting
    fidelity is critical.

    Pipeline:
    1. Grayscale conversion.
    2. Background normalization via morphological dilation + blur to level shadow gradients.
    3. Gaussian adaptive thresholding using a validated odd block size.

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        config: Optional configuration thresholds.

    Returns:
        1-channel 2D binary NumPy array of shape (height, width) with values in {0, 255}.
    """
    if config is None:
        config = EnhancementConfig()

    bgr = _extract_bgr_array(image)

    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr.copy()

    h, w = gray.shape[:2]
    if h < 2 or w < 2:
        return np.where(gray >= 128, 255, 0).astype(np.uint8)

    # 1. Background normalization
    if config.bw_bg_normalize:
        k = config.bw_bg_kernel_size
        k = min(k, max(3, (min(h, w) // 4) * 2 + 1))  # Prevent kernel larger than image
        struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, struct_elem)
        bg = cv2.medianBlur(bg, k)
        # Avoid division by zero on black background pixels
        bg = np.maximum(bg, 1)
        normalized = cv2.divide(gray, bg, scale=255)
    else:
        normalized = gray

    # 2. Gaussian adaptive thresholding
    block_size = config.bw_adaptive_block_size
    # Ensure block size is strictly odd and smaller than min image dimension
    min_dim = min(h, w)
    if block_size >= min_dim:
        block_size = max(3, (min_dim // 2) * 2 + 1)
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        normalized,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        config.bw_adaptive_c,
    )

    return binary


def enhance_color(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Enhance image in Color mode, preserving official stamps and colored signatures.

    Pipeline:
    1. Convert BGR to CIE LAB color space.
    2. Apply CLAHE strictly to the Luminance (L) channel to enhance contrast without color shift.
    3. Recombine L, A, B channels and convert back to BGR.
    4. Apply mild unsharp masking to enhance fine text edges.

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        config: Optional configuration thresholds.

    Returns:
        3-channel BGR NumPy array of shape (height, width, 3) with dtype uint8.
    """
    if config is None:
        config = EnhancementConfig()

    bgr = _extract_bgr_array(image)

    # If single-channel gray, convert to 3-channel BGR
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    h, w = bgr.shape[:2]
    if h < 2 or w < 2:
        return bgr.copy()

    # 1. Convert to LAB color space
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    # 2. CLAHE only on Luminance channel
    clahe = cv2.createCLAHE(
        clipLimit=config.color_clahe_clip_limit,
        tileGridSize=config.color_clahe_tile_grid,
    )
    enhanced_l = clahe.apply(l_chan)

    # 3. Recombine channels
    merged_lab = cv2.merge([enhanced_l, a_chan, b_chan])
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

    # 4. Mild unsharp mask sharpening on BGR
    if config.color_sharpen_amount > 0.0:
        blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=config.color_sharpen_sigma)
        sharpened = cv2.addWeighted(
            enhanced_bgr,
            1.0 + config.color_sharpen_amount,
            blurred,
            -config.color_sharpen_amount,
            0,
        )
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    return enhanced_bgr


def enhance_image(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Enhance a document image according to the specified ScanMode (Gray, B&W, or Color).

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        mode: Enhancement filter mode (ScanMode.GRAY, ScanMode.BW, or ScanMode.COLOR).
        config: Optional configuration parameters. Uses defaults if omitted.

    Returns:
        Enhanced NumPy array: 1-channel uint8 (H, W) for Gray and B&W,
        3-channel uint8 (H, W, 3) for Color.

    Raises:
        ValueError: If an unsupported mode or invalid image format is passed.
    """
    # Normalize mode string or enum
    mode_str = mode.value.lower() if isinstance(mode, ScanMode) else str(mode).lower()

    if mode_str == ScanMode.GRAY.value:
        return enhance_gray(image, config)
    elif mode_str == ScanMode.BW.value:
        return enhance_bw(image, config)
    elif mode_str in (ScanMode.COLOR.value, "color_enhanced", "color-enhanced"):
        return enhance_color(image, config)
    else:
        raise ValueError(
            f"Unsupported scan enhancement mode '{mode}'. "
            f"Expected one of: {[m.value for m in ScanMode]}"
        )
