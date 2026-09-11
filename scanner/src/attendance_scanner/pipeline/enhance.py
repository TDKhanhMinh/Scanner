"""Scan enhancement filters for document images (Gray, B&W, Color, and Smart Document).

This module is the fourth stage of the document scanning pipeline. Operating as pure
in-memory functions on OpenCV images (without filesystem I/O), it applies contrast
enhancement, noise reduction, and binarization according to the selected ScanMode:
1. Gray (default): Grayscale + CLAHE + edge-preserving bilateral denoise + mild unsharp mask.
   Preserves faint pen strokes and table grid lines.
2. B&W (Magic Pro): CamScanner-style background flattening + deep black text + colored ink preservation.
   Crisp white paper, dark text, and vibrant signatures/stamps without shadow casts.
3. Color Enhanced: LAB color-space CLAHE strictly on Luminance (L) + unsharp mask.
   Preserves stamps and colored signatures without color cast distortion.
4. Smart Document: background illumination correction + paper whitening + LAB contrast.
   Targets clean CamScanner-style output while preserving colored handwriting and stamps.
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

    # B&W mode parameters (legacy fields preserved for backward compatibility)
    bw_adaptive_block_size: int = Field(default=21, ge=3, le=101)
    bw_adaptive_c: int = Field(default=10, ge=-30, le=50)
    bw_bg_normalize: bool = True
    bw_bg_kernel_size: int = Field(default=25, ge=3, le=101)
    # B&W Magic Pro parameters
    bw_magic_kernel_size: int = Field(default=61, ge=9, le=201)
    bw_magic_black_point: int = Field(default=130, ge=0, le=255)
    bw_magic_white_point: int = Field(default=215, ge=0, le=255)
    bw_magic_sat_boost: float = Field(default=1.3, ge=0.5, le=3.0)
    bw_magic_sharpen: float = Field(default=0.3, ge=0.0, le=2.0)

    # Color Enhanced mode parameters
    color_clahe_clip_limit: float = Field(default=1.5, gt=0.0, le=20.0)
    color_clahe_tile_grid: Tuple[int, int] = (8, 8)
    color_sharpen_amount: float = Field(default=0.2, ge=0.0, le=2.0)
    color_sharpen_sigma: float = Field(default=1.0, gt=0.0, le=5.0)

    # Smart Document mode parameters
    smart_background_kernel_size: int = Field(default=51, ge=9, le=201)
    smart_clahe_clip_limit: float = Field(default=1.6, gt=0.0, le=20.0)
    smart_clahe_tile_grid: Tuple[int, int] = (8, 8)
    smart_paper_l_threshold: int = Field(default=165, ge=0, le=255)
    smart_neutralize_strength: float = Field(default=0.65, ge=0.0, le=1.0)
    smart_denoise_d: int = Field(default=5, ge=1, le=15)
    smart_sharpen_amount: float = Field(default=0.25, ge=0.0, le=2.0)
    smart_sharpen_sigma: float = Field(default=1.0, gt=0.0, le=5.0)

    @field_validator(
        "gray_clahe_tile_grid",
        "color_clahe_tile_grid",
        "smart_clahe_tile_grid",
    )
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
        if self.bw_magic_kernel_size % 2 == 0:
            self.bw_magic_kernel_size += 1
        if self.smart_background_kernel_size % 2 == 0:
            self.smart_background_kernel_size += 1
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
    """Enhance image in CamScanner Magic Pro mode (replacing legacy adaptive binarization).

    CamScanner Magic Pro:
    1. Multi-channel morphological background illumination map (dilation + Gaussian blur)
       to completely eliminate uneven lighting, shadows, and yellowish/grayish casts.
    2. Divide original channels by the background map to flatten the paper to pure white.
    3. Non-linear tone curve on luminance/value:
       - Deepens dark strokes, table lines, and text (below black point).
       - Clamps bright paper pixels (above white point) to pure spotless white (255).
    4. Intelligent chroma & saturation handling:
       - Paper background has saturation neutralized to 0 (pure clean white paper).
       - Colored ink (red official stamps, blue ballpoint signatures, colored marks)
         has saturation preserved and boosted (+30%).
    5. High-pass unsharp masking for crisp text edges.

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        config: Optional configuration thresholds.

    Returns:
        3-channel BGR uint8 array (or 1-channel if input was 1-channel) with pure white paper,
        deep black text/tables, and preserved colored ink/stamps.
    """
    if config is None:
        config = EnhancementConfig()

    bgr = _extract_bgr_array(image)
    is_2d = (bgr.ndim == 2)
    if is_2d:
        img_bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = bgr.copy()

    h, w = img_bgr.shape[:2]
    if h < 4 or w < 4:
        return bgr.copy()

    # Kernel for local paper background illumination estimation
    k = min(config.bw_magic_kernel_size, max(3, (min(h, w) // 2) * 2 + 1))
    if k % 2 == 0:
        k -= 1
    struct_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    img_f = img_bgr.astype(np.float32)

    # 1. Background map per channel (removes color cast, dark corners, and shadows)
    bg = np.zeros_like(img_f)
    for c in range(3):
        dilated = cv2.dilate(img_f[:, :, c], struct_elem)
        bg[:, :, c] = cv2.GaussianBlur(dilated, (0, 0), sigmaX=max(1.0, k / 3.0))
    bg = np.maximum(bg, 1.0)

    # 2. Divide by background -> flattened paper (paper pixels reach ~255)
    norm = np.clip((img_f / bg) * 255.0, 0.0, 255.0)
    norm_u8 = norm.astype(np.uint8)

    # 3. Tone curve & saturation in HSV
    hsv = cv2.cvtColor(norm_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_c, s_c, v_c = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    black_pt = float(config.bw_magic_black_point)
    white_pt = float(config.bw_magic_white_point)
    v_scaled = (v_c - black_pt) / max(1.0, white_pt - black_pt)
    v_curved = np.clip(v_scaled * 255.0, 0.0, 255.0)
    v_curved = 255.0 * np.power(v_curved / 255.0, 0.85)

    # 4. Saturation handling:
    # If paper is white (v_curved > 230), kill saturation to prevent yellow/gray stains.
    # If colored ink / stamp (s_c > 25 and v_c < 220), boost saturation.
    is_colored_ink = (s_c > 25.0) & (v_c < 220.0)
    s_new = np.where(
        is_colored_ink,
        np.clip(s_c * config.bw_magic_sat_boost, 0.0, 255.0),
        np.where(v_curved > 230.0, 0.0, s_c * 0.5),
    )

    hsv_new = cv2.merge([h_c, np.clip(s_new, 0.0, 255.0), np.clip(v_curved, 0.0, 255.0)]).astype(np.uint8)
    res_bgr = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    # 5. Crisp edge sharpening (Unsharp Mask)
    if config.bw_magic_sharpen > 0.0:
        blurred = cv2.GaussianBlur(res_bgr, (0, 0), sigmaX=1.0)
        res_bgr = cv2.addWeighted(res_bgr, 1.0 + config.bw_magic_sharpen, blurred, -config.bw_magic_sharpen, 0)
        res_bgr = np.clip(res_bgr, 0, 255).astype(np.uint8)

    if is_2d:
        return cv2.cvtColor(res_bgr, cv2.COLOR_BGR2GRAY)
    return res_bgr


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


def enhance_smart_document(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Create a color-preserving, paper-whitened document scan.

    Smart Document mode is intentionally not binary. It corrects slow illumination
    gradients on the LAB luminance channel, applies restrained local contrast, and
    neutralizes bright paper pixels while leaving dark colored ink untouched.
    """
    if config is None:
        config = EnhancementConfig()

    bgr = _extract_bgr_array(image)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)

    height, width = bgr.shape[:2]
    if height < 2 or width < 2:
        return bgr.copy()

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Estimate the slowly varying paper/background illumination. The kernel is
    # bounded by the image size so tiny fixtures remain valid OpenCV inputs.
    kernel_size = min(config.smart_background_kernel_size, max(3, min(height, width) - 1))
    if kernel_size % 2 == 0:
        kernel_size -= 1
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(l_channel, cv2.MORPH_CLOSE, background_kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(1.0, kernel_size / 6.0))
    background = np.maximum(background, 1)
    normalized_l = cv2.divide(l_channel, background, scale=255)

    # Use robust percentiles rather than min/max so a few dark strokes do not
    # dictate the entire page contrast.
    low_percentile, high_percentile = np.percentile(normalized_l, (1.0, 99.0))
    if high_percentile - low_percentile >= 1.0:
        stretched_l = np.clip(
            (normalized_l.astype(np.float32) - float(low_percentile))
            * (255.0 / float(high_percentile - low_percentile)),
            0.0,
            255.0,
        ).astype(np.uint8)
    else:
        stretched_l = normalized_l

    clahe = cv2.createCLAHE(
        clipLimit=config.smart_clahe_clip_limit,
        tileGridSize=config.smart_clahe_tile_grid,
    )
    enhanced_l = clahe.apply(stretched_l)
    enhanced_l = cv2.bilateralFilter(
        enhanced_l,
        d=config.smart_denoise_d,
        sigmaColor=20.0,
        sigmaSpace=20.0,
    )

    # Neutralize the paper cast only where the original pixel is bright. Dark
    # handwriting/stamps keep their original chroma, including red and blue ink.
    paper_weight = (
        np.clip(
            (l_channel.astype(np.float32) - float(config.smart_paper_l_threshold))
            / max(1.0, 255.0 - float(config.smart_paper_l_threshold)),
            0.0,
            1.0,
        )
        * config.smart_neutralize_strength
    )
    neutralized_a = np.clip(
        a_channel.astype(np.float32) + (128.0 - a_channel.astype(np.float32)) * paper_weight,
        0.0,
        255.0,
    ).astype(np.uint8)
    neutralized_b = np.clip(
        b_channel.astype(np.float32) + (128.0 - b_channel.astype(np.float32)) * paper_weight,
        0.0,
        255.0,
    ).astype(np.uint8)

    enhanced_bgr = cv2.cvtColor(
        cv2.merge([enhanced_l, neutralized_a, neutralized_b]),
        cv2.COLOR_LAB2BGR,
    )

    if config.smart_sharpen_amount > 0.0:
        blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), sigmaX=config.smart_sharpen_sigma)
        enhanced_bgr = cv2.addWeighted(
            enhanced_bgr,
            1.0 + config.smart_sharpen_amount,
            blurred,
            -config.smart_sharpen_amount,
            0,
        )

    return np.clip(enhanced_bgr, 0, 255).astype(np.uint8)


def enhance_image(
    image: Union[np.ndarray, WarpedDocument, LoadedImage],
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    config: Optional[EnhancementConfig] = None,
) -> np.ndarray:
    """Enhance a document image according to the specified scan mode.

    Args:
        image: Source image array, WarpedDocument, or LoadedImage.
        mode: Enhancement filter mode, including ScanMode.SMART_DOCUMENT.
        config: Optional configuration parameters. Uses defaults if omitted.

    Returns:
        Enhanced NumPy array: 1-channel uint8 (H, W) for Gray,
        3-channel uint8 (H, W, 3) for B&W (Magic Pro), Color, and Smart Document.

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
    elif mode_str in (
        ScanMode.SMART_DOCUMENT.value,
        "smart",
        "smart_document",
        "smart-document",
    ):
        return enhance_smart_document(image, config)
    else:
        raise ValueError(
            f"Unsupported scan enhancement mode '{mode}'. "
            f"Expected one of: {[m.value for m in ScanMode]}"
        )
