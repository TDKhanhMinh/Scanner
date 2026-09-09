"""Unit tests for scan enhancement filters."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from attendance_scanner.contracts import ScanMode
from attendance_scanner.pipeline.detect import detect_document_boundary
from attendance_scanner.pipeline.enhance import (
    EnhancementConfig,
    enhance_bw,
    enhance_color,
    enhance_gray,
    enhance_image,
    enhance_smart_document,
)
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.pipeline.perspective import warp_perspective


def test_enhance_output_invariants():
    """Verify channel counts, dimensions, and data types for all four enhancement modes."""
    w, h = 320, 240
    # Synthetic document with text and lines
    bgr = np.full((h, w, 3), 230, dtype=np.uint8)
    cv2.rectangle(bgr, (20, 20), (300, 220), (50, 50, 50), 2)
    cv2.putText(bgr, "ATTENDANCE", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)

    # 1. Gray mode: 1-channel (2D) uint8
    gray_out = enhance_gray(bgr)
    assert gray_out.ndim == 2
    assert gray_out.shape == (h, w)
    assert gray_out.dtype == np.uint8

    # 2. B&W mode: 1-channel (2D) binary uint8 ({0, 255})
    bw_out = enhance_bw(bgr)
    assert bw_out.ndim == 2
    assert bw_out.shape == (h, w)
    assert bw_out.dtype == np.uint8
    unique_vals = set(np.unique(bw_out))
    assert unique_vals.issubset({0, 255})

    # 3. Color Enhanced mode: 3-channel (3D) uint8
    color_out = enhance_color(bgr)
    assert color_out.ndim == 3
    assert color_out.shape == (h, w, 3)
    assert color_out.dtype == np.uint8

    # 4. Smart Document mode: 3-channel color-preserving output
    smart_out = enhance_smart_document(bgr)
    assert smart_out.ndim == 3
    assert smart_out.shape == (h, w, 3)
    assert smart_out.dtype == np.uint8


def test_enhance_synthetic_uneven_lighting_and_stamp_preservation():
    """Verify contrast under shadow gradients and official red stamp color preservation."""
    w, h = 400, 300
    # Create non-uniform shadow: left side 80 (shadowed), right side 220 (bright)
    grad = np.linspace(80, 220, w, dtype=np.uint8)
    canvas = np.tile(grad, (h, 1))
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    # Add dark text line across the page
    cv2.putText(bgr, "Nguyen Van A", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)

    # Add faint pencil signature in shadowed area
    cv2.putText(bgr, "Faint Mark", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1)

    # Add red official stamp circle (R=220, G=30, B=30 -> in BGR: [30, 30, 220])
    cv2.circle(bgr, (320, 180), 30, (30, 30, 220), -1)

    # --- Test Gray mode ---
    gray_res = enhance_image(bgr, mode=ScanMode.GRAY)
    assert gray_res.shape == (h, w)
    # The faint signature and dark text should both have local contrast
    assert gray_res[120, 50] < gray_res[100, 50]  # text pixel darker than surrounding paper

    # --- Test B&W mode ---
    bw_res = enhance_image(bgr, mode=ScanMode.BW)
    assert bw_res.shape == (h, w)
    # Paper background on both sides should be normalized to pure white (255)
    assert bw_res[40, 40] == 255
    assert bw_res[40, 360] == 255

    # --- Test Color Enhanced mode ---
    color_res = enhance_image(bgr, mode=ScanMode.COLOR)
    assert color_res.shape == (h, w, 3)
    # Stamp center should remain distinctly red (Red channel >> Blue and Green)
    stamp_pixel = color_res[180, 320]
    b_val, g_val, r_val = int(stamp_pixel[0]), int(stamp_pixel[1]), int(stamp_pixel[2])
    assert r_val > 150
    assert r_val > b_val + 50
    assert r_val > g_val + 50

    # --- Test Smart Document mode ---
    smart_res = enhance_image(bgr, mode=ScanMode.SMART_DOCUMENT)
    assert smart_res.shape == (h, w, 3)
    smart_stamp = smart_res[180, 320]
    smart_b, smart_g, smart_r = [int(value) for value in smart_stamp]
    assert smart_r > 150
    assert smart_r > smart_b + 40
    assert smart_r > smart_g + 40


def test_enhance_uniform_images_no_crash():
    """Verify uniform black, white, and gray images do not crash with division by zero."""
    for val in [0, 128, 255]:
        uniform_img = np.full((100, 100, 3), val, dtype=np.uint8)
        # Should execute cleanly without error across all 4 modes
        res_gray = enhance_gray(uniform_img)
        assert res_gray.shape == (100, 100)

        res_bw = enhance_bw(uniform_img)
        assert res_bw.shape == (100, 100)

        res_color = enhance_color(uniform_img)
        assert res_color.shape == (100, 100, 3)

        res_smart = enhance_smart_document(uniform_img)
        assert res_smart.shape == (100, 100, 3)


def test_enhance_small_image():
    """Verify small images (e.g. 4x4) are handled safely without kernel boundary errors."""
    tiny = np.full((4, 4, 3), 150, dtype=np.uint8)
    assert enhance_gray(tiny).shape == (4, 4)
    assert enhance_bw(tiny).shape == (4, 4)
    assert enhance_color(tiny).shape == (4, 4, 3)
    assert enhance_smart_document(tiny).shape == (4, 4, 3)


def test_enhancement_config_validation():
    """Verify EnhancementConfig validates parameters and auto-adjusts block sizes to odd."""
    # Even block size 20 should auto-adjust to odd 21
    cfg = EnhancementConfig(bw_adaptive_block_size=20)
    assert cfg.bw_adaptive_block_size == 21
    smart_cfg = EnhancementConfig(smart_background_kernel_size=50)
    assert smart_cfg.smart_background_kernel_size == 51

    # Invalid CLAHE clip limit (must be > 0)
    with pytest.raises(ValidationError):
        EnhancementConfig(gray_clahe_clip_limit=0.0)

    # Invalid denoise d (must be >= 1)
    with pytest.raises(ValidationError):
        EnhancementConfig(gray_denoise_d=0)

    # Invalid CLAHE tile grid (must have positive dimensions > 0)
    with pytest.raises(ValidationError):
        EnhancementConfig(gray_clahe_tile_grid=(0, 0))

    with pytest.raises(ValidationError):
        EnhancementConfig(gray_clahe_tile_grid=(0, 8))

    with pytest.raises(ValidationError):
        EnhancementConfig(color_clahe_tile_grid=(8, -1))


def test_source_image_is_not_mutated():
    """Verify enhancement functions do not mutate the input image array."""
    bgr = np.full((200, 200, 3), 100, dtype=np.uint8)
    cv2.rectangle(bgr, (20, 20), (180, 180), (240, 240, 240), -1)

    initial_hash = hashlib.sha256(bgr.tobytes()).hexdigest()

    _ = enhance_gray(bgr)
    _ = enhance_bw(bgr)
    _ = enhance_color(bgr)
    _ = enhance_smart_document(bgr)
    _ = enhance_image(bgr, ScanMode.GRAY)

    after_hash = hashlib.sha256(bgr.tobytes()).hexdigest()
    assert initial_hash == after_hash


def test_enhance_image_mode_dispatch_and_errors():
    """Verify enhance_image dispatches correctly with ScanMode enums and strings."""
    bgr = np.full((100, 100, 3), 200, dtype=np.uint8)

    # Enum dispatch
    assert enhance_image(bgr, ScanMode.GRAY).ndim == 2
    assert enhance_image(bgr, ScanMode.BW).ndim == 2
    assert enhance_image(bgr, ScanMode.COLOR).ndim == 3
    assert enhance_image(bgr, ScanMode.SMART_DOCUMENT).ndim == 3

    # String dispatch
    assert enhance_image(bgr, "gray").ndim == 2
    assert enhance_image(bgr, "GRAY").ndim == 2
    assert enhance_image(bgr, "bw").ndim == 2
    assert enhance_image(bgr, "color").ndim == 3
    assert enhance_image(bgr, "color_enhanced").ndim == 3
    assert enhance_image(bgr, "smart").ndim == 3

    # Unsupported mode
    with pytest.raises(ValueError) as exc_info:
        enhance_image(bgr, "sepia")
    assert "unsupported" in str(exc_info.value).lower()


def test_end_to_end_pipeline_integration(tmp_path: Path):
    """Verify end-to-end integration: load -> detect -> warp -> enhance."""
    # 1. Create source file
    img_path = tmp_path / "sheet_pipeline.png"
    w, h = 600, 450
    img = Image.new("RGB", (w, h), color=(30, 30, 30))

    # Draw document rectangle
    for y in range(50, 400):
        for x in range(80, 520):
            img.putpixel((x, y), (230, 230, 230))
    img.save(img_path)

    # 2. Stage 1: Load image
    loaded = load_image(img_path)

    # 3. Stage 2: Detect document boundary
    detection = detect_document_boundary(loaded)
    assert detection is not None

    # 4. Stage 3: Perspective warp
    warped = warp_perspective(loaded, detection)
    assert warped.width > 400
    assert warped.height > 300

    # 5. Stage 4: Scan enhancement in all 4 modes
    enhanced_gray = enhance_image(warped, ScanMode.GRAY)
    assert enhanced_gray.shape == (warped.height, warped.width)
    assert enhanced_gray.dtype == np.uint8

    enhanced_bw = enhance_image(warped, ScanMode.BW)
    assert enhanced_bw.shape == (warped.height, warped.width)
    assert set(np.unique(enhanced_bw)).issubset({0, 255})

    enhanced_color = enhance_image(warped, ScanMode.COLOR)
    assert enhanced_color.shape == (warped.height, warped.width, 3)
    assert enhanced_color.dtype == np.uint8

    enhanced_smart = enhance_image(warped, ScanMode.SMART_DOCUMENT)
    assert enhanced_smart.shape == (warped.height, warped.width, 3)
    assert enhanced_smart.dtype == np.uint8
