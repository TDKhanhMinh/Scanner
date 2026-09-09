"""Unit tests for document boundary detection and geometric validation."""

import hashlib
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from attendance_scanner.pipeline.detect import (
    BoundaryCollisionResult,
    DetectionConfig,
    DetectionRejectionReason,
    DetectionResult,
    assess_boundary_collision,
    detect_document_boundary,
    order_corners,
)
from attendance_scanner.pipeline.load import load_image


def test_order_corners_canonical_order():
    """Verify order_corners orders arbitrary permutations into [TL, TR, BR, BL]."""
    # Canonical rectangle: TL=(100, 100), TR=(500, 100), BR=(500, 400), BL=(100, 400)
    expected_tl = (100.0, 100.0)
    expected_tr = (500.0, 100.0)
    expected_br = (500.0, 400.0)
    expected_bl = (100.0, 400.0)

    canonical = np.array([expected_tl, expected_tr, expected_br, expected_bl], dtype=np.float32)

    # Test permutations
    permutations = [
        [1, 0, 3, 2],  # TR, TL, BL, BR
        [2, 3, 0, 1],  # BR, BL, TL, TR
        [3, 2, 1, 0],  # BL, BR, TR, TL
        [0, 2, 1, 3],  # TL, BR, TR, BL
    ]

    for perm in permutations:
        scrambled = canonical[perm]
        ordered = order_corners(scrambled)
        np.testing.assert_allclose(ordered[0], expected_tl, atol=1e-3)
        np.testing.assert_allclose(ordered[1], expected_tr, atol=1e-3)
        np.testing.assert_allclose(ordered[2], expected_br, atol=1e-3)
        np.testing.assert_allclose(ordered[3], expected_bl, atol=1e-3)


def test_detect_high_contrast_synthetic_document():
    """Verify document detection on a clear high-contrast rectangular document."""
    # Create 800x600 dark canvas (BGR dark wood / desk: [30, 30, 30])
    w, h = 800, 600
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    # Draw white paper sheet at known coordinates: [100, 80] to [700, 520]
    expected_pts = np.array(
        [[100, 80], [700, 80], [700, 520], [100, 520]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [expected_pts], (245, 245, 245))

    result = detect_document_boundary(canvas)

    assert result is not None
    assert result.detected is True
    assert len(result.corners) == 4
    assert result.scale_factor == 1.0
    assert result.confidence > 0.8
    assert 0.4 < result.area_ratio < 0.8

    # Verify detected corners match expected within 5 pixels
    detected_arr = result.corners_array
    for i in range(4):
        exp = expected_pts[i]
        det = detected_arr[i]
        dist = np.linalg.norm(det - exp)
        assert dist < 5.0, f"Corner {i} offset {dist} exceeds tolerance"


def test_detect_prefers_outer_paper_over_inner_attendance_table():
    """The bright sheet boundary should win over a darker inner table frame."""
    canvas = np.full((700, 900, 3), 80, dtype=np.uint8)
    outer = np.array([[70, 55], [830, 70], [815, 645], [75, 625]], dtype=np.int32)
    inner = np.array([[170, 155], [740, 165], [735, 545], [175, 535]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, outer, (215, 215, 215))
    cv2.polylines(canvas, [outer], True, (185, 185, 185), 3)
    cv2.polylines(canvas, [inner], True, (20, 20, 20), 5)
    for y in np.linspace(205, 505, 7, dtype=np.int32):
        cv2.line(canvas, (175, int(y)), (735, int(y)), (20, 20, 20), 2)

    result = detect_document_boundary(canvas)

    assert result is not None
    assert result.diagnostics["candidate_source"] in {
        "paper_mask",
        "paper_min_area_rect",
    }
    assert result.area_ratio > 0.55
    assert result.corners_array[:, 0].min() < 110
    assert result.corners_array[:, 1].min() < 100


def test_detect_rotated_perspective_quadrilateral():
    """Verify detection on a slightly rotated quadrilateral document."""
    w, h = 900, 700
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)

    # Quadrilateral with perspective angles
    pts = np.array(
        [[150, 100], [760, 140], [710, 590], [120, 550]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [pts], (240, 240, 240))

    result = detect_document_boundary(canvas)

    assert result is not None
    assert result.detected is True
    assert result.confidence > 0.7

    detected_arr = result.corners_array
    # Verify corners are in clockwise order [TL, TR, BR, BL]
    assert detected_arr[0, 1] < detected_arr[3, 1]  # TL is above BL
    assert detected_arr[1, 1] < detected_arr[2, 1]  # TR is above BR
    assert detected_arr[0, 0] < detected_arr[1, 0]  # TL is left of TR
    assert detected_arr[3, 0] < detected_arr[2, 0]  # BL is left of BR


def test_detect_large_image_scaled_down():
    """Verify large images (e.g. 2400x1600) are downscaled and mapped back accurately."""
    orig_w, orig_h = 2400, 1600
    canvas = np.full((orig_h, orig_w, 3), 25, dtype=np.uint8)

    expected_pts = np.array(
        [[300, 200], [2100, 200], [2100, 1400], [300, 1400]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [expected_pts], (250, 250, 250))

    config = DetectionConfig(max_dimension=1200)
    result = detect_document_boundary(canvas, config=config)

    assert result is not None
    assert result.detected is True
    # Scale factor should be 1200 / 2400 = 0.5
    assert abs(result.scale_factor - 0.5) < 1e-4

    # Check corners mapped back to original 2400x1600 scale
    detected_arr = result.corners_array
    for i in range(4):
        exp = expected_pts[i]
        det = detected_arr[i]
        dist = np.linalg.norm(det - exp)
        assert dist < 15.0, f"Scaled corner {i} offset {dist} exceeds tolerance"

    # Verify all coordinates are strictly within bounds [0, orig_w - 1], [0, orig_h - 1]
    assert np.all(detected_arr[:, 0] >= 0.0)
    assert np.all(detected_arr[:, 0] <= float(orig_w - 1))
    assert np.all(detected_arr[:, 1] >= 0.0)
    assert np.all(detected_arr[:, 1] <= float(orig_h - 1))


def test_detect_loaded_image_integration(tmp_path: Path):
    """Verify passing a LoadedImage instance from pipeline.load directly to detector."""
    img_path = tmp_path / "sheet.png"
    img = Image.new("RGB", (600, 400), color=(30, 30, 30))
    # Draw white box on PIL image
    for y in range(50, 350):
        for x in range(80, 520):
            img.putpixel((x, y), (240, 240, 240))
    img.save(img_path)

    loaded = load_image(img_path)
    result = detect_document_boundary(loaded)

    assert result is not None
    assert result.detected is True
    assert len(result.corners) == 4


def test_safe_fallback_blank_image():
    """Verify solid black, solid gray, and solid white images safely return None."""
    for val in [0, 128, 255]:
        blank = np.full((400, 400, 3), val, dtype=np.uint8)
        result = detect_document_boundary(blank)
        assert result is None


def test_safe_fallback_random_noise():
    """Verify random noise without quadrilateral structure safely returns None."""
    rng = np.random.default_rng(42)
    noise = rng.integers(0, 256, (400, 400, 3), dtype=np.uint8)
    result = detect_document_boundary(noise)
    assert result is None


def test_safe_fallback_too_small_document():
    """Verify shapes with area ratio below min_area_ratio are rejected."""
    canvas = np.full((600, 600, 3), 30, dtype=np.uint8)
    # Tiny 20x20 square in 600x600 canvas: area ratio = 400 / 360000 = 0.0011 (< 0.05)
    cv2.rectangle(canvas, (200, 200), (220, 220), (255, 255, 255), -1)

    result = detect_document_boundary(canvas)
    assert result is None


def test_safe_fallback_non_convex_l_shape():
    """Verify non-convex L-shaped polygon is rejected by convexity check."""
    canvas = np.full((600, 600, 3), 30, dtype=np.uint8)
    # L-shaped polygon (concave)
    l_pts = np.array(
        [[100, 100], [400, 100], [400, 250], [250, 250], [250, 500], [100, 500]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [l_pts], (250, 250, 250))

    result = detect_document_boundary(canvas)
    assert result is None


def test_source_image_is_not_mutated():
    """Verify detect_document_boundary does not mutate the input array."""
    w, h = 600, 400
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (100, 50), (500, 350), (240, 240, 240), -1)

    initial_hash = hashlib.sha256(canvas.tobytes()).hexdigest()
    _ = detect_document_boundary(canvas)
    after_hash = hashlib.sha256(canvas.tobytes()).hexdigest()

    assert initial_hash == after_hash


def test_detection_result_serialization_and_contract():
    """Verify DetectionResult Pydantic contract and camelCase aliasing."""
    corners: List[Tuple[float, float]] = [
        (10.0, 20.0),
        (500.0, 25.0),
        (490.0, 380.0),
        (15.0, 375.0),
    ]
    res = DetectionResult(
        detected=True,
        corners=corners,
        confidence=0.92,
        area_ratio=0.65,
        scale_factor=0.5,
        diagnostics={"test": True},
    )

    dumped = res.model_dump(by_alias=True)
    assert dumped["detected"] is True
    assert dumped["areaRatio"] == 0.65
    assert dumped["scaleFactor"] == 0.5
    assert dumped["confidence"] == 0.92
    assert len(dumped["corners"]) == 4

    arr = res.corners_array
    assert arr.shape == (4, 2)
    assert arr.dtype == np.float32

    with pytest.raises(ValidationError):
        DetectionResult(
            accepted=True,
            clipped=True,
            corners=corners,
            confidence=0.5,
            area_ratio=0.5,
            scale_factor=1.0,
        )

    with pytest.raises(ValidationError):
        DetectionResult(
            accepted=False,
            clipped=False,
            corners=corners,
            confidence=0.5,
            area_ratio=0.5,
            scale_factor=1.0,
        )


def test_detection_config_validation_invalid_bounds():
    """Verify DetectionConfig rejects invalid parameter bounds and conflicting thresholds."""
    # Invalid max_dimension (must be >= 100)
    with pytest.raises(ValidationError):
        DetectionConfig(max_dimension=0)

    with pytest.raises(ValidationError):
        DetectionConfig(max_dimension=-50)

    # Invalid morph_kernel_size (must be >= 1)
    with pytest.raises(ValidationError):
        DetectionConfig(morph_kernel_size=0)

    # Invalid blur_kernel_size (must be >= 1)
    with pytest.raises(ValidationError):
        DetectionConfig(blur_kernel_size=0)

    # Invalid area ratio bounds
    with pytest.raises(ValidationError):
        DetectionConfig(min_area_ratio=0.0)

    with pytest.raises(ValidationError):
        DetectionConfig(max_area_ratio=1.5)

    # Conflicting area ratios: min >= max
    with pytest.raises(ValidationError):
        DetectionConfig(min_area_ratio=0.8, max_area_ratio=0.2)

    # Conflicting angles: min >= max
    with pytest.raises(ValidationError):
        DetectionConfig(min_angle_deg=150.0, max_angle_deg=30.0)

    # Conflicting Canny thresholds: threshold1 > threshold2
    with pytest.raises(ValidationError):
        DetectionConfig(canny_threshold1=200, canny_threshold2=50)


def test_detect_document_boundary_with_degenerate_config_safely_returns_none():
    """Verify detect_document_boundary safely returns None when given degenerate/raw config."""
    canvas = np.full((400, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (50, 50), (350, 350), (250, 250, 250), -1)

    # Construct degenerate config bypassing standard validation
    degenerate_dim = DetectionConfig.model_construct(max_dimension=0)
    assert detect_document_boundary(canvas, config=degenerate_dim) is None

    degenerate_morph = DetectionConfig.model_construct(morph_kernel_size=0)
    assert detect_document_boundary(canvas, config=degenerate_morph) is None


def test_assess_boundary_collision_pure_helper():
    """Verify pure helper accurately detects out-of-bounds corners and border collisions."""
    w, h = 800, 600

    # 1. Fully inside corners
    inside_corners = np.array(
        [[100.0, 80.0], [700.0, 80.0], [700.0, 520.0], [100.0, 520.0]], dtype=np.float32
    )
    res_inside = assess_boundary_collision(inside_corners, (w, h))
    assert isinstance(res_inside, BoundaryCollisionResult)
    assert res_inside.is_clipped is False
    assert res_inside.out_of_bounds_count == 0
    assert len(res_inside.reasons) == 0

    # 2. Negative coordinate out-of-bounds
    neg_corners = np.array(
        [[-15.0, 80.0], [700.0, 80.0], [700.0, 520.0], [100.0, 520.0]], dtype=np.float32
    )
    res_neg = assess_boundary_collision(neg_corners, (w, h))
    assert res_neg.is_clipped is True
    assert res_neg.out_of_bounds_count == 1
    assert "1_corners_out_of_bounds" in res_neg.reasons

    # 3. Coordinate exceeding width/height
    overflow_corners = np.array(
        [[100.0, 80.0], [820.0, 80.0], [700.0, 610.0], [100.0, 520.0]], dtype=np.float32
    )
    res_overflow = assess_boundary_collision(overflow_corners, (w, h))
    assert res_overflow.is_clipped is True
    assert res_overflow.out_of_bounds_count == 2
    assert "2_corners_out_of_bounds" in res_overflow.reasons

    # 4. Edge running along top border with paper mask extending beyond
    top_border_corners = np.array(
        [[50.0, 2.0], [750.0, 2.0], [750.0, 450.0], [50.0, 450.0]], dtype=np.float32
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[0:50, :] = 255  # paper continues to the very top edge
    res_top = assess_boundary_collision(top_border_corners, (w, h), paper_mask=mask)
    assert res_top.is_clipped is True
    assert "top" in res_top.border_touch_edges
    assert "paper_mask_exceeds_top_border" in res_top.reasons


def test_detect_clipped_document_returns_accepted_false_with_document_clipped():
    """Verify document clipped by camera border is detected but rejected with DOCUMENT_CLIPPED."""
    w, h = 800, 600
    canvas = np.full((h, w, 3), 30, dtype=np.uint8)

    # Document sheet cut off on the top edge (touches y=0 from x=50 to x=750)
    sheet_pts = np.array(
        [[50, 0], [750, 0], [750, 480], [50, 480]],
        dtype=np.int32,
    )
    cv2.fillPoly(canvas, [sheet_pts], (245, 245, 245))

    result = detect_document_boundary(canvas)

    assert result is not None
    assert result.detected is True
    assert result.accepted is False
    assert result.clipped is True
    assert result.rejection_reason == DetectionRejectionReason.DOCUMENT_CLIPPED
    assert len(result.corners) == 4
    assert result.diagnostics.get("collision_reasons") is not None


def test_detect_inner_table_not_warped_when_paper_clipped():
    """When the paper sheet is clipped off-camera, inner table is not accepted as unclipped doc."""
    w, h = 900, 700
    canvas = np.full((h, w, 3), 40, dtype=np.uint8)

    # Paper sheet clipped at the left edge (touches x=0)
    paper = np.array([[0, 20], [800, 20], [800, 680], [0, 680]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, paper, (235, 235, 235))

    # Inner table grid fully inside the frame
    inner_table = np.array([[120, 100], [750, 100], [750, 600], [120, 600]], dtype=np.int32)
    cv2.polylines(canvas, [inner_table], True, (20, 20, 20), 4)
    for y in np.linspace(140, 560, 6, dtype=np.int32):
        cv2.line(canvas, (120, int(y)), (750, int(y)), (20, 20, 20), 2)

    result = detect_document_boundary(canvas)

    assert result is not None
    # Because paper touches border, candidate cannot be accepted for perspective warp
    assert result.accepted is False
    assert result.clipped is True
    assert result.rejection_reason == DetectionRejectionReason.DOCUMENT_CLIPPED


def test_small_unrelated_border_fragment_does_not_clip_complete_candidate():
    """A small bright object at the frame edge must not invalidate the main page."""
    canvas = np.full((700, 900, 3), 50, dtype=np.uint8)
    paper = np.array([[180, 100], [820, 120], [800, 620], [190, 640]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, paper, (235, 235, 235))
    cv2.polylines(canvas, [paper], True, (180, 180, 180), 3)
    cv2.rectangle(canvas, (0, 0), (130, 25), (235, 235, 235), -1)

    result = detect_document_boundary(canvas)

    assert result is not None
    assert result.accepted is True
    assert result.clipped is False


def test_inner_table_candidate_is_rejected_when_outer_margin_contains_paper_ink():
    """A strong inner frame must not be accepted when the surrounding margin is still paper."""
    canvas = np.full((700, 900, 3), 170, dtype=np.uint8)
    outer = np.array([[70, 55], [830, 70], [815, 645], [75, 625]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, outer, (200, 200, 200))

    inner = np.array([[170, 155], [740, 165], [735, 545], [175, 535]], dtype=np.int32)
    cv2.polylines(canvas, [inner], True, (20, 20, 20), 5)
    for y in np.linspace(205, 505, 7, dtype=np.int32):
        cv2.line(canvas, (175, int(y)), (735, int(y)), (20, 20, 20), 2)
    for x in (230, 410, 590):
        cv2.rectangle(canvas, (x, 130), (x + 80, 148), (30, 30, 30), -1)

    result = detect_document_boundary(
        canvas,
        config=DetectionConfig(paper_min_threshold=240, paper_max_threshold=245),
    )

    assert result is not None
    assert result.accepted is False
    assert result.clipped is False
    assert result.rejection_reason == DetectionRejectionReason.INNER_TABLE_COLLAPSE


def test_real_fixtures_optional_regression():
    """Optional manual regression test for user real images via SCANNER_REAL_FIXTURE_ROOT."""
    fixture_root_str = os.environ.get("SCANNER_REAL_FIXTURE_ROOT")
    if not fixture_root_str or not os.path.isdir(fixture_root_str):
        pytest.skip(
            "SCANNER_REAL_FIXTURE_ROOT not configured; skipping manual real image regression."
        )

    fixture_dir = Path(fixture_root_str)
    image_files = [
        p for p in fixture_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
    assert len(image_files) > 0, f"No image files found in {fixture_dir}"

    for img_path in image_files:
        loaded = load_image(img_path)
        result = detect_document_boundary(loaded)
        if result is not None and result.clipped:
            # Clipped candidates must never be marked accepted
            assert result.accepted is False
            assert result.rejection_reason == DetectionRejectionReason.DOCUMENT_CLIPPED
