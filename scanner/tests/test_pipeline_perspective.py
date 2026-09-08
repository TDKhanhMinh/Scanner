"""Unit tests for perspective transformation, corner ordering, and document warping."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from attendance_scanner.pipeline.detect import detect_document_boundary
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.pipeline.perspective import (
    DegenerateCornersError,
    PerspectiveConfig,
    WarpedDocument,
    compute_destination_dimensions,
    warp_perspective,
)


def test_compute_destination_dimensions_rectangle():
    """Verify compute_destination_dimensions on a standard upright rectangle."""
    # 400 wide x 250 high
    corners = np.array(
        [[100.0, 50.0], [500.0, 50.0], [500.0, 300.0], [100.0, 300.0]],
        dtype=np.float32,
    )
    w, h = compute_destination_dimensions(corners)
    assert w == 400
    assert h == 250


def test_compute_destination_dimensions_trapezoid_and_natural_aspect_ratio():
    """Verify max opposite-side Euclidean distance calculation without arbitrary A4 stretching."""
    # Top width: 300, Bottom width: 340
    # Left height: 200, Right height: 220
    # Destination should be max(300, 340) = 340 wide by max(200, 220) = 220 high
    tl = [100.0, 100.0]
    tr = [400.0, 100.0]  # width_top = 300
    br = [420.0, 320.0]  # height_right ~= 220.9
    bl = [80.0, 300.0]  # width_bottom ~= 340.6, height_left ~= 201.0

    corners = np.array([tl, tr, br, bl], dtype=np.float32)
    w, h = compute_destination_dimensions(corners)
    assert w == 341
    assert h == 221


def test_warp_perspective_applies_configured_target_aspect_ratio():
    """A configured template ratio normalizes perspective output page geometry."""
    image = np.full((500, 700, 3), 240, dtype=np.uint8)
    corners = np.array(
        [[100.0, 80.0], [600.0, 90.0], [580.0, 420.0], [110.0, 410.0]],
        dtype=np.float32,
    )

    warped = warp_perspective(
        image,
        corners,
        PerspectiveConfig(target_aspect_ratio=2**0.5),
    )

    assert abs((warped.width / warped.height) - 2**0.5) < 0.01


def test_warp_perspective_synthetic_grid_rectification():
    """Verify warping a synthetic tilted grid rectifies it to true rectangular geometry."""
    orig_w, orig_h = 400, 300
    grid_img = np.full((orig_h, orig_w, 3), 240, dtype=np.uint8)

    # Draw grid lines on flat document
    for y in range(0, orig_h, 30):
        cv2.line(grid_img, (0, y), (orig_w, y), (50, 50, 50), 2)
    for x in range(0, orig_w, 40):
        cv2.line(grid_img, (x, 0), (x, orig_h), (50, 50, 50), 2)

    # Perspective warp grid onto a larger dark canvas
    canvas_w, canvas_h = 800, 600
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    # Quad coordinates on canvas
    src_rect = np.array(
        [[0.0, 0.0], [orig_w - 1, 0.0], [orig_w - 1, orig_h - 1], [0.0, orig_h - 1]],
        dtype=np.float32,
    )
    quad_on_canvas = np.array(
        [[120.0, 80.0], [680.0, 110.0], [620.0, 520.0], [160.0, 480.0]],
        dtype=np.float32,
    )

    h_mat = cv2.getPerspectiveTransform(src_rect, quad_on_canvas)
    distorted = cv2.warpPerspective(grid_img, h_mat, (canvas_w, canvas_h))

    # Composite distorted grid onto dark canvas
    mask = cv2.warpPerspective(
        np.full((orig_h, orig_w), 255, dtype=np.uint8), h_mat, (canvas_w, canvas_h)
    )
    canvas[mask > 0] = distorted[mask > 0]

    # Now rectify back using warp_perspective
    warped = warp_perspective(canvas, quad_on_canvas)

    assert isinstance(warped, WarpedDocument)
    assert warped.width > 500
    assert warped.height > 400
    assert warped.image.shape == (warped.height, warped.width, 3)
    assert warped.transform_matrix.shape == (3, 3)

    # Check that center of warped image has valid non-dark grid pixels
    h_start = warped.height // 4
    h_end = 3 * warped.height // 4
    w_start = warped.width // 4
    w_end = 3 * warped.width // 4
    center_roi = warped.image[h_start:h_end, w_start:w_end]
    assert np.mean(center_roi) > 100

    # Geometric assertion 1: Verify destination corner re-projection MAE
    reprojected_corners = cv2.perspectiveTransform(
        warped.source_corners.reshape(1, 4, 2), warped.transform_matrix
    ).reshape(4, 2)
    expected_dst = np.array(
        [
            [0.0, 0.0],
            [float(warped.width - 1), 0.0],
            [float(warped.width - 1), float(warped.height - 1)],
            [0.0, float(warped.height - 1)],
        ],
        dtype=np.float32,
    )
    corner_mae = float(np.mean(np.abs(reprojected_corners - expected_dst)))
    assert corner_mae < 1e-3

    # Geometric assertion 2: Sample interior grid points and verify unwarped landmark MAE
    grid_pts = np.array(
        [
            [float(x), float(y)]
            for y in range(30, orig_h - 30, 30)
            for x in range(40, orig_w - 40, 40)
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)
    canvas_pts = cv2.perspectiveTransform(grid_pts, h_mat)
    rectified_pts = cv2.perspectiveTransform(canvas_pts, warped.transform_matrix).reshape(-1, 2)
    scale_norm = np.array(
        [(orig_w - 1) / float(warped.width - 1), (orig_h - 1) / float(warped.height - 1)]
    )
    mapped_back = rectified_pts * scale_norm
    geometric_mae = float(np.mean(np.abs(mapped_back - grid_pts.reshape(-1, 2))))
    assert geometric_mae < 5.0  # Tight sub-pixel/geometric tolerance to prevent regression


def test_warp_perspective_scrambled_corner_order():
    """Verify warp_perspective automatically orders corners into canonical [TL, TR, BR, BL]."""
    canvas = np.full((500, 500, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (100, 100), (400, 350), (220, 220, 220), -1)

    canonical_corners = np.array(
        [[100.0, 100.0], [400.0, 100.0], [400.0, 350.0], [100.0, 350.0]],
        dtype=np.float32,
    )

    # Scramble order: BR, TL, BL, TR
    scrambled = canonical_corners[[2, 0, 3, 1]]

    res_canonical = warp_perspective(canvas, canonical_corners)
    res_scrambled = warp_perspective(canvas, scrambled)

    assert res_canonical.width == res_scrambled.width
    assert res_canonical.height == res_scrambled.height
    np.testing.assert_allclose(res_canonical.image, res_scrambled.image, atol=1.0)


def test_degenerate_corners_collinear_raises_error():
    """Verify 4 collinear points raise DegenerateCornersError."""
    canvas = np.full((300, 300, 3), 50, dtype=np.uint8)
    # 4 points along horizontal line y = 100
    collinear = np.array([[50.0, 100.0], [100.0, 100.0], [150.0, 100.0], [200.0, 100.0]])

    with pytest.raises(DegenerateCornersError) as exc_info:
        warp_perspective(canvas, collinear)
    assert "degenerate" in str(exc_info.value).lower() or "collinear" in str(exc_info.value).lower()


def test_degenerate_corners_coincident_points_raises_error():
    """Verify 4 identical points raise DegenerateCornersError."""
    canvas = np.full((300, 300, 3), 50, dtype=np.uint8)
    identical = np.array([[100.0, 100.0]] * 4)

    with pytest.raises(DegenerateCornersError):
        warp_perspective(canvas, identical)


def test_degenerate_corners_sub_minimum_dimension_raises_error():
    """Verify corners smaller than min_dimension threshold raise DegenerateCornersError."""
    canvas = np.full((300, 300, 3), 50, dtype=np.uint8)
    # Tiny 4x4 box (less than default min_dimension = 10)
    tiny = np.array([[50.0, 50.0], [54.0, 50.0], [54.0, 54.0], [50.0, 54.0]])

    with pytest.raises(DegenerateCornersError) as exc_info:
        warp_perspective(canvas, tiny, min_dimension=10)
    assert "smaller than minimum dimension" in str(exc_info.value).lower()


def test_degenerate_corners_nan_or_inf_raises_error():
    """Verify NaN or Inf coordinates raise DegenerateCornersError."""
    canvas = np.full((300, 300, 3), 50, dtype=np.uint8)
    nan_corners = np.array([[100.0, 100.0], [np.nan, 100.0], [400.0, 300.0], [100.0, 300.0]])

    with pytest.raises(DegenerateCornersError) as exc_info:
        warp_perspective(canvas, nan_corners)
    assert "non-finite" in str(exc_info.value).lower()


def test_degenerate_corners_invalid_point_count_raises_error():
    """Verify passing other than 4 points raises DegenerateCornersError."""
    canvas = np.full((300, 300, 3), 50, dtype=np.uint8)
    three_pts = np.array([[100.0, 100.0], [400.0, 100.0], [400.0, 300.0]])

    with pytest.raises(DegenerateCornersError) as exc_info:
        warp_perspective(canvas, three_pts)
    assert "convertible to 4 points" in str(exc_info.value).lower()


def test_source_image_is_not_mutated():
    """Verify warp_perspective does not mutate the source image array."""
    canvas = np.full((400, 400, 3), 30, dtype=np.uint8)
    cv2.rectangle(canvas, (80, 80), (320, 320), (240, 240, 240), -1)
    corners = np.array([[80.0, 80.0], [320.0, 80.0], [320.0, 320.0], [80.0, 320.0]])

    initial_hash = hashlib.sha256(canvas.tobytes()).hexdigest()
    _ = warp_perspective(canvas, corners)
    after_hash = hashlib.sha256(canvas.tobytes()).hexdigest()

    assert initial_hash == after_hash


def test_integration_with_loaded_image_and_detection_result(tmp_path: Path):
    """Verify end-to-end flow: load_image -> detect_document_boundary -> warp_perspective."""
    # Create synthetic test file
    img_path = tmp_path / "sheet_skewed.png"
    w, h = 800, 600
    img = Image.new("RGB", (w, h), color=(30, 30, 30))

    # Draw white paper sheet
    for y in range(100, 500):
        for x in range(150, 650):
            img.putpixel((x, y), (245, 245, 245))
    img.save(img_path)

    # 1. Load image
    loaded = load_image(img_path)

    # 2. Detect boundary
    detection = detect_document_boundary(loaded)
    assert detection is not None
    assert detection.detected is True

    # 3. Warp perspective passing LoadedImage and DetectionResult directly
    warped = warp_perspective(loaded, detection)

    assert isinstance(warped, WarpedDocument)
    assert abs(warped.width - 500) < 10
    assert abs(warped.height - 400) < 10
    assert warped.channels == 3
