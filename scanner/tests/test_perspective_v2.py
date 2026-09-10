"""AS-50 crop-safe Perspective V2 tests."""

import cv2
import numpy as np

from attendance_scanner.pipeline.perspective import (
    PerspectiveV2Config,
    warp_perspective_v2,
)


def test_v2_infers_geometry_without_forcing_a4_ratio():
    image = np.full((500, 700, 3), 240, dtype=np.uint8)
    corners = [(100, 80), (600, 90), (580, 420), (110, 410)]

    result = warp_perspective_v2(image, corners)

    assert result.fallback_used is False
    assert abs(result.width / result.height - 500 / 330) < 0.03
    assert result.diagnostics["aspectRatioInferred"] is True
    assert np.isfinite(result.transform_matrix).all()


def test_v2_orientation_policy_and_no_upscale():
    image = np.full((600, 400, 3), 240, dtype=np.uint8)
    corners = [(50, 50), (350, 50), (350, 550), (50, 550)]

    natural = warp_perspective_v2(image, corners)
    landscape = warp_perspective_v2(
        image,
        corners,
        PerspectiveV2Config(preferred_orientation="landscape"),
    )

    assert natural.width < natural.height
    assert landscape.width > landscape.height
    assert landscape.diagnostics["orientationRotationDegrees"] == 90
    assert natural.width <= image.shape[1]
    assert natural.height <= image.shape[0]
    projected = cv2.perspectiveTransform(
        landscape.source_corners.reshape(1, 4, 2), landscape.transform_matrix
    ).reshape(4, 2)
    expected = np.array(
        [
            [0.0, 0.0],
            [landscape.width - 1.0, 0.0],
            [landscape.width - 1.0, landscape.height - 1.0],
            [0.0, landscape.height - 1.0],
        ],
        dtype=np.float32,
    )
    projected = np.round(projected, 6)
    projected_sorted = projected[np.lexsort((projected[:, 1], projected[:, 0]))]
    expected_sorted = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    assert np.max(np.abs(projected_sorted - expected_sorted)) < 1e-3


def test_v2_near_border_quad_is_valid_and_margin_is_bounded():
    image = np.full((300, 400, 3), 220, dtype=np.uint8)
    corners = [(0, 0), (399, 3), (395, 299), (2, 295)]

    result = warp_perspective_v2(
        image,
        corners,
        PerspectiveV2Config(safety_margin_px=4, max_dimension=1000),
    )

    assert result.fallback_used is False
    assert result.width <= 1000
    assert result.height <= 1000
    assert result.diagnostics["safetyMarginPx"] == 4


def test_v2_invalid_or_oversized_transform_falls_back_to_full_image():
    image = np.full((100, 120, 3), 180, dtype=np.uint8)
    original = image.copy()
    degenerate = warp_perspective_v2(
        image,
        [(10, 10), (20, 10), (30, 10), (40, 10)],
    )
    oversized = warp_perspective_v2(
        image,
        [(0, 0), (119, 0), (119, 99), (0, 99)],
        PerspectiveV2Config(max_pixels=1000),
    )

    for result in (degenerate, oversized):
        assert result.fallback_used is True
        assert result.warning == "PERSPECTIVE_V2_FALLBACK_FULL_IMAGE"
        assert result.transform_matrix.shape == (3, 3)
        assert result.source_corners.shape == (0, 2)
        assert result.image.shape == image.shape
        assert result.diagnostics["fallbackReason"]
    assert np.array_equal(image, original)


def test_v2_accepts_canonical_detection_result_and_preserves_content():
    image = np.full((120, 160, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 100), (240, 240, 240), 2)
    result = warp_perspective_v2(
        image,
        [(20, 20), (140, 20), (140, 100), (20, 100)],
        PerspectiveV2Config(safety_margin_px=2),
    )

    assert result.fallback_used is False
    assert result.image.max() > 200
    assert result.source_corners.shape == (4, 2)
