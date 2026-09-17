"""Safety and mapping tests for the opt-in grid dewarp stage."""

import math

import cv2
import numpy as np
import pytest

from attendance_scanner.grid_dewarp import (
    GridDewarpConfig,
    _build_inverse_maps,
    _inverse_map_1d,
    _safe_map,
    dewarp_document_grid,
)


def test_grid_dewarp_is_opt_in_and_preserves_image_when_disabled():
    image = np.full((80, 120, 3), 180, dtype=np.uint8)
    image[20:60, 30:90] = (20, 80, 160)

    result = dewarp_document_grid(image)

    assert result.applied is False
    assert result.diagnostics["reason"] == "disabled"
    assert np.array_equal(result.image, image)


def test_grid_dewarp_rejects_low_evidence_without_changing_pixels():
    image = np.full((300, 400, 3), 220, dtype=np.uint8)
    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is False
    assert result.diagnostics["reason"] in {
        "insufficient_horizontal_lines",
        "insufficient_vertical_lines",
    }
    assert np.array_equal(result.image, image)


def test_grid_dewarp_accepts_a_mild_curved_grid_with_safe_jacobian():
    height, width = 900, 1400
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for row in np.linspace(150, 780, 14):
        points = [
            (
                x,
                round(row + 18.0 * ((x - width / 2.0) / (width / 2.0)) ** 2),
            )
            for x in range(80, width - 80, 4)
        ]
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (30, 30, 30), 2)

    for column in np.linspace(100, 1300, 10):
        points = [
            (
                round(column + 14.0 * ((y - height / 2.0) / (height / 2.0)) ** 2),
                y,
            )
            for y in range(120, 820, 4)
        ]
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], False, (30, 30, 30), 2)

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is True
    assert result.diagnostics["reason"] == "accepted"
    assert float(result.diagnostics["maxDisplacementPx"]) > 0.0
    assert float(result.diagnostics["minimumJacobian"]) >= 0.15
    assert float(result.diagnostics["maximumJacobian"]) <= 4.0


def test_inverse_map_is_identity_for_straight_control_curves():
    height, width = 40, 60
    rows = [np.full(width, y, dtype=np.float32) for y in (10.0, 20.0, 30.0)]
    columns = [np.full(height, x, dtype=np.float32) for x in (15.0, 30.0, 45.0)]

    source_x, source_y = _build_inverse_maps(
        (height, width),
        rows,
        [10.0, 20.0, 30.0],
        columns,
        [15.0, 30.0, 45.0],
        iterations=1,
    )

    np.testing.assert_allclose(
        source_x,
        np.broadcast_to(np.arange(width, dtype=np.float32)[None, :], (height, width)),
    )
    np.testing.assert_allclose(
        source_y,
        np.broadcast_to(np.arange(height, dtype=np.float32)[:, None], (height, width)),
    )


def test_inverse_map_keeps_canvas_borders_in_bounds():
    axis_len, travel_len = 80, 120
    travel = np.arange(travel_len, dtype=np.float32)
    curves = [
        np.full(travel_len, target, dtype=np.float32)
        + 7.0 * np.sin(travel / max(travel_len - 1, 1) * np.pi)
        for target in (20.0, 40.0, 60.0)
    ]

    inverse = _inverse_map_1d(curves, (20.0, 40.0, 60.0), axis_len, travel_len)

    np.testing.assert_allclose(inverse[0], 0.0)
    np.testing.assert_allclose(inverse[-1], float(axis_len - 1))
    assert float(inverse.min()) >= 0.0
    assert float(inverse.max()) <= axis_len - 1.0


def test_safe_map_rejects_a_fold_or_large_local_jump():
    height, width = 20, 30
    x, y = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    x[8:, 15:] -= 4.0

    safe, valid_ratio, displacement, minimum_step, maximum_step = _safe_map(
        x,
        y,
        source_shape=(height, width),
        minimum_valid_ratio=0.98,
    )

    assert safe is False
    assert valid_ratio == 1.0
    assert displacement == 4.0
    assert minimum_step < 0.0
    assert maximum_step >= 3.0


def test_safe_map_rejects_cross_axis_tearing_even_when_axis_steps_are_monotone():
    height, width = 20, 30
    x_axis = np.arange(width, dtype=np.float32) * 0.5
    x = np.broadcast_to(x_axis, (height, width)).copy()
    x[1::2] += 3.0
    y = np.broadcast_to(
        np.arange(height, dtype=np.float32)[:, None],
        (height, width),
    ).copy()

    safe, valid_ratio, _, minimum_step, maximum_step = _safe_map(
        x,
        y,
        source_shape=(height, width),
        minimum_valid_ratio=0.98,
    )

    assert safe is False
    assert valid_ratio == 1.0
    assert minimum_step == 0.5
    assert maximum_step == 1.0


def test_grid_dewarp_config_rejects_inverted_roi():
    with pytest.raises(ValueError, match="roi_top_ratio"):
        GridDewarpConfig(roi_top_ratio=0.8, roi_bottom_ratio=0.7)


def test_synthetic_curved_mesh_orthogonal_dewarp():
    height, width = 800, 1200
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for r in np.linspace(140, 700, 12):
        pts = [
            (x, round(r + 14.0 * math.sin(x / width * math.pi))) for x in range(60, width - 60, 4)
        ]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)
    for c in np.linspace(100, 1100, 8):
        pts = [(round(c + 12.0 * math.sin(y / height * math.pi)), y) for y in range(120, 720, 4)]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is True
    assert result.diagnostics["reason"] == "accepted"
    assert result.diagnostics["horizontalLines"] >= 8
    assert result.diagnostics["verticalLines"] >= 6
    assert float(result.diagnostics["minimumJacobian"]) > 0.0


def test_arbitrary_grid_dimensions_handled_dynamically():
    height, width = 900, 1000
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for r in np.linspace(150, 800, 16):
        pts = [
            (x, round(r + 10.0 * ((x - width / 2.0) / (width / 2.0)) ** 2))
            for x in range(50, width - 50, 4)
        ]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)
    for c in np.linspace(80, 920, 6):
        pts = [
            (round(c + 8.0 * ((y - height / 2.0) / (height / 2.0)) ** 2), y)
            for y in range(130, 820, 4)
        ]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is True
    assert result.diagnostics["reason"] == "accepted"
    assert result.diagnostics["horizontalLines"] >= 12
    assert result.diagnostics["verticalLines"] >= 5


def test_fail_safe_non_table_document_returns_unmodified():
    height, width = 600, 800
    image = np.full((height, width, 3), 250, dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(40):
        x1 = rng.integers(50, 650)
        y1 = rng.integers(100, 500)
        cv2.putText(
            image,
            "Sample paragraph text sentence",
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
        )

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is False
    assert result.diagnostics["applied"] is False
    assert np.array_equal(result.image, image)


def test_color_retention_preserves_colored_ink():
    height, width = 800, 1200
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for r in np.linspace(140, 700, 12):
        pts = [
            (x, round(r + 14.0 * math.sin(x / width * math.pi))) for x in range(60, width - 60, 4)
        ]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)
    for c in np.linspace(100, 1100, 8):
        pts = [(round(c + 12.0 * math.sin(y / height * math.pi)), y) for y in range(120, 720, 4)]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (30, 30, 30), 2)

    # Put colored data near both side borders to catch canvas clipping.
    cv2.circle(image, (30, 400), 20, (0, 0, 220), -1)  # Red stamp
    cv2.circle(image, (width - 30, 400), 20, (220, 50, 20), -1)  # Blue ink

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is True
    reds = np.sum((result.image[:, :, 2] > 180) & (result.image[:, :, 0] < 80))
    blues = np.sum((result.image[:, :, 0] > 180) & (result.image[:, :, 2] < 80))
    assert reds > 500
    assert blues > 300


def test_single_pass_remap_sharpness():
    height, width = 800, 1200
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for r in np.linspace(140, 700, 10):
        pts = [
            (x, round(r + 10.0 * math.sin(x / width * math.pi))) for x in range(60, width - 60, 4)
        ]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (20, 20, 20), 2)
    for c in np.linspace(100, 1100, 8):
        pts = [(round(c + 10.0 * math.sin(y / height * math.pi)), y) for y in range(120, 720, 4)]
        cv2.polylines(image, [np.asarray(pts, dtype=np.int32)], False, (20, 20, 20), 2)

    result = dewarp_document_grid(image, config=GridDewarpConfig(enabled=True))

    assert result.applied is True
    # Output must retain sharp contrast edges
    sobel = cv2.Sobel(cv2.cvtColor(result.image, cv2.COLOR_BGR2GRAY), cv2.CV_32F, 1, 1)
    assert float(np.max(np.abs(sobel))) > 150.0
