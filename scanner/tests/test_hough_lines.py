"""AS-43 Hough line evidence tests."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from attendance_scanner.hough_lines import (
    HoughLineConfig,
    HoughLineSegment,
    _cluster_segments,
    detect_hough_lines,
    extrapolate_line_to_bounds,
)
from attendance_scanner.pipeline.load import load_image


def _line_image(path: Path, *, border: bool = False, rotated: bool = False) -> Path:
    canvas = np.zeros((240, 320, 3), dtype=np.uint8)
    if rotated:
        cv2.line(canvas, (35, 50), (280, 170), (255, 255, 255), 3)
        cv2.line(canvas, (35, 170), (280, 50), (255, 255, 255), 3)
    else:
        left = 0 if border else 40
        cv2.line(canvas, (left, 30), (300, 30), (255, 255, 255), 3)
        cv2.line(canvas, (left, 210), (300, 210), (255, 255, 255), 3)
        cv2.line(canvas, (left, 30), (left, 210), (255, 255, 255), 3)
        cv2.line(canvas, (300, 30), (300, 210), (255, 255, 255), 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).save(path)
    return path


def test_horizontal_vertical_lines_have_normalized_diagnostics(tmp_path: Path):
    evidence = detect_hough_lines(
        load_image(_line_image(tmp_path / "lines.png")),
        config=HoughLineConfig(
            hough_threshold=20,
            min_line_length_ratio=0.2,
            max_segments=32,
        ),
    )

    assert evidence.segments
    assert any(segment.orientation == "horizontal" for segment in evidence.segments)
    assert any(segment.orientation == "vertical" for segment in evidence.segments)
    assert all(0.0 <= segment.angle_deg < 180.0 for segment in evidence.segments)
    assert all(segment.length_px > 0.0 for segment in evidence.segments)
    assert all(0.0 <= segment.support <= 1.0 for segment in evidence.segments)
    assert evidence.diagnostics["keptSegmentCount"] <= 32


def test_border_touching_lines_are_kept_and_grid_pool_is_bounded(tmp_path: Path):
    evidence = detect_hough_lines(
        load_image(_line_image(tmp_path / "border.png", border=True)),
        config=HoughLineConfig(
            hough_threshold=15,
            min_line_length_ratio=0.05,
            max_segments=12,
            max_clusters=4,
        ),
    )

    assert len(evidence.segments) <= 12
    assert len(evidence.clusters) <= 4
    assert any(segment.border_contact for segment in evidence.segments)
    assert all(cluster.segment_ids for cluster in evidence.clusters)


def test_rotated_lines_are_detected_and_result_is_deterministic(tmp_path: Path):
    image = load_image(_line_image(tmp_path / "rotated.png", rotated=True))
    config = HoughLineConfig(hough_threshold=20, min_line_length_ratio=0.2)

    first = detect_hough_lines(image, config=config)
    second = detect_hough_lines(image, config=config)

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert any(segment.orientation == "diagonal" for segment in first.segments)


def test_partial_line_can_be_extrapolated_to_image_bounds_with_limit(tmp_path: Path):
    segment = detect_hough_lines(
        load_image(_line_image(tmp_path / "lines.png")),
        config=HoughLineConfig(hough_threshold=20, min_line_length_ratio=0.2),
    ).segments[0]

    extended = extrapolate_line_to_bounds(
        segment,
        image_size=(320, 240),
        max_extrapolation_ratio=3.0,
    )

    assert extended.length_px >= segment.length_px
    assert extended.length_px <= segment.length_px * 7.0
    assert "extrapolated" in extended.border_contact


def test_clustering_is_invariant_to_reversed_endpoints_and_angle_wrap():
    config = HoughLineConfig(angle_cluster_deg=5.0, distance_cluster_px=10.0)
    segments = [
        HoughLineSegment(
            segment_id=0,
            p1=(0.0, 10.0),
            p2=(100.0, 10.0),
            angle_deg=0.0,
            length_px=100.0,
            support=0.9,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=1,
            p1=(100.0, 10.0),
            p2=(0.0, 10.0),
            angle_deg=0.0,
            length_px=100.0,
            support=0.8,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=2,
            p1=(0.0, 10.0),
            p2=(100.0, 8.25),
            angle_deg=179.0,
            length_px=100.0,
            support=0.7,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=3,
            p1=(0.0, 10.0),
            p2=(100.0, 11.75),
            angle_deg=1.0,
            length_px=100.0,
            support=0.7,
            orientation="horizontal",
        ),
    ]

    clusters = _cluster_segments(segments, 200, 200, config)

    assert len(clusters) == 1
    assert clusters[0].mean_angle_deg < 5.0 or clusters[0].mean_angle_deg > 175.0
