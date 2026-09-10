"""AS-48 robust edge-line fitting tests."""

from typing import List

import numpy as np

from attendance_scanner.corner_search import CornerRoiResult, CornerSearchResult, EdgeSample
from attendance_scanner.hough_lines import HoughLineConfig, HoughLineEvidence, HoughLineSegment
from attendance_scanner.line_fitting import LineFittingConfig, fit_document_edge_lines


def _corners() -> list[tuple[float, float]]:
    return [(20.0, 20.0), (300.0, 25.0), (295.0, 220.0), (22.0, 215.0)]


def _local_samples() -> CornerSearchResult:
    corners = _corners()
    labels = ("TL", "TR", "BR", "BL")
    rois: List[CornerRoiResult] = []
    for index, label in enumerate(labels):
        samples: List[EdgeSample] = []
        for distance in (12.0, 40.0, 80.0):
            start = np.asarray(corners[index])
            end = np.asarray(corners[(index + 1) % 4])
            point = start + (end - start) * (distance / np.linalg.norm(end - start))
            samples.append(
                EdgeSample(
                    side="next",
                    point=(float(point[0]), float(point[1])),
                    distance_from_corner_px=distance,
                    normal_offset_px=0.0,
                    gradient_score=0.9,
                    edge_detected=True,
                )
            )
            previous_start = np.asarray(corners[index])
            previous_end = np.asarray(corners[(index - 1) % 4])
            previous_point = previous_start + (previous_end - previous_start) * (
                distance / np.linalg.norm(previous_end - previous_start)
            )
            samples.append(
                EdgeSample(
                    side="previous",
                    point=(float(previous_point[0]), float(previous_point[1])),
                    distance_from_corner_px=distance,
                    normal_offset_px=0.0,
                    gradient_score=0.9,
                    edge_detected=True,
                )
            )
        rois.append(
            CornerRoiResult(
                label=label,
                predicted_point=corners[index],
                roi_xywh=(0, 0, 320, 240),
                radius_px=20,
                samples=samples,
                confidence=0.9,
            )
        )
    return CornerSearchResult(source_width=320, source_height=240, radius_px=20, corners=rois)


def _hough_evidence() -> HoughLineEvidence:
    lines = [
        HoughLineSegment(
            segment_id=0,
            p1=(20, 20),
            p2=(300, 25),
            angle_deg=1,
            length_px=280,
            support=0.9,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=1,
            p1=(300, 25),
            p2=(295, 220),
            angle_deg=89,
            length_px=195,
            support=0.9,
            orientation="vertical",
        ),
        HoughLineSegment(
            segment_id=2,
            p1=(295, 220),
            p2=(22, 215),
            angle_deg=179,
            length_px=273,
            support=0.9,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=3,
            p1=(22, 215),
            p2=(20, 20),
            angle_deg=91,
            length_px=195,
            support=0.9,
            orientation="vertical",
        ),
    ]
    return HoughLineEvidence(
        source_width=320,
        source_height=240,
        detection_width=320,
        detection_height=240,
        scale_factor=1.0,
        segments=lines,
        preprocessing=HoughLineConfig(),
    )


def test_local_samples_fit_four_lines_with_residual_diagnostics():
    result = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        local_search=_local_samples(),
    )

    assert len(result.lines) == 4
    assert all(not line.fallback_used for line in result.lines)
    assert all(line.inlier_count >= 4 for line in result.lines)
    assert all(line.residual_p95_px <= 3.0 for line in result.lines)
    assert all(line.confidence > 0.5 for line in result.lines)


def test_missing_local_support_uses_global_hough_points_or_original_edge_fallback():
    hough_result = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        hough_evidence=_hough_evidence(),
        config=LineFittingConfig(minimum_inliers=2, minimum_inlier_ratio=0.1),
    )
    fallback_result = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        config=LineFittingConfig(minimum_inliers=4),
    )

    assert all(line.source == "local+hough" for line in hough_result.lines)
    assert all(not line.fallback_used for line in hough_result.lines)
    assert all(line.source == "original_edge" for line in fallback_result.lines)
    assert all(line.fallback_used for line in fallback_result.lines)
    assert all(
        np.isfinite([line.normal_a, line.normal_b, line.normal_c]).all()
        for line in fallback_result.lines
    )


def test_partial_noisy_edge_keeps_direction_and_rejects_grid_direction_flip():
    local = _local_samples()
    noisy = local.corners[0].model_copy(
        update={
            "samples": [
                *local.corners[0].samples,
                EdgeSample(
                    side="next",
                    point=(80.0, 160.0),
                    distance_from_corner_px=30.0,
                    normal_offset_px=0.0,
                    gradient_score=0.2,
                    edge_detected=True,
                ),
            ]
        }
    )
    local.corners[0] = noisy

    result = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        local_search=local,
        config=LineFittingConfig(max_residual_px=4.0),
    )

    assert result.lines[0].angle_deg < 20.0 or result.lines[0].angle_deg > 160.0
    assert result.lines[0].residual_p95_px <= 4.0


def test_disabled_fallback_returns_no_fit_and_hough_uses_configured_tolerance():
    no_fallback = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        config=LineFittingConfig(
            use_hough_fallback=False,
            use_original_edge_fallback=False,
        ),
    )
    evidence = _hough_evidence()
    evidence.segments[0] = evidence.segments[0].model_copy(update={"angle_deg": 30.0})
    strict_hough = fit_document_edge_lines(
        _corners(),
        source_size=(320, 240),
        hough_evidence=evidence,
        config=LineFittingConfig(
            minimum_inliers=2,
            direction_tolerance_deg=5.0,
        ),
    )

    assert all(line.source == "none" and not line.fallback_used for line in no_fallback.lines)
    assert strict_hough.lines[0].source == "original_edge"
    assert strict_hough.lines[0].fallback_used is True
