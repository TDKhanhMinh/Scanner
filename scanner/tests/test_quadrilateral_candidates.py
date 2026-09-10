"""AS-44 bounded multi-source quadrilateral candidate tests."""

import cv2
import numpy as np

from attendance_scanner.cv_candidates import CvCandidate, CvCandidateConfig, CvCandidateSet
from attendance_scanner.detector import CandidateCorners
from attendance_scanner.geometry_validator import GeometryReasonCode
from attendance_scanner.hough_lines import HoughLineConfig, HoughLineEvidence, HoughLineSegment
from attendance_scanner.quadrilateral_candidates import (
    QuadrilateralCandidateConfig,
    build_quadrilateral_candidates,
)
from attendance_scanner.segmentation import SegmentationTransform


def _document_mask(width: int = 140, height: int = 100) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.array([[20, 10], [120, 14], [110, 88], [25, 92]], dtype=np.int32)],
        1,
    )
    return mask


def _contour_set(points: list[tuple[float, float]]) -> CvCandidateSet:
    corners = CandidateCorners(
        points=points,
        source="cv_contour",
        confidence=0.8,
    )
    candidate = CvCandidate(
        candidate_id=0,
        corners=corners,
        contour_area_px=6400.0,
        area_ratio=0.4,
        perimeter_px=320.0,
        convex=True,
        confidence=0.8,
        source_contour_index=1,
        approximation_epsilon_ratio=0.02,
    )
    return CvCandidateSet(
        source_width=140,
        source_height=100,
        detection_width=140,
        detection_height=100,
        scale_factor=1.0,
        candidates=[candidate],
        preprocessing=CvCandidateConfig(),
    )


def _line_evidence() -> HoughLineEvidence:
    lines = [
        HoughLineSegment(
            segment_id=0,
            p1=(20, 10),
            p2=(120, 10),
            angle_deg=0,
            length_px=100,
            support=0.9,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=1,
            p1=(20, 90),
            p2=(120, 90),
            angle_deg=0,
            length_px=100,
            support=0.9,
            orientation="horizontal",
        ),
        HoughLineSegment(
            segment_id=2,
            p1=(20, 10),
            p2=(20, 90),
            angle_deg=90,
            length_px=80,
            support=0.9,
            orientation="vertical",
        ),
        HoughLineSegment(
            segment_id=3,
            p1=(120, 10),
            p2=(120, 90),
            angle_deg=90,
            length_px=80,
            support=0.9,
            orientation="vertical",
        ),
    ]
    return HoughLineEvidence(
        source_width=140,
        source_height=100,
        detection_width=140,
        detection_height=100,
        scale_factor=1.0,
        segments=lines,
        preprocessing=HoughLineConfig(),
    )


def test_mask_only_candidate_has_mask_quad_iou_evidence():
    result = build_quadrilateral_candidates(
        image_size=(140, 100),
        mask=_document_mask(),
    )

    assert result.candidates
    assert any(candidate.source == "mask_fit" for candidate in result.candidates)
    assert max(candidate.mask_quad_iou or 0.0 for candidate in result.candidates) > 0.8
    assert len(result.candidates) <= 16


def test_contour_only_and_line_only_sources_are_supported():
    contour_result = build_quadrilateral_candidates(
        image_size=(140, 100),
        cv_candidates=_contour_set([(20, 10), (120, 10), (120, 90), (20, 90)]),
    )
    line_result = build_quadrilateral_candidates(
        image_size=(140, 100),
        hough_evidence=_line_evidence(),
    )

    assert contour_result.source_counts == {"contour": 1}
    assert line_result.source_counts == {"hough": 1}
    assert line_result.candidates[0].evidence["lineIds"]


def test_mixed_candidate_and_duplicate_quads_are_bounded_and_deduplicated():
    contour = _contour_set([(30, 20), (110, 20), (110, 80), (30, 80)])
    result = build_quadrilateral_candidates(
        image_size=(140, 100),
        mask=_document_mask(),
        cv_candidates=contour,
        config=QuadrilateralCandidateConfig(
            max_candidates=4,
            dedup_corner_distance_px=1.0,
            dedup_polygon_iou=0.99,
        ),
    )

    assert len(result.candidates) <= 4
    assert "mask_fit" in result.source_counts
    assert "contour" in result.source_counts
    assert "mixed" in result.source_counts
    assert len({candidate.candidate_id for candidate in result.candidates}) == len(
        result.candidates
    )


def test_detection_size_mask_maps_to_original_size_before_fitting():
    detection_mask = np.zeros((40, 50), dtype=np.uint8)
    cv2.rectangle(detection_mask, (5, 5), (45, 35), 1, -1)
    transform = SegmentationTransform(
        source_width=100,
        source_height=80,
        input_width=50,
        input_height=40,
        scale_x=0.5,
        scale_y=0.5,
        resized_width=50,
        resized_height=40,
    )

    result = build_quadrilateral_candidates(
        image_size=(100, 80),
        mask=detection_mask,
        transform=transform,
    )

    assert result.image_width == 100
    assert result.image_height == 80
    assert result.candidates
    assert max(point[0] for point in result.candidates[0].corners.as_list()) > 80


def test_rejected_tiny_contour_remains_bounded_evidence_for_downstream_scoring():
    tiny = _contour_set([(10, 10), (20, 10), (20, 20), (10, 20)])

    result = build_quadrilateral_candidates(
        image_size=(200, 200),
        cv_candidates=tiny,
        config=QuadrilateralCandidateConfig(max_rejected_candidates=2),
    )

    assert result.candidates == []
    assert len(result.rejected_candidates) == 1
    rejected = result.rejected_candidates[0]
    assert rejected.source == "contour"
    assert GeometryReasonCode.QUAD_AREA_OUT_OF_RANGE in rejected.reason_codes
    assert result.diagnostics["rejectedCandidateCount"] == 1
