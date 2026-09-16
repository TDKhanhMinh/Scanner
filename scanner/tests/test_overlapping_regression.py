"""Synthetic regression test for overlapping documents and controlled diagnostics (Stage 1)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from attendance_scanner.candidate_scoring import (
    CandidateScoringConfig,
    rank_candidate_pool,
)
from attendance_scanner.cv_candidates import generate_cv_candidates
from attendance_scanner.detector import CanonicalCorners
from attendance_scanner.pipeline.load import LoadedImage, LoadedImageMetadata
from attendance_scanner.pipeline.orchestrator import scan_one
from attendance_scanner.quadrilateral_candidates import (
    QuadrilateralCandidate,
    build_quadrilateral_candidates,
)


def _create_synthetic_overlapping_image() -> np.ndarray:
    """Generate a deterministic synthetic 800x600 image with two overlapping sheets.

    Uses a seeded random generator (seed=42) for deterministic background texture.
    - Background: Gray textured noise (simulating tabletop/mat).
    - Sheet 1 (Underlying): White sheet at the top-left (from (80, 50) to (450, 420)).
    - Sheet 2 (Topmost): Attendance sheet with grid table (from (180, 120) to (720, 540)).
    """
    canvas = np.full((600, 800, 3), 70, dtype=np.uint8)
    rng = np.random.default_rng(42)
    noise = rng.integers(-10, 10, canvas.shape, dtype=np.int16)
    canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Underlying paper (partially covered)
    under_poly = np.array([[80, 50], [450, 60], [420, 420], [60, 400]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, under_poly, (240, 238, 235))
    cv2.polylines(canvas, [under_poly], True, (160, 155, 150), 2)

    # Topmost paper (overlapping on top)
    top_poly = np.array([[180, 120], [720, 130], [710, 530], [170, 510]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, top_poly, (250, 250, 250))
    cv2.polylines(canvas, [top_poly], True, (140, 140, 140), 2)

    # Table grid inside topmost paper
    for y in range(180, 490, 40):
        cv2.line(canvas, (210, y), (680, y), (80, 80, 80), 1)
    for x in range(210, 690, 80):
        cv2.line(canvas, (x, 180), (x, 460), (80, 80, 80), 1)

    return canvas


def test_synthetic_overlapping_fixture_creation():
    image = _create_synthetic_overlapping_image()
    assert image.shape == (600, 800, 3)
    assert image.dtype == np.uint8
    # Point between table grid lines should be white paper
    assert int(image[200, 300, 0]) >= 240


def test_overlapping_candidate_diagnostics_observability():
    """Verify that candidate scoring diagnostics record delta, contributions, and corners."""
    # Synthetic candidate 1: wider mask_fit that encompasses both sheets
    c1 = QuadrilateralCandidate(
        candidate_id=1,
        corners=CanonicalCorners.from_sequence([(75, 45), (725, 125), (715, 535), (55, 405)]),
        source="mask_fit",
        score=0.0,
        mask_quad_iou=0.92,
        edge_support=0.65,
        geometry_quality=0.75,
        evidence={
            "fitMethod": "convex_hull",
            "quadAreaRatio": 0.65,
            "maskCoverage": 0.94,
        },
    )

    # Synthetic candidate 2: contour that hugs the topmost sheet
    c2 = QuadrilateralCandidate(
        candidate_id=2,
        corners=CanonicalCorners.from_sequence([(180, 120), (720, 130), (710, 530), (170, 510)]),
        source="contour",
        score=0.0,
        mask_quad_iou=0.78,
        edge_support=0.88,
        geometry_quality=0.92,
        evidence={
            "fitMethod": "topmost_contour",
            "quadAreaRatio": 0.48,
            "maskCoverage": 0.79,
        },
    )

    config = CandidateScoringConfig(debug_diagnostics=True)
    result = rank_candidate_pool([c1, c2], config=config)

    # 1. Observability: scoreDelta is explicitly measured and > 0
    assert "scoreDelta" in result.diagnostics
    assert result.diagnostics["scoreDelta"] is not None
    assert result.diagnostics["scoreDelta"] > 0

    # 2. Detailed candidate diagnostics are present and IPC-safe
    assert "candidates" in result.diagnostics
    diag_cands = result.diagnostics["candidates"]
    assert len(diag_cands) == 2

    for item in diag_cands:
        assert "rank" in item
        assert "candidateId" in item
        assert "source" in item
        assert "fitMethod" in item
        assert "finalScore" in item
        assert "contributions" in item
        assert "corners" in item
        assert "geometryQuality" in item
        assert "mask_iou" in item["contributions"]
        assert isinstance(item["contributions"]["mask_iou"], float)

    methods = {item["candidateId"]: item["fitMethod"] for item in diag_cands}
    assert methods[1] == "convex_hull"
    assert methods[2] == "topmost_contour"


def test_overlapping_candidate_generation_and_pipeline_flow():
    """Verify end-to-end candidate generation and diagnostic propagation through scan_one."""
    image_arr = _create_synthetic_overlapping_image()
    loaded = LoadedImage(
        path=Path("<memory>"),
        image=image_arr,
        metadata=LoadedImageMetadata(
            original_width=800,
            original_height=600,
            width=800,
            height=600,
            original_mode="BGR",
            normalized_mode="BGR",
            has_transparency=False,
        ),
    )

    # 1. Test CV candidate generation from actual image pixels
    cv_candidates = generate_cv_candidates(loaded)
    assert len(cv_candidates.candidates) > 0

    # 2. Build candidate pool using actual CV candidate set and a combined paper mask
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)
    pool = build_quadrilateral_candidates(
        image_size=(800, 600),
        mask=mask,
        cv_candidates=cv_candidates,
    )
    assert len(pool.candidates) > 0

    # 3. Test scoring on the generated candidates with debug diagnostics enabled
    debug_scoring = CandidateScoringConfig(debug_diagnostics=True)
    ranking = rank_candidate_pool(pool.candidates, config=debug_scoring)
    assert ranking.status in {"selected", "ambiguous"}
    assert "scoreDelta" in ranking.diagnostics
    assert "candidates" in ranking.diagnostics
    assert len(ranking.diagnostics["candidates"]) == len(ranking.ranked_candidates)

    # 4. Test scan_one pipeline execution with debug_diagnostics=True
    scan_result = scan_one(
        loaded,
        detector_mode="cv_v2",
        debug_diagnostics=True,
        force_preview=True,
    )
    # Check quality summary propagation
    assert "scoreDelta" in scan_result.detection_quality_summary
    assert "decisionPath" in scan_result.detection_quality_summary
    assert "rankingStatus" in scan_result.detection_quality_summary

    # Check DetectionPreview propagation
    assert scan_result.detection_preview is not None
    assert hasattr(scan_result.detection_preview, "score_delta")
    for cand in scan_result.detection_preview.candidate_corners:
        assert hasattr(cand, "fit_method")
        assert hasattr(cand, "rank")
        assert hasattr(cand, "contributions")
        if cand.rank is not None:
            assert isinstance(cand.rank, int)
            assert cand.rank >= 1
        if cand.contributions:
            assert isinstance(cand.contributions, dict)
            for k, v in cand.contributions.items():
                assert isinstance(k, str)
                assert isinstance(v, float)


def test_cv_candidate_generation_isolates_topmost_sheet():
    """Verify that OpenCV contour candidate generation detects the topmost sheet quad."""
    image_arr = _create_synthetic_overlapping_image()
    loaded = LoadedImage(
        path=Path("<memory>"),
        image=image_arr,
        metadata=LoadedImageMetadata(
            original_width=800,
            original_height=600,
            width=800,
            height=600,
            original_mode="BGR",
            normalized_mode="BGR",
            has_transparency=False,
        ),
    )
    cv_candidates = generate_cv_candidates(loaded)
    assert len(cv_candidates.candidates) > 0

    top_poly = np.array([[180, 120], [720, 130], [710, 530], [170, 510]], dtype=np.float32)
    top_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.fillConvexPoly(top_mask, np.round(top_poly).astype(np.int32), 1)

    cv_ious = []
    for cand in cv_candidates.candidates:
        cand_poly = np.array(cand.corners.points, dtype=np.float32)
        cand_mask = np.zeros((600, 800), dtype=np.uint8)
        cv2.fillConvexPoly(cand_mask, np.round(cand_poly).astype(np.int32), 1)
        intersection = np.logical_and(top_mask, cand_mask).sum()
        union = np.logical_or(top_mask, cand_mask).sum()
        cv_ious.append(float(intersection / max(union, 1)))

    best_cv_iou = max(cv_ious)
    # OpenCV edge contours isolate the topmost sheet with extremely high fidelity (>= 0.95 IoU)
    assert best_cv_iou >= 0.95


def test_underlying_paper_candidate_scoring_and_penalization():
    """Verify that an underlying paper candidate is scored below topmost candidate."""
    # Topmost sheet
    top_cand = QuadrilateralCandidate(
        candidate_id=1,
        corners=CanonicalCorners.from_sequence([(180, 120), (720, 130), (710, 530), (170, 510)]),
        source="contour",
        score=0.0,
        mask_quad_iou=0.78,
        edge_support=0.88,
        geometry_quality=0.92,
        evidence={"fitMethod": "topmost_contour", "quadAreaRatio": 0.48, "maskCoverage": 0.79},
    )
    # Underlying sheet (partially covered, lower edge support on occluded side)
    under_cand = QuadrilateralCandidate(
        candidate_id=2,
        corners=CanonicalCorners.from_sequence([(80, 50), (450, 60), (420, 420), (60, 400)]),
        source="contour",
        score=0.0,
        mask_quad_iou=0.35,
        edge_support=0.45,
        geometry_quality=0.68,
        evidence={"fitMethod": "contour_approx", "quadAreaRatio": 0.28, "maskCoverage": 0.38},
    )

    config = CandidateScoringConfig(debug_diagnostics=True)
    ranking = rank_candidate_pool([top_cand, under_cand], config=config)

    assert ranking.status == "selected"
    assert ranking.selected_candidate_id == 1  # Topmost sheet wins over underlying sheet
    assert ranking.top_score is not None and ranking.second_score is not None
    assert ranking.top_score > ranking.second_score
    assert ranking.diagnostics["scoreDelta"] > 0.15


def test_hybrid_detector_verbose_diagnostics_and_candidate_rankings():
    """Verify that HybridDocumentDetector surfaces candidate_rankings and wire diagnostics."""
    from attendance_scanner.hybrid import HybridConfig, HybridDocumentDetector

    image_arr = _create_synthetic_overlapping_image()
    loaded = LoadedImage(
        path=Path("<memory>"),
        image=image_arr,
        metadata=LoadedImageMetadata(
            original_width=800,
            original_height=600,
            width=800,
            height=600,
            original_mode="BGR",
            normalized_mode="BGR",
            has_transparency=False,
        ),
    )
    cfg = HybridConfig(
        segmentation_enabled=False,  # Test CV rescue mode with debug diagnostics
        scoring=CandidateScoringConfig(debug_diagnostics=True),
    )
    detector = HybridDocumentDetector(config=cfg)
    result = detector.detect(loaded)

    assert result.detected is True
    assert len(result.candidate_rankings) > 0
    for cand_info in result.candidate_rankings:
        assert cand_info.candidate_id >= 0
        assert cand_info.rank >= 1
        assert 0.0 <= cand_info.final_score <= 1.0
        assert isinstance(cand_info.contributions, dict)

    # Verify CandidateCorners diagnostics
    assert len(result.candidate_corners) > 0
    for cand in result.candidate_corners:
        assert "candidateScore" in cand.diagnostics
        assert "geometryQuality" in cand.diagnostics
        # Check rank was attached
        assert "rank" in cand.diagnostics

