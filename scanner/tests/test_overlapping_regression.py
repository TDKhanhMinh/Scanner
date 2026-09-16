"""Synthetic regression test for overlapping documents and controlled diagnostics (Stage 1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from attendance_scanner.candidate_scoring import (
    CandidateScoringConfig,
    rank_candidate_pool,
)
from attendance_scanner.cv_candidates import CvCandidate, CvCandidateSet, generate_cv_candidates
from attendance_scanner.detector import CandidateCorners, CanonicalCorners
from attendance_scanner.edge_support import build_edge_map
from attendance_scanner.occlusion_evidence import (
    detect_occlusion_evidence,
    evaluate_candidate_occlusion,
)
from attendance_scanner.pipeline.load import LoadedImage, LoadedImageMetadata
from attendance_scanner.pipeline.orchestrator import scan_one
from attendance_scanner.quadrilateral_candidates import (
    QuadrilateralCandidate,
    QuadrilateralCandidateConfig,
    _deduplicate_candidates,
    _quad_iou,
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


def test_topmost_candidate_generation_and_dedup_protection():
    """Verify Stage 2: topmost candidates are generated and protected from deduplication."""
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
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)

    pool = build_quadrilateral_candidates(
        image_size=(800, 600),
        mask=mask,
        cv_candidates=cv_candidates,
    )

    top_poly = np.array([[180, 120], [720, 130], [710, 530], [170, 510]], dtype=np.float32)
    top_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.fillConvexPoly(top_mask, np.round(top_poly).astype(np.int32), 1)

    topmost_candidates = [
        c
        for c in pool.candidates
        if c.evidence.get("fitMethod") in {"topmost_contour", "topmost_submask"}
    ]
    # 1. At least one topmost candidate is generated and retained in pool
    assert len(topmost_candidates) >= 1

    # 2. At least one topmost candidate has IoU >= 0.90 with ground truth topmost sheet
    topmost_ious = []
    for cand in topmost_candidates:
        cand_poly = np.array(cand.corners.as_list(), dtype=np.float32)
        cand_mask = np.zeros((600, 800), dtype=np.uint8)
        cv2.fillConvexPoly(cand_mask, np.round(cand_poly).astype(np.int32), 1)
        intersection = np.logical_and(top_mask, cand_mask).sum()
        union = np.logical_or(top_mask, cand_mask).sum()
        topmost_ious.append(float(intersection / max(union, 1)))

    assert max(topmost_ious) >= 0.90

    # 3. Both full-mask candidate and topmost candidate coexist in pool
    fit_methods = {c.evidence.get("fitMethod") for c in pool.candidates}
    assert "min_area_rect" in fit_methods
    assert any(m in fit_methods for m in {"topmost_contour", "topmost_submask"})


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


def test_competing_hypothesis_deduplication_isolation():
    """Verify that two same-hypothesis candidates at IoU ~0.88 dedup into 1,
    whereas a topmost candidate at the same IoU is protected (yielding 2 candidates),
    tested through both _deduplicate_candidates and build_quadrilateral_candidates."""
    cA = CanonicalCorners.from_sequence([(50, 50), (750, 50), (750, 550), (50, 550)])
    cB = CanonicalCorners.from_sequence([(50, 50), (750, 50), (750, 550), (220, 550)])
    cC = CanonicalCorners.from_sequence([(135, 50), (750, 50), (750, 550), (135, 550)])

    iou_ab = _quad_iou(cA, cB, 800, 600)
    iou_ac = _quad_iou(cA, cC, 800, 600)
    assert 0.85 <= iou_ab <= 0.92
    assert 0.85 <= iou_ac <= 0.92

    candA = QuadrilateralCandidate(
        candidate_id=1,
        corners=cA,
        source="mask_fit",
        score=0.85,
        geometry_quality=0.9,
        evidence={"fitMethod": "min_area_rect"},
    )
    candB = QuadrilateralCandidate(
        candidate_id=2,
        corners=cB,
        source="mask_fit",
        score=0.80,
        geometry_quality=0.85,
        evidence={"fitMethod": "convex_hull_approx"},
    )
    candC = QuadrilateralCandidate(
        candidate_id=3,
        corners=cC,
        source="mask_fit",
        score=0.75,
        geometry_quality=0.88,
        evidence={"fitMethod": "topmost_submask"},
    )

    policy = QuadrilateralCandidateConfig(
        dedup_corner_distance_px=8.0,
        dedup_same_source_iou=0.85,
    )

    # 1. Direct deduplication testing
    # Scenario 1: Same hypothesis (full-mask family) -> deduplicated to 1 candidate
    pool_ab = _deduplicate_candidates(
        [candA, candB], policy=policy, image_width=800, image_height=600
    )
    assert len(pool_ab) == 1

    # Scenario 2: Competing hypothesis (full-mask vs topmost) -> protected, 2 candidates survive
    pool_ac = _deduplicate_candidates(
        [candA, candC], policy=policy, image_width=800, image_height=600
    )
    assert len(pool_ac) == 2

    # 2. Testing via full build_quadrilateral_candidates pipeline with CvCandidateSet
    cv_c1 = CvCandidate(
        candidate_id=1,
        corners=CandidateCorners(points=cA.as_list(), source="contour"),
        contour_area_px=350000.0,
        area_ratio=0.72,
        perimeter_px=2400.0,
        convex=True,
        confidence=0.8,
        source_contour_index=0,
        approximation_epsilon_ratio=0.02,
    )
    cv_c2 = CvCandidate(
        candidate_id=2,
        corners=CandidateCorners(points=cB.as_list(), source="contour"),
        contour_area_px=320000.0,
        area_ratio=0.66,
        perimeter_px=2300.0,
        convex=True,
        confidence=0.75,
        source_contour_index=1,
        approximation_epsilon_ratio=0.02,
    )
    cv_same = CvCandidateSet(
        source_width=800,
        source_height=600,
        detection_width=800,
        detection_height=600,
        scale_factor=1.0,
        candidates=[cv_c1, cv_c2],
        preprocessing={"method": "bilateral"},
    )
    pool_same = build_quadrilateral_candidates(
        image_size=(800, 600),
        cv_candidates=cv_same,
        config=policy,
    )
    assert len(pool_same.candidates) == 1

    # Overlapping mask pipeline generates both min_area_rect and topmost_submask, both surviving
    test_mask = np.zeros((600, 800), dtype=np.uint8)
    cv2.fillConvexPoly(test_mask, np.array([[80, 50], [450, 60], [420, 420], [60, 400]]), 1)
    cv2.fillConvexPoly(test_mask, np.array([[180, 120], [720, 130], [710, 530], [170, 510]]), 1)
    mask_pool = build_quadrilateral_candidates(
        image_size=(800, 600),
        mask=test_mask,
        config=policy,
    )
    pool_methods = {c.evidence.get("fitMethod") for c in mask_pool.candidates}
    assert "min_area_rect" in pool_methods
    assert "topmost_submask" in pool_methods


def test_occlusion_ridge_detection_and_continuity():
    """Verify that detect_occlusion_evidence identifies the physical ridge
    connecting T-junctions."""
    image_arr = _create_synthetic_overlapping_image()
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)
    edge_map = build_edge_map(image_arr)

    evidence = detect_occlusion_evidence(image_arr, mask=mask, edge_map=edge_map)
    assert evidence.has_overlapping_cues is True
    assert len(evidence.verified_ridges) >= 1
    ridge = evidence.verified_ridges[0]
    assert ridge.is_verified is True
    assert ridge.edge_support >= 0.35
    assert ridge.continuity_ratio >= 0.60
    assert len(evidence.t_junctions) >= 2


def test_negative_table_lines_not_flagged_as_occlusion():
    """Verify that a document page with table lines and no overlapping sheet
    produces NO occlusion ridges."""
    canvas = np.full((600, 800, 3), 70, dtype=np.uint8)
    # Single page with rectangular border
    page_poly = np.array([[100, 80], [700, 80], [700, 520], [100, 520]], dtype=np.int32)
    cv2.fillConvexPoly(canvas, page_poly, (250, 250, 250))
    cv2.polylines(canvas, [page_poly], True, (120, 120, 120), 2)
    # Internal table lines
    for y in range(140, 500, 40):
        cv2.line(canvas, (130, y), (670, y), (80, 80, 80), 2)
    for x in range(180, 650, 80):
        cv2.line(canvas, (x, 140), (x, 480), (80, 80, 80), 2)

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)
    edge_map = build_edge_map(canvas)

    evidence = detect_occlusion_evidence(canvas, mask=mask, edge_map=edge_map)
    assert evidence.has_overlapping_cues is False
    assert len(evidence.verified_ridges) == 0


def test_segment_level_candidate_occlusion_evaluation():
    """Verify segment-level containment distinguishes full-mask from topmost candidate."""
    image_arr = _create_synthetic_overlapping_image()
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)
    edge_map = build_edge_map(image_arr)
    evidence = detect_occlusion_evidence(image_arr, mask=mask, edge_map=edge_map)
    assert evidence.has_overlapping_cues is True

    # Full-mask candidate covering both sheets
    full_cand = CanonicalCorners.from_sequence([(60, 50), (720, 60), (710, 530), (60, 510)])
    ev_full = evaluate_candidate_occlusion(full_cand, evidence)
    assert ev_full["enclosed_occlusion_ridges"] >= 1
    assert ev_full["enclosed_occlusion_length"] > 50.0
    assert ev_full["aligns_with_occlusion_ridge"] is False

    # Topmost candidate closely aligning with the dividing ridge
    top_cand = CanonicalCorners.from_sequence([(180, 120), (720, 130), (710, 530), (170, 510)])
    ev_top = evaluate_candidate_occlusion(top_cand, evidence)
    assert ev_top["enclosed_occlusion_ridges"] == 0
    assert ev_top["aligns_with_occlusion_ridge"] is True


def test_single_instance_edge_map_reuse():
    """Verify that HybridDocumentDetector shares a single EdgeMap instance
    across Hough, EdgeMap, and Occlusion."""
    import attendance_scanner.hybrid as hybrid_module
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
        segmentation_enabled=False,
        scoring=CandidateScoringConfig(debug_diagnostics=True),
    )
    detector = HybridDocumentDetector(config=cfg)

    original_build_edge_map = hybrid_module.build_edge_map
    edge_map_instances = []

    def spy_build_edge_map(*args, **kwargs):
        em = original_build_edge_map(*args, **kwargs)
        edge_map_instances.append(em)
        return em

    with patch.object(hybrid_module, "build_edge_map", side_effect=spy_build_edge_map):
        result = detector.detect(loaded)

    assert result.detected is True
    # EdgeMap must be created exactly ONCE per detection cycle
    assert len(edge_map_instances) == 1
    shared_em = edge_map_instances[0]
    assert shared_em.scale_factor > 0.0
    assert shared_em.detection_width > 0
    assert shared_em.detection_height > 0
    # Preview candidates contain candidateScore or diagnostics
    for cand in result.candidate_corners:
        assert "candidateScore" in cand.diagnostics


def test_occlusion_evidence_calibration_matrix():
    """Test calibration matrix across 6 diverse conditions:
    1. Positive overlapping ridge (straight or corner) -> detected.
    2. Table grid lines -> negative (no false positive).
    3. Page crease -> negative (no re-entrant defect endpoints).
    4. Shadow edge gradient -> negative.
    5. Text / signature -> negative.
    6. Downscaled / low-resolution image -> positive retained adaptively.
    """
    # 1. Positive overlapping ridge
    image_arr = _create_synthetic_overlapping_image()
    gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 120).astype(np.uint8)
    em_pos = build_edge_map(image_arr)
    ev_pos = detect_occlusion_evidence(image_arr, mask=mask, edge_map=em_pos)
    assert ev_pos.has_overlapping_cues is True
    assert len(ev_pos.verified_ridges) >= 1

    # Base page for single-document negatives
    canvas_base = np.full((600, 800, 3), 70, dtype=np.uint8)
    page_rect = np.array([[100, 80], [700, 80], [700, 520], [100, 520]], dtype=np.int32)
    cv2.fillConvexPoly(canvas_base, page_rect, (250, 250, 250))
    cv2.polylines(canvas_base, [page_rect], True, (120, 120, 120), 2)

    # 2. Table grid (negative)
    canvas_grid = canvas_base.copy()
    for y in range(140, 500, 40):
        cv2.line(canvas_grid, (130, y), (670, y), (80, 80, 80), 2)
    for x in range(180, 650, 80):
        cv2.line(canvas_grid, (x, 140), (x, 480), (80, 80, 80), 2)
    mask_grid = (cv2.cvtColor(canvas_grid, cv2.COLOR_BGR2GRAY) > 120).astype(np.uint8)
    em_grid = build_edge_map(canvas_grid)
    ev_grid = detect_occlusion_evidence(canvas_grid, mask=mask_grid, edge_map=em_grid)
    assert ev_grid.has_overlapping_cues is False
    assert len(ev_grid.verified_ridges) == 0

    # 3. Page crease (negative)
    canvas_crease = canvas_base.copy()
    cv2.line(canvas_crease, (100, 300), (700, 300), (200, 200, 200), 1)
    mask_crease = (cv2.cvtColor(canvas_crease, cv2.COLOR_BGR2GRAY) > 120).astype(np.uint8)
    em_crease = build_edge_map(canvas_crease)
    ev_crease = detect_occlusion_evidence(canvas_crease, mask=mask_crease, edge_map=em_crease)
    assert ev_crease.has_overlapping_cues is False
    assert len(ev_crease.verified_ridges) == 0

    # 4. Shadow edge gradient (negative)
    canvas_shadow = canvas_base.copy()
    for x in range(300, 500):
        factor = 1.0 - 0.25 * ((x - 300) / 200.0)
        canvas_shadow[80:520, x] = np.clip(
            canvas_shadow[80:520, x] * factor, 0, 255
        ).astype(np.uint8)
    mask_shadow = (cv2.cvtColor(canvas_shadow, cv2.COLOR_BGR2GRAY) > 120).astype(np.uint8)
    em_shadow = build_edge_map(canvas_shadow)
    ev_shadow = detect_occlusion_evidence(canvas_shadow, mask=mask_shadow, edge_map=em_shadow)
    assert ev_shadow.has_overlapping_cues is False
    assert len(ev_shadow.verified_ridges) == 0

    # 5. Text / signature strokes (negative)
    canvas_text = canvas_base.copy()
    for y in range(150, 480, 25):
        cv2.putText(
            canvas_text,
            "Attendance Scanner Record Verification",
            (120, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 40, 40),
            1,
        )
    mask_text = (cv2.cvtColor(canvas_text, cv2.COLOR_BGR2GRAY) > 120).astype(np.uint8)
    em_text = build_edge_map(canvas_text)
    ev_text = detect_occlusion_evidence(canvas_text, mask=mask_text, edge_map=em_text)
    assert ev_text.has_overlapping_cues is False
    assert len(ev_text.verified_ridges) == 0

    # 6. Downscaled / low-resolution image (positive preserved)
    downscaled = cv2.resize(image_arr, (400, 300), interpolation=cv2.INTER_AREA)
    mask_down = (cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY) > 120).astype(np.uint8)
    em_down = build_edge_map(downscaled)
    ev_down = detect_occlusion_evidence(downscaled, mask=mask_down, edge_map=em_down)
    assert ev_down.has_overlapping_cues is True
    assert len(ev_down.verified_ridges) >= 1


