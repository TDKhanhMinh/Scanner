import math
from typing import Optional

from attendance_scanner.candidate_scoring import (
    CandidateScoreWeights,
    CandidateScoringConfig,
    rank_candidate_pool,
    score_candidate,
    sweep_candidate_scores,
)
from attendance_scanner.detector import CanonicalCorners
from attendance_scanner.quadrilateral_candidates import QuadrilateralCandidate


def compute_legacy_base_score(
    candidate: QuadrilateralCandidate,
    *,
    config: Optional[CandidateScoringConfig] = None,
    mask_confidence: Optional[float] = None,
    aspect_hint: Optional[float] = None,
) -> float:
    """Test oracle computing candidate score using strictly the 9 legacy base components."""
    policy = config or CandidateScoringConfig()
    weights = policy.weights
    total_weight = sum(weights.model_dump().values())
    border_val = 1.0
    if "borderContact" in candidate.evidence:
        border_val = policy.border_touch_score if candidate.evidence["borderContact"] else 1.0
    convexity = 1.0
    if "convex" in candidate.evidence and candidate.evidence["convex"] is not None:
        convexity = float(candidate.evidence["convex"])
    source_val = policy.source_reliability.get(candidate.source, 0.5)

    def _comp(val: Optional[float], weight: float, neutral: float) -> float:
        v = val if val is not None else neutral
        norm_w = weight / total_weight if total_weight else 0.0
        return v * norm_w

    aspect_score: Optional[float] = None
    if aspect_hint is not None:
        pts = candidate.corners.as_list()
        w = (math.dist(pts[0], pts[1]) + math.dist(pts[2], pts[3])) / 2.0
        h = (math.dist(pts[1], pts[2]) + math.dist(pts[3], pts[0])) / 2.0
        if h > 1e-9:
            ratio = w / h
            aspect_score = max(0.0, min(1.0, 1.0 - abs(ratio - aspect_hint) / aspect_hint))

    raw = [
        _comp(candidate.mask_quad_iou, weights.mask_iou, policy.neutral_mask_score),
        _comp(
            float(candidate.evidence["maskCoverage"])
            if "maskCoverage" in candidate.evidence
            else None,
            weights.mask_coverage,
            policy.neutral_mask_score,
        ),
        _comp(mask_confidence, weights.mask_confidence, policy.neutral_mask_score),
        _comp(candidate.edge_support, weights.edge_support, policy.neutral_edge_score),
        _comp(candidate.geometry_quality, weights.geometry, 0.0),
        _comp(convexity, weights.convexity, 0.0),
        _comp(border_val, weights.border, 1.0),
        _comp(aspect_score, weights.aspect, policy.neutral_aspect_score),
        _comp(source_val, weights.source_reliability, 0.5),
    ]
    return max(0.0, min(1.0, sum(raw)))


def _candidate(
    candidate_id: int,
    source: str,
    *,
    mask_iou: float | None,
    geometry: float,
    edge: float | None,
    area_ratio: float = 0.5,
) -> QuadrilateralCandidate:
    return QuadrilateralCandidate(
        candidate_id=candidate_id,
        corners=CanonicalCorners.from_sequence([(10, 10), (110, 10), (110, 90), (10, 90)]),
        source=source,  # type: ignore[arg-type]
        score=0.0,
        mask_quad_iou=mask_iou,
        edge_support=edge,
        geometry_quality=geometry,
        evidence={"quadAreaRatio": area_ratio},
    )


def test_score_breakdown_contains_all_components_and_neutral_cv_policy():
    candidate = _candidate(1, "contour", mask_iou=None, geometry=0.8, edge=None)

    score = score_candidate(candidate)

    assert set(score.breakdown) == {
        "mask_iou",
        "mask_coverage",
        "mask_confidence",
        "edge_support",
        "geometry",
        "convexity",
        "border",
        "aspect",
        "source_reliability",
        "occlusion_penalty",
        "occlusion_alignment",
    }
    assert score.breakdown["mask_iou"].applicable is False
    assert score.breakdown["mask_iou"].value == 0.5
    assert score.breakdown["edge_support"].policy == "neutral:edge_support"
    assert score.breakdown["occlusion_penalty"].contribution == 0.0
    assert score.breakdown["occlusion_penalty"].applicable is False
    assert score.breakdown["occlusion_penalty"].normalized_weight == 0.0
    assert score.breakdown["occlusion_alignment"].contribution == 0.0
    assert score.breakdown["occlusion_alignment"].applicable is False
    assert score.breakdown["occlusion_alignment"].normalized_weight == 0.0
    assert 0.0 <= score.final_score <= 1.0


def test_mask_and_edge_evidence_change_score_in_expected_direction():
    weak = _candidate(1, "contour", mask_iou=0.4, geometry=0.8, edge=0.3)
    strong = _candidate(2, "mask_fit", mask_iou=0.95, geometry=0.8, edge=0.9)

    weak_score = score_candidate(weak).final_score
    strong_score = score_candidate(strong).final_score

    assert strong_score > weak_score


def test_tiny_low_geometry_candidate_does_not_win_with_high_mask_iou():
    tiny = _candidate(1, "contour", mask_iou=0.99, geometry=0.05, edge=0.95, area_ratio=0.01)
    document = _candidate(2, "mask_fit", mask_iou=0.8, geometry=0.9, edge=0.7, area_ratio=0.6)
    config = CandidateScoringConfig(minimum_final_score=0.0, ambiguity_margin=0.0)

    result = rank_candidate_pool([tiny, document], config=config)

    assert result.status == "selected"
    assert result.selected_candidate_id == 2
    assert result.ranked_candidates[0].candidate_id == 2


def test_threshold_and_ambiguity_decisions_are_explicit():
    candidate = _candidate(1, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8)
    below = rank_candidate_pool([candidate], config=CandidateScoringConfig(minimum_final_score=1.0))
    ambiguous = rank_candidate_pool(
        [
            _candidate(1, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8),
            _candidate(2, "mixed", mask_iou=0.8, geometry=0.8, edge=0.8),
        ],
        config=CandidateScoringConfig(minimum_final_score=0.0, ambiguity_margin=1.0),
    )

    assert below.status == "below_threshold"
    assert below.selected_candidate_id is None
    assert ambiguous.status == "ambiguous"
    assert ambiguous.selected_candidate_id is None


def test_ranking_is_deterministic_and_weight_sweep_is_configurable():
    candidates = [
        _candidate(1, "mask_fit", mask_iou=0.9, geometry=0.7, edge=0.4),
        _candidate(2, "hough", mask_iou=None, geometry=0.8, edge=0.95),
    ]
    default = rank_candidate_pool(
        candidates, config=CandidateScoringConfig(minimum_final_score=0.0)
    )
    repeated = rank_candidate_pool(
        candidates, config=CandidateScoringConfig(minimum_final_score=0.0)
    )
    edge_heavy = CandidateScoringConfig(
        minimum_final_score=0.0,
        weights=CandidateScoreWeights(
            mask_iou=0.0,
            mask_coverage=0.0,
            mask_confidence=0.0,
            edge_support=1.0,
            geometry=0.0,
            convexity=0.0,
            border=0.0,
            aspect=0.0,
            source_reliability=0.0,
        ),
    )
    swept = sweep_candidate_scores(candidates, [edge_heavy])

    assert default.model_dump(mode="json") == repeated.model_dump(mode="json")
    assert swept[0].ranked_candidates[0].candidate_id == 2


def test_candidate_scoring_diagnostics_and_score_delta():
    c1 = _candidate(1, "mask_fit", mask_iou=0.9, geometry=0.85, edge=0.8)
    c2 = _candidate(2, "contour", mask_iou=0.8, geometry=0.8, edge=0.75)

    # 1. Default diagnostics: scoreDelta present, no verbose candidates list
    res_default = rank_candidate_pool([c1, c2])
    assert "scoreDelta" in res_default.diagnostics
    assert isinstance(res_default.diagnostics["scoreDelta"], float)
    assert res_default.diagnostics["scoreDelta"] > 0.0
    assert "candidates" not in res_default.diagnostics

    # 2. Debug diagnostics enabled: candidates list populated with contributions and corners
    debug_config = CandidateScoringConfig(debug_diagnostics=True)
    res_debug = rank_candidate_pool([c1, c2], config=debug_config)
    assert "candidates" in res_debug.diagnostics
    candidates_diag = res_debug.diagnostics["candidates"]
    assert len(candidates_diag) == 2
    top = candidates_diag[0]
    assert top["candidateId"] == res_debug.selected_candidate_id
    assert "contributions" in top
    assert "mask_iou" in top["contributions"]
    assert "corners" in top
    assert len(top["corners"]) == 4


def test_non_overlapping_document_score_identical_to_legacy_baseline():
    """Verify clean, non-overlapping documents retain 100% exact absolute score
    identical to legacy baseline."""
    candidates = [
        _candidate(1, "mask_fit", mask_iou=0.92, geometry=0.88, edge=0.85, area_ratio=0.7),
        _candidate(2, "contour", mask_iou=0.75, geometry=0.80, edge=0.70, area_ratio=0.6),
        _candidate(3, "hough", mask_iou=None, geometry=0.65, edge=0.90, area_ratio=0.5),
    ]
    config = CandidateScoringConfig()
    for cand in candidates:
        legacy_score = compute_legacy_base_score(cand, config=config)
        new_score = score_candidate(cand, config=config).final_score
        assert math.isclose(new_score, legacy_score, rel_tol=1e-9, abs_tol=1e-9)


def test_penalty_applied_only_once_and_preliminary_score_isolated():
    """Verify preliminary score does not leak into final score (no double penalty)."""
    corners = CanonicalCorners.from_sequence([(0, 0), (100, 0), (100, 100), (0, 100)])
    c_low_prelim = QuadrilateralCandidate(
        candidate_id=1,
        corners=corners,
        source="mask_fit",
        score=0.10,
        mask_quad_iou=0.85,
        edge_support=0.80,
        geometry_quality=0.90,
        evidence={
            "has_overlapping_cues": True,
            "enclosed_occlusion_length": 50.0,
            "aligns_with_occlusion_ridge": False,
        },
    )
    c_high_prelim = QuadrilateralCandidate(
        candidate_id=2,
        corners=corners,
        source="mask_fit",
        score=0.95,
        mask_quad_iou=0.85,
        edge_support=0.80,
        geometry_quality=0.90,
        evidence={
            "has_overlapping_cues": True,
            "enclosed_occlusion_length": 50.0,
            "aligns_with_occlusion_ridge": False,
        },
    )

    s1 = score_candidate(c_low_prelim)
    s2 = score_candidate(c_high_prelim)

    assert s1.final_score == s2.final_score
    base = compute_legacy_base_score(c_low_prelim)
    assert s1.final_score < base
    assert s1.breakdown["occlusion_penalty"].contribution < 0.0


def test_occlusion_penalty_not_applied_without_verified_cues():
    """Verify penalty is strictly guarded by has_overlapping_cues=True
    (no unverified defect penalty)."""
    corners = CanonicalCorners.from_sequence([(0, 0), (100, 0), (100, 100), (0, 100)])
    c_unverified = QuadrilateralCandidate(
        candidate_id=1,
        corners=corners,
        source="mask_fit",
        score=0.85,
        mask_quad_iou=0.85,
        edge_support=0.80,
        geometry_quality=0.90,
        evidence={
            "has_overlapping_cues": False,
            "enclosed_occlusion_length": 150.0,
            "aligns_with_occlusion_ridge": True,
        },
    )

    score = score_candidate(c_unverified)
    legacy = compute_legacy_base_score(c_unverified)

    assert score.final_score == legacy
    assert score.breakdown["occlusion_penalty"].contribution == 0.0
    assert score.breakdown["occlusion_alignment"].contribution == 0.0


def test_ambiguity_margin_defaults_to_three_percent():
    assert CandidateScoringConfig().ambiguity_margin == 0.03


def test_occlusion_tiebreaker_prefers_candidate_with_fewer_enclosed_ridges():
    safe = _candidate(1, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8)
    risky = _candidate(2, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8).model_copy(
        update={
            "evidence": {
                "has_overlapping_cues": True,
                "enclosed_occlusion_ridges": 1,
                "aligns_with_occlusion_ridge": False,
            }
        }
    )

    result = rank_candidate_pool(
        [safe, risky],
        config=CandidateScoringConfig(minimum_final_score=0.0),
    )

    assert result.status == "selected"
    assert result.selected_candidate_id == safe.candidate_id
    assert result.top_score is not None
    assert result.second_score is not None
    assert result.top_score - result.second_score <= 0.03


def test_occlusion_tiebreaker_prefers_aligned_candidate_when_ridge_count_matches():
    unaligned = _candidate(1, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8).model_copy(
        update={
            "evidence": {
                "has_overlapping_cues": True,
                "enclosed_occlusion_ridges": 0,
                "aligns_with_occlusion_ridge": False,
            }
        }
    )
    aligned = _candidate(2, "mask_fit", mask_iou=0.8, geometry=0.8, edge=0.8).model_copy(
        update={
            "evidence": {
                "has_overlapping_cues": True,
                "enclosed_occlusion_ridges": 0,
                "aligns_with_occlusion_ridge": True,
            }
        }
    )

    result = rank_candidate_pool(
        [unaligned, aligned],
        config=CandidateScoringConfig(minimum_final_score=0.0),
    )

    assert result.status == "selected"
    assert result.selected_candidate_id == aligned.candidate_id


def test_occlusion_tiebreaker_does_not_override_a_clear_score_winner():
    risky_high_score = _candidate(
        1, "mask_fit", mask_iou=0.99, geometry=0.95, edge=0.95
    ).model_copy(
        update={
            "evidence": {
                "has_overlapping_cues": True,
                "enclosed_occlusion_ridges": 1,
                "aligns_with_occlusion_ridge": False,
            }
        }
    )
    safe_low_score = _candidate(2, "mask_fit", mask_iou=0.55, geometry=0.55, edge=0.55)

    result = rank_candidate_pool(
        [risky_high_score, safe_low_score],
        config=CandidateScoringConfig(minimum_final_score=0.0),
    )

    assert result.status == "selected"
    assert result.selected_candidate_id == risky_high_score.candidate_id
    assert result.top_score is not None
    assert result.second_score is not None
    assert result.top_score - result.second_score > 0.03
