"""AS-45 candidate score and ranking tests."""

from attendance_scanner.candidate_scoring import (
    CandidateScoreWeights,
    CandidateScoringConfig,
    rank_candidate_pool,
    score_candidate,
    sweep_candidate_scores,
)
from attendance_scanner.detector import CanonicalCorners
from attendance_scanner.quadrilateral_candidates import QuadrilateralCandidate


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
    }
    assert score.breakdown["mask_iou"].applicable is False
    assert score.breakdown["mask_iou"].value == 0.5
    assert score.breakdown["edge_support"].policy == "neutral:edge_support"
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
