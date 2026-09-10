"""Explainable, configurable scoring and ranking for V2 quad candidates."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract
from attendance_scanner.quadrilateral_candidates import (
    QuadrilateralCandidate,
    QuadrilateralCandidateSet,
)

CANDIDATE_SCORING_VERSION = "1.0"
RankingStatus = Literal["selected", "ambiguous", "below_threshold", "no_candidate"]


class CandidateScoreWeights(BaseContract):
    """Tunable weighted score components; weights are normalized at evaluation time."""

    mask_iou: float = Field(default=0.3, ge=0.0, le=1.0)
    mask_coverage: float = Field(default=0.1, ge=0.0, le=1.0)
    mask_confidence: float = Field(default=0.1, ge=0.0, le=1.0)
    edge_support: float = Field(default=0.2, ge=0.0, le=1.0)
    geometry: float = Field(default=0.15, ge=0.0, le=1.0)
    convexity: float = Field(default=0.05, ge=0.0, le=1.0)
    border: float = Field(default=0.05, ge=0.0, le=1.0)
    aspect: float = Field(default=0.05, ge=0.0, le=1.0)
    source_reliability: float = Field(default=0.05, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_positive_weight(self) -> "CandidateScoreWeights":
        if sum(self.model_dump().values()) <= 0.0:
            raise ValueError("at least one score weight must be positive")
        return self


class CandidateScoringConfig(BaseContract):
    """Score policy and decision thresholds that can be benchmark-swept."""

    weights: CandidateScoreWeights = Field(default_factory=CandidateScoreWeights)
    neutral_mask_score: float = Field(default=0.5, ge=0.0, le=1.0)
    neutral_edge_score: float = Field(default=0.5, ge=0.0, le=1.0)
    neutral_aspect_score: float = Field(default=0.5, ge=0.0, le=1.0)
    border_touch_score: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_final_score: float = Field(default=0.55, ge=0.0, le=1.0)
    ambiguity_margin: float = Field(default=0.05, ge=0.0, le=1.0)
    source_reliability: Dict[str, float] = Field(
        default_factory=lambda: {
            "mask_fit": 0.95,
            "mixed": 0.9,
            "contour": 0.65,
            "hough": 0.6,
        }
    )

    @model_validator(mode="after")
    def validate_source_reliability(self) -> "CandidateScoringConfig":
        if any(value < 0.0 or value > 1.0 for value in self.source_reliability.values()):
            raise ValueError("source_reliability values must be in [0,1]")
        return self


class ScoreComponent(BaseContract):
    """One score component with applicability and contribution evidence."""

    value: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    normalized_weight: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=0.0, le=1.0)
    applicable: bool
    policy: str


class CandidateScore(BaseContract):
    """Explainable score breakdown for one valid candidate."""

    candidate_id: int = Field(ge=0)
    source: str = Field(min_length=1)
    final_score: float = Field(ge=0.0, le=1.0)
    breakdown: Dict[str, ScoreComponent]


class RankedCandidate(BaseContract):
    """Candidate paired with its deterministic rank and explainable score."""

    rank: int = Field(ge=1)
    candidate_id: int = Field(ge=0)
    source: str = Field(min_length=1)
    score: CandidateScore


class CandidateRankingResult(BaseContract):
    """Decision output including threshold/ambiguity state, not just a winner."""

    scorer_version: str = CANDIDATE_SCORING_VERSION
    status: RankingStatus
    selected_candidate_id: Optional[int] = None
    top_score: Optional[float] = None
    second_score: Optional[float] = None
    minimum_final_score: float
    ambiguity_margin: float
    ranked_candidates: List[RankedCandidate] = Field(default_factory=list)
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


def _evidence_number(candidate: QuadrilateralCandidate, key: str) -> Optional[float]:
    value = candidate.evidence.get(key)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return max(0.0, min(1.0, float(value)))
    return None


def _component(
    value: Optional[float],
    *,
    weight: float,
    total_weight: float,
    neutral: float,
    policy: str,
) -> ScoreComponent:
    applicable = value is not None
    effective_value = value if value is not None else neutral
    normalized_weight = weight / total_weight if total_weight else 0.0
    return ScoreComponent(
        value=effective_value,
        weight=weight,
        normalized_weight=normalized_weight,
        contribution=effective_value * normalized_weight,
        applicable=applicable,
        policy=policy if applicable else f"neutral:{policy}",
    )


def _aspect_score(
    candidate: QuadrilateralCandidate, aspect_hint: Optional[float]
) -> Optional[float]:
    if aspect_hint is None:
        return None
    points = candidate.corners.as_list()
    width = (math.dist(points[0], points[1]) + math.dist(points[2], points[3])) / 2.0
    height = (math.dist(points[1], points[2]) + math.dist(points[3], points[0])) / 2.0
    if height <= 1e-9:
        return 0.0
    ratio = width / height
    return max(0.0, min(1.0, 1.0 - abs(ratio - aspect_hint) / aspect_hint))


def score_candidate(
    candidate: QuadrilateralCandidate,
    *,
    config: Optional[CandidateScoringConfig] = None,
    mask_confidence: Optional[float] = None,
    aspect_hint: Optional[float] = None,
) -> CandidateScore:
    """Score one already geometry-gated candidate with an explicit breakdown."""
    policy = config or CandidateScoringConfig()
    weights = policy.weights
    total_weight = sum(weights.model_dump().values())
    border_value: Optional[float] = None
    border_policy = "border_not_reported"
    if "borderContact" in candidate.evidence:
        border_value = policy.border_touch_score if candidate.evidence["borderContact"] else 1.0
        border_policy = "border_touch_allowed"
    convexity = _evidence_number(candidate, "convex")
    if convexity is None:
        convexity = 1.0
    source_value = policy.source_reliability.get(candidate.source, 0.5)
    components = {
        "mask_iou": _component(
            candidate.mask_quad_iou,
            weight=weights.mask_iou,
            total_weight=total_weight,
            neutral=policy.neutral_mask_score,
            policy="mask_quad_iou",
        ),
        "mask_coverage": _component(
            _evidence_number(candidate, "maskCoverage"),
            weight=weights.mask_coverage,
            total_weight=total_weight,
            neutral=policy.neutral_mask_score,
            policy="mask_coverage",
        ),
        "mask_confidence": _component(
            mask_confidence,
            weight=weights.mask_confidence,
            total_weight=total_weight,
            neutral=policy.neutral_mask_score,
            policy="mask_confidence",
        ),
        "edge_support": _component(
            candidate.edge_support,
            weight=weights.edge_support,
            total_weight=total_weight,
            neutral=policy.neutral_edge_score,
            policy="edge_support",
        ),
        "geometry": _component(
            candidate.geometry_quality,
            weight=weights.geometry,
            total_weight=total_weight,
            neutral=0.0,
            policy="geometry_gate_quality",
        ),
        "convexity": _component(
            convexity,
            weight=weights.convexity,
            total_weight=total_weight,
            neutral=0.0,
            policy="convexity",
        ),
        "border": _component(
            border_value,
            weight=weights.border,
            total_weight=total_weight,
            neutral=1.0,
            policy=border_policy,
        ),
        "aspect": _component(
            _aspect_score(candidate, aspect_hint),
            weight=weights.aspect,
            total_weight=total_weight,
            neutral=policy.neutral_aspect_score,
            policy="soft_aspect_hint",
        ),
        "source_reliability": _component(
            source_value,
            weight=weights.source_reliability,
            total_weight=total_weight,
            neutral=0.5,
            policy="source_reliability",
        ),
    }
    final_score = max(
        0.0, min(1.0, sum(component.contribution for component in components.values()))
    )
    return CandidateScore(
        candidate_id=candidate.candidate_id,
        source=candidate.source,
        final_score=final_score,
        breakdown=components,
    )


def rank_candidate_pool(
    candidates: Sequence[QuadrilateralCandidate] | QuadrilateralCandidateSet,
    *,
    config: Optional[CandidateScoringConfig] = None,
    mask_confidence: Optional[float] = None,
    aspect_hint: Optional[float] = None,
) -> CandidateRankingResult:
    """Rank valid candidates deterministically and surface threshold ambiguity."""
    policy = config or CandidateScoringConfig()
    candidate_list = (
        candidates.candidates
        if isinstance(candidates, QuadrilateralCandidateSet)
        else list(candidates)
    )
    scored = [
        score_candidate(
            candidate,
            config=policy,
            mask_confidence=mask_confidence,
            aspect_hint=aspect_hint,
        )
        for candidate in candidate_list
    ]
    scored.sort(
        key=lambda score: (
            -score.final_score,
            -score.breakdown["geometry"].value,
            score.source,
            score.candidate_id,
        )
    )
    ranked = [
        RankedCandidate(
            rank=index,
            candidate_id=score.candidate_id,
            source=score.source,
            score=score,
        )
        for index, score in enumerate(scored, start=1)
    ]
    top_score = scored[0].final_score if scored else None
    second_score = scored[1].final_score if len(scored) > 1 else None
    if not scored:
        status: RankingStatus = "no_candidate"
        selected_id = None
    elif top_score is None or top_score < policy.minimum_final_score:
        status = "below_threshold"
        selected_id = None
    elif second_score is not None and top_score - second_score <= policy.ambiguity_margin:
        status = "ambiguous"
        selected_id = None
    else:
        status = "selected"
        selected_id = scored[0].candidate_id
    return CandidateRankingResult(
        status=status,
        selected_candidate_id=selected_id,
        top_score=top_score,
        second_score=second_score,
        minimum_final_score=policy.minimum_final_score,
        ambiguity_margin=policy.ambiguity_margin,
        ranked_candidates=ranked,
        diagnostics={
            "candidateCount": len(candidate_list),
            "weightSum": sum(policy.weights.model_dump().values()),
            "neutralMaskPolicy": policy.neutral_mask_score,
            "neutralEdgePolicy": policy.neutral_edge_score,
        },
    )


def sweep_candidate_scores(
    candidates: Sequence[QuadrilateralCandidate] | QuadrilateralCandidateSet,
    configs: Sequence[CandidateScoringConfig],
    *,
    mask_confidence: Optional[float] = None,
    aspect_hint: Optional[float] = None,
) -> List[CandidateRankingResult]:
    """Evaluate score configurations without changing detector/core code."""
    return [
        rank_candidate_pool(
            candidates,
            config=config,
            mask_confidence=mask_confidence,
            aspect_hint=aspect_hint,
        )
        for config in configs
    ]


__all__ = [
    "CANDIDATE_SCORING_VERSION",
    "CandidateRankingResult",
    "CandidateScore",
    "CandidateScoreWeights",
    "CandidateScoringConfig",
    "RankedCandidate",
    "ScoreComponent",
    "rank_candidate_pool",
    "score_candidate",
    "sweep_candidate_scores",
]
