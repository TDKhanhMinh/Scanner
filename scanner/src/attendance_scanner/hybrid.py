"""Segmentation-first hybrid detector decision policy."""

from __future__ import annotations

import time
from typing import Callable, List, Literal, Optional, Protocol

import numpy as np
from pydantic import Field

from attendance_scanner.candidate_scoring import (
    CandidateRankingResult,
    CandidateScoringConfig,
    rank_candidate_pool,
)
from attendance_scanner.contracts import BaseContract
from attendance_scanner.cv_candidates import (
    CvCandidateConfig,
    CvCandidateSet,
    generate_cv_candidates,
)
from attendance_scanner.detector import (
    CandidateCorners,
    DetectionTiming,
    DetectorDecisionTrace,
    DetectorEvidence,
    DocumentDetectionResult,
    DocumentDetector,
    GeometrySummary,
)
from attendance_scanner.edge_support import (
    EdgeSupportConfig,
    build_edge_map,
    score_quad_edge_support,
)
from attendance_scanner.geometry_validator import GeometryValidationConfig, validate_quadrilateral
from attendance_scanner.hough_lines import HoughLineConfig, HoughLineEvidence, detect_hough_lines
from attendance_scanner.pipeline.load import LoadedImage
from attendance_scanner.quadrilateral_candidates import (
    QuadrilateralCandidate,
    QuadrilateralCandidateConfig,
    QuadrilateralCandidateSet,
    build_quadrilateral_candidates,
)
from attendance_scanner.segmentation import SegmentationOutput

HybridDecisionProvider = Callable[[LoadedImage], CvCandidateSet]
HybridLineProvider = Callable[[LoadedImage], HoughLineEvidence]


class HybridConfig(BaseContract):
    """Hybrid decision policy and bounded source configuration."""

    segmentation_enabled: bool = True
    use_hough: bool = True
    fallback_full_image: bool = True
    geometry: GeometryValidationConfig = Field(default_factory=GeometryValidationConfig)
    candidates: QuadrilateralCandidateConfig = Field(default_factory=QuadrilateralCandidateConfig)
    scoring: CandidateScoringConfig = Field(default_factory=CandidateScoringConfig)
    cv: CvCandidateConfig = Field(default_factory=CvCandidateConfig)
    hough: HoughLineConfig = Field(default_factory=HoughLineConfig)
    edge: EdgeSupportConfig = Field(default_factory=EdgeSupportConfig)


class SegmentationProvider(Protocol):
    """Minimal mask provider needed by the hybrid policy."""

    def segment(self, image: LoadedImage) -> SegmentationOutput:
        """Return internal segmentation output and generic evidence."""


class HybridDocumentDetector:
    """Segmentation-first detector with CV/Hough rescue and explicit ambiguity."""

    detector_version = "hybrid"
    pipeline_version = "hybrid-v1"

    def __init__(
        self,
        segmentation_provider: Optional[SegmentationProvider] = None,
        *,
        config: Optional[HybridConfig] = None,
        cv_provider: Optional[HybridDecisionProvider] = None,
        hough_provider: Optional[HybridLineProvider] = None,
    ) -> None:
        self.segmentation_provider = segmentation_provider
        self.config = config or HybridConfig()
        self.cv_provider = cv_provider
        self.hough_provider = hough_provider

    def detect(self, image: LoadedImage) -> DocumentDetectionResult:
        started_at = time.perf_counter()
        segmentation_state: Literal["not_configured", "success", "failed", "ambiguous"] = (
            "not_configured"
        )
        segmentation_output: Optional[SegmentationOutput] = None
        mask: Optional[np.ndarray] = None
        reason_codes: List[str] = []
        if self.config.segmentation_enabled and self.segmentation_provider is not None:
            try:
                segmentation_output = self.segmentation_provider.segment(image)
                mask = segmentation_output.binary_mask
                segmentation_state = "success"
                if segmentation_output.detection.evidence.model_specific.get(
                    "componentAmbiguous", False
                ):
                    segmentation_state = "ambiguous"
                    reason_codes.append("ambiguous_segmentation_components")
            except Exception as exc:  # noqa: BLE001 - provider failure becomes structured fallback
                segmentation_state = "failed"
                reason_codes.append(f"segmentation_{type(exc).__name__.lower()}")
        elif self.config.segmentation_enabled:
            reason_codes.append("segmentation_provider_not_configured")

        try:
            cv_candidates = (
                self.cv_provider(image)
                if self.cv_provider is not None
                else generate_cv_candidates(image, config=self.config.cv)
            )
            hough_evidence = None
            if self.config.use_hough:
                hough_evidence = (
                    self.hough_provider(image)
                    if self.hough_provider is not None
                    else detect_hough_lines(image, config=self.config.hough)
                )
            pool = build_quadrilateral_candidates(
                image_size=(image.width, image.height),
                mask=mask,
                cv_candidates=cv_candidates,
                hough_evidence=hough_evidence,
                config=self.config.candidates.model_copy(update={"geometry": self.config.geometry}),
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            reason_codes.append(f"candidate_generation_{type(exc).__name__.lower()}")
            pool = QuadrilateralCandidateSet(
                image_width=image.width,
                image_height=image.height,
                config=self.config.candidates.model_copy(update={"geometry": self.config.geometry}),
            )

        scored_candidates = self._add_edge_evidence(image, pool)
        mask_confidence = (
            segmentation_output.detection.evidence.mask_confidence
            if segmentation_output is not None
            else None
        )
        ranking = rank_candidate_pool(
            scored_candidates.candidates,
            config=self.config.scoring,
            mask_confidence=mask_confidence,
        )
        selected = self._selected_candidate(scored_candidates, ranking)
        decision_path = self._decision_path(segmentation_state, ranking, selected)
        fallback_used = (
            selected is None and self.config.fallback_full_image and ranking.status != "ambiguous"
        )
        if segmentation_state == "failed":
            reason_codes.append("segmentation_fallback")
        if ranking.status == "ambiguous":
            reason_codes.append("candidate_ambiguity")
        if fallback_used:
            reason_codes.append("FALLBACK_FULL_IMAGE")
            decision_path = "full_image_fallback"
        trace = DetectorDecisionTrace(
            path=decision_path,
            segmentation_state=segmentation_state,
            candidate_count=len(scored_candidates.candidates),
            rejected_candidate_count=len(scored_candidates.rejected_candidates),
            source_counts=scored_candidates.source_counts,
            ranking_status=ranking.status,
            selected_candidate_id=ranking.selected_candidate_id,
            fallback_used=fallback_used,
            reason_codes=reason_codes,
        )
        total_ms = (time.perf_counter() - started_at) * 1000.0
        if selected is not None and ranking.status == "selected":
            validation = validate_quadrilateral(
                selected.corners.as_list(),
                image_size=(image.width, image.height),
                config=self.config.geometry,
            )
            return DocumentDetectionResult(
                detected=True,
                corners=selected.corners,
                confidence=ranking.top_score,
                confidence_source="hybrid_candidate_score",
                detector_version=self.detector_version,
                model_version=(
                    segmentation_output.detection.model_version
                    if segmentation_output is not None
                    else None
                ),
                pipeline_version=self.pipeline_version,
                candidate_corners=[
                    CandidateCorners(
                        points=candidate.corners.as_list(),
                        source=candidate.source,
                        confidence=candidate.score,
                        diagnostics={"candidateScore": candidate.score},
                    )
                    for candidate in scored_candidates.candidates
                ],
                geometry=GeometrySummary(
                    area_ratio=validation.area_ratio,
                    is_convex=validation.convex,
                    clipped=False,
                    aspect_ratio=validation.aspect_ratio,
                ),
                evidence=self._evidence(segmentation_output, selected),
                fallback_used=False,
                warnings=self._warnings(segmentation_state, ranking),
                timing=DetectionTiming(
                    segmentation_inference_ms=(
                        segmentation_output.detection.timing.segmentation_inference_ms
                        if segmentation_output is not None
                        else None
                    ),
                    total_detection_ms=total_ms,
                ),
                metadata={
                    "rankingStatus": ranking.status,
                    "selectedSource": selected.source,
                    "segmentationState": segmentation_state,
                },
                decision_trace=trace,
            )

        failure_code: Literal["ambiguous", "no_document"] = (
            "ambiguous" if ranking.status == "ambiguous" else "no_document"
        )
        return DocumentDetectionResult(
            detected=False,
            detector_version=self.detector_version,
            model_version=(
                segmentation_output.detection.model_version
                if segmentation_output is not None
                else None
            ),
            pipeline_version=self.pipeline_version,
            evidence=self._evidence(segmentation_output, selected),
            fallback_used=fallback_used,
            warnings=self._warnings(segmentation_state, ranking) + reason_codes,
            failure_code=failure_code,
            timing=DetectionTiming(total_detection_ms=total_ms),
            metadata={
                "rankingStatus": ranking.status,
                "segmentationState": segmentation_state,
            },
            decision_trace=trace,
        )

    def _add_edge_evidence(
        self,
        image: LoadedImage,
        pool: QuadrilateralCandidateSet,
    ) -> QuadrilateralCandidateSet:
        if not pool.candidates:
            return pool
        edge_map = build_edge_map(image, config=self.config.edge)
        candidates: List[QuadrilateralCandidate] = []
        for candidate in pool.candidates:
            support = score_quad_edge_support(edge_map, candidate.corners)
            candidates.append(
                candidate.model_copy(
                    update={
                        "edge_support": support.overall_score,
                        "evidence": {
                            **candidate.evidence,
                            "borderContact": bool(
                                any(
                                    point[0] <= self.config.edge.minimum_band_px
                                    or point[1] <= self.config.edge.minimum_band_px
                                    for point in candidate.corners.as_list()
                                )
                            ),
                        },
                    }
                )
            )
        return pool.model_copy(update={"candidates": candidates})

    @staticmethod
    def _selected_candidate(
        pool: QuadrilateralCandidateSet,
        ranking: CandidateRankingResult,
    ) -> Optional[QuadrilateralCandidate]:
        if ranking.selected_candidate_id is None:
            return None
        return next(
            (
                candidate
                for candidate in pool.candidates
                if candidate.candidate_id == ranking.selected_candidate_id
            ),
            None,
        )

    @staticmethod
    def _decision_path(
        segmentation_state: str,
        ranking: CandidateRankingResult,
        selected: Optional[QuadrilateralCandidate],
    ) -> Literal[
        "segmentation_first",
        "cv_fallback",
        "agreement",
        "disagreement",
        "ambiguous",
        "below_threshold",
        "no_candidate",
        "full_image_fallback",
    ]:
        if ranking.status == "ambiguous":
            return "ambiguous"
        if selected is None:
            return "below_threshold" if ranking.status == "below_threshold" else "no_candidate"
        if segmentation_state in {"failed", "not_configured"}:
            return "cv_fallback"
        if selected.source == "mixed":
            return "agreement"
        if selected.source == "mask_fit":
            return "segmentation_first"
        return "disagreement"

    @staticmethod
    def _warnings(segmentation_state: str, ranking: CandidateRankingResult) -> List[str]:
        warnings: List[str] = []
        if segmentation_state in {"failed", "ambiguous", "not_configured"}:
            warnings.append("SEGMENTATION_FALLBACK")
        if segmentation_state in {"failed", "not_configured"}:
            warnings.append("SEGMENTATION_LOW_CONFIDENCE")
        if ranking.status == "ambiguous":
            warnings.append("DETECTION_AMBIGUOUS")
        return warnings

    @staticmethod
    def _evidence(
        segmentation_output: Optional[SegmentationOutput],
        selected: Optional[QuadrilateralCandidate],
    ) -> DetectorEvidence:
        if segmentation_output is None:
            return DetectorEvidence()
        source_evidence = segmentation_output.detection.evidence
        return DetectorEvidence(
            mask_confidence=source_evidence.mask_confidence,
            mask_area_ratio=source_evidence.mask_area_ratio,
            component_count=source_evidence.component_count,
            mask_to_quad_iou=selected.mask_quad_iou if selected is not None else None,
            model_specific={
                "componentAmbiguous": source_evidence.model_specific.get(
                    "componentAmbiguous", False
                )
            },
        )


def create_detector(
    mode: Literal["v1_cv", "segmentation_only", "cv_v2", "hybrid", "docaligner_reference"],
    *,
    segmentation_provider: Optional[SegmentationProvider] = None,
    reference_provider: Optional[DocumentDetector] = None,
    config: Optional[HybridConfig] = None,
) -> DocumentDetector:
    """Create a configured detector for A/B benchmark selection."""
    if mode == "v1_cv":
        from attendance_scanner.detector import adapt_v1_detector

        return adapt_v1_detector()
    if mode == "segmentation_only":
        if segmentation_provider is None or not hasattr(segmentation_provider, "detect"):
            raise ValueError("segmentation_only mode requires a segmentation DocumentDetector")
        return segmentation_provider  # type: ignore[return-value]
    if mode == "docaligner_reference":
        if reference_provider is None:
            raise ValueError("docaligner_reference mode requires an explicit provider")
        return reference_provider
    if mode == "cv_v2":
        return HybridDocumentDetector(
            None,
            config=(config or HybridConfig()).model_copy(update={"segmentation_enabled": False}),
        )
    return HybridDocumentDetector(segmentation_provider, config=config)


__all__ = [
    "HybridConfig",
    "HybridDocumentDetector",
    "SegmentationProvider",
    "create_detector",
]
