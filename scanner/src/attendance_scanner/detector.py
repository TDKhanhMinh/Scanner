"""Provider-neutral V2 document detector contracts and the V1 CV adapter."""

from __future__ import annotations

import math
import time
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from pydantic import Field, model_validator

from attendance_scanner.contracts import BaseContract, ScannerWarningCode
from attendance_scanner.pipeline.detect import (
    DetectionConfig,
    DetectionRejectionReason,
    DetectionResult,
    detect_document_boundary,
)
from attendance_scanner.pipeline.load import LoadedImage

DetectorPoint = Tuple[float, float]
DetectorCoordinateSpace = Literal["original_pixels", "detection_pixels"]
DetectorFailureCode = Literal[
    "no_document",
    "mask_only",
    "document_clipped",
    "invalid_geometry",
    "provider_unavailable",
    "inference_failed",
]
WireScalar = Optional[Union[str, int, float, bool]]
WireDiagnostics = Dict[str, WireScalar]
_FORBIDDEN_WIRE_KEYS = {
    "rawmask",
    "binarymask",
    "probabilitymask",
    "probabilitymap",
    "maskarray",
}


def _validate_wire_keys(values: Mapping[str, WireScalar]) -> None:
    for key in values:
        normalized = "".join(character for character in key.lower() if character.isalnum())
        if normalized in _FORBIDDEN_WIRE_KEYS:
            raise ValueError(f"{key} is an internal raw-mask field and cannot be serialized")


def _scalar_wire_diagnostics(values: Mapping[str, Any]) -> WireDiagnostics:
    """Keep only JSON-safe scalar diagnostics at the provider/wire boundary."""
    scalar_values: WireDiagnostics = {}
    for key, value in values.items():
        if value is None or isinstance(value, (str, int, float, bool)):
            scalar_values[key] = value
    _validate_wire_keys(scalar_values)
    return scalar_values


def _cross(a: DetectorPoint, b: DetectorPoint, c: DetectorPoint) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _validate_canonical_points(points: Sequence[DetectorPoint]) -> None:
    if len(points) != 4:
        raise ValueError("canonical corners must contain exactly four points")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("canonical corners must contain finite coordinates")
    if len({(round(x, 6), round(y, 6)) for x, y in points}) != 4:
        raise ValueError("canonical corners must contain four unique points")
    crosses = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    if any(value <= 1e-6 for value in crosses):
        raise ValueError("canonical corners must be convex and ordered TL -> TR -> BR -> BL")
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    expected_indices = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if expected_indices != [0, 1, 2, 3]:
        raise ValueError("canonical corners must use semantic TL -> TR -> BR -> BL labels")


class CanonicalCorners(BaseContract):
    """Validated four-corner result in TL/TR/BR/BL order."""

    tl: DetectorPoint
    tr: DetectorPoint
    br: DetectorPoint
    bl: DetectorPoint

    @model_validator(mode="after")
    def validate_geometry(self) -> "CanonicalCorners":
        _validate_canonical_points(self.as_list())
        return self

    @classmethod
    def from_sequence(cls, points: Sequence[DetectorPoint]) -> "CanonicalCorners":
        if len(points) != 4:
            raise ValueError("canonical corners require exactly four points")
        return cls(tl=points[0], tr=points[1], br=points[2], bl=points[3])

    def as_list(self) -> List[DetectorPoint]:
        return [self.tl, self.tr, self.br, self.bl]


class CandidateCorners(BaseContract):
    """Model-agnostic candidate geometry from segmentation, keypoint, or CV providers."""

    points: List[DetectorPoint] = Field(min_length=1)
    source: str = Field(min_length=1)
    coordinate_space: DetectorCoordinateSpace = "original_pixels"
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    diagnostics: WireDiagnostics = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_raw_mask_keys(self) -> "CandidateCorners":
        _validate_wire_keys(self.diagnostics)
        return self


class GeometrySummary(BaseContract):
    """Optional geometry evidence that does not couple consumers to a provider."""

    polygon_area_px: Optional[float] = Field(default=None, ge=0.0)
    area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    aspect_ratio: Optional[float] = Field(default=None, gt=0.0)
    is_convex: Optional[bool] = None
    clipped: bool = False
    border_touch_edges: List[str] = Field(default_factory=list)
    diagnostics: WireDiagnostics = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_raw_mask_keys(self) -> "GeometrySummary":
        _validate_wire_keys(self.diagnostics)
        return self


class DetectorEvidence(BaseContract):
    """Small typed evidence summary; raw masks remain internal/debug artifacts."""

    mask_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mask_area_ratio: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    component_count: Optional[int] = Field(default=None, ge=0)
    mask_to_quad_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_specific: WireDiagnostics = Field(default_factory=dict)

    @model_validator(mode="after")
    def reject_raw_mask_keys(self) -> "DetectorEvidence":
        _validate_wire_keys(self.model_specific)
        return self


class DetectionTiming(BaseContract):
    """Stage timing breakdown in milliseconds."""

    segmentation_inference_ms: Optional[float] = Field(default=None, ge=0.0)
    mask_postprocess_ms: Optional[float] = Field(default=None, ge=0.0)
    quadrilateral_fitting_ms: Optional[float] = Field(default=None, ge=0.0)
    cv_refinement_ms: Optional[float] = Field(default=None, ge=0.0)
    total_detection_ms: Optional[float] = Field(default=None, ge=0.0)
    additional_ms: Dict[str, float] = Field(default_factory=dict)


class DocumentDetectionResult(BaseContract):
    """Canonical detector result shared by AI, CV, hybrid, and fallback providers."""

    detected: bool = False
    corners: Optional[CanonicalCorners] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence_source: Optional[str] = None
    detector_version: str = Field(min_length=1)
    model_version: Optional[str] = None
    pipeline_version: str = Field(min_length=1)
    coordinate_space: DetectorCoordinateSpace = "original_pixels"
    candidate_corners: List[CandidateCorners] = Field(default_factory=list)
    geometry: GeometrySummary = Field(default_factory=GeometrySummary)
    evidence: DetectorEvidence = Field(default_factory=DetectorEvidence)
    fallback_used: bool = False
    warnings: List[str] = Field(default_factory=list)
    failure_code: Optional[DetectorFailureCode] = None
    timing: DetectionTiming = Field(default_factory=DetectionTiming)
    metadata: WireDiagnostics = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_result_state(self) -> "DocumentDetectionResult":
        _validate_wire_keys(self.metadata)
        if self.detected and self.corners is None:
            raise ValueError("detected results require canonical corners")
        if not self.detected and self.corners is not None:
            raise ValueError("undetected results must not expose final corners")
        if not self.detected and self.failure_code is None:
            raise ValueError("undetected results require a structured failure_code")
        if self.detected and self.failure_code is not None:
            raise ValueError("detected results must not contain a failure_code")
        return self

    def diagnostics_extension(self) -> Dict[str, Any]:
        """Return a backward-compatible diagnostics extension for JSONL callers."""
        return {"documentDetection": self.model_dump(by_alias=True, exclude_none=True)}

    def canonical_points(self) -> Optional[List[DetectorPoint]]:
        """Return plain points for the existing perspective API."""
        return self.corners.as_list() if self.corners is not None else None


@runtime_checkable
class DocumentDetector(Protocol):
    """Minimal provider interface accepting normalized images and returning typed results."""

    def detect(self, image: LoadedImage) -> DocumentDetectionResult:
        """Detect one document without raising for an expected no-document outcome."""


def _geometry_from_v1(result: DetectionResult) -> GeometrySummary:
    diagnostics = result.diagnostics
    side_lengths = diagnostics.get("side_lengths")
    aspect_ratio = None
    if isinstance(side_lengths, list) and len(side_lengths) == 4:
        width = (float(side_lengths[0]) + float(side_lengths[2])) / 2.0
        height = (float(side_lengths[1]) + float(side_lengths[3])) / 2.0
        if height > 1e-9:
            aspect_ratio = width / height
    return GeometrySummary(
        area_ratio=result.area_ratio,
        aspect_ratio=aspect_ratio,
        is_convex=result.accepted,
        clipped=result.clipped,
        border_touch_edges=list(diagnostics.get("border_touch_edges", [])),
        diagnostics=_scalar_wire_diagnostics(diagnostics),
    )


class V1CvDocumentDetector:
    """Adapter that exposes the existing OpenCV detector through DocumentDetector."""

    detector_version = "v1_cv"
    model_version = "opencv-classical"
    pipeline_version = "scanner-v1"

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()

    def detect(self, image: LoadedImage) -> DocumentDetectionResult:
        started_at = time.perf_counter()
        result = detect_document_boundary(image, config=self.config)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        timing = DetectionTiming(total_detection_ms=elapsed_ms)
        if result is None:
            return DocumentDetectionResult(
                detector_version=self.detector_version,
                model_version=self.model_version,
                pipeline_version=self.pipeline_version,
                failure_code="no_document",
                timing=timing,
            )

        candidate = CandidateCorners(
            points=result.corners,
            source="v1_cv",
            confidence=result.confidence,
            diagnostics=_scalar_wire_diagnostics(result.diagnostics),
        )
        if result.accepted and not result.clipped:
            corners = CanonicalCorners.from_sequence(result.corners)
            return DocumentDetectionResult(
                detected=True,
                corners=corners,
                confidence=result.confidence,
                confidence_source="opencv_geometry_score",
                detector_version=self.detector_version,
                model_version=self.model_version,
                pipeline_version=self.pipeline_version,
                candidate_corners=[candidate],
                geometry=_geometry_from_v1(result),
                timing=timing,
            )

        failure_code: DetectorFailureCode = (
            "document_clipped"
            if result.clipped
            or result.rejection_reason == DetectionRejectionReason.DOCUMENT_CLIPPED
            else "invalid_geometry"
        )
        warnings = [
            ScannerWarningCode.DOCUMENT_CLIPPED.value
            if failure_code == "document_clipped"
            else "V1_DETECTION_REJECTED"
        ]
        return DocumentDetectionResult(
            detector_version=self.detector_version,
            model_version=self.model_version,
            pipeline_version=self.pipeline_version,
            candidate_corners=[candidate],
            geometry=_geometry_from_v1(result),
            warnings=warnings,
            failure_code=failure_code,
            timing=timing,
        )


def adapt_v1_detector(config: Optional[DetectionConfig] = None) -> DocumentDetector:
    """Build the V1 compatibility adapter used for benchmark A/B comparisons."""
    return V1CvDocumentDetector(config=config)


def canonical_points_for_perspective(
    result: DocumentDetectionResult,
) -> Optional[List[DetectorPoint]]:
    """Bridge a canonical result to the existing perspective/export input shape."""
    return result.canonical_points()


__all__ = [
    "CandidateCorners",
    "CanonicalCorners",
    "DetectionTiming",
    "DetectorEvidence",
    "DocumentDetectionResult",
    "DocumentDetector",
    "GeometrySummary",
    "V1CvDocumentDetector",
    "adapt_v1_detector",
    "canonical_points_for_perspective",
]
