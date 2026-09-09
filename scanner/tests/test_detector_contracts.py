"""AS-36 detector contract and V1 compatibility adapter tests."""

from pathlib import Path

import pytest
from PIL import Image

from attendance_scanner.detector import (
    CandidateCorners,
    CanonicalCorners,
    DetectionTiming,
    DocumentDetectionResult,
    DocumentDetector,
    GeometrySummary,
    adapt_v1_detector,
    canonical_points_for_perspective,
)
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.pipeline.perspective import warp_perspective


def _corners() -> list[tuple[float, float]]:
    return [(10.0, 10.0), (90.0, 12.0), (88.0, 90.0), (12.0, 88.0)]


def test_canonical_corners_validate_order_and_serialize_with_wire_aliases():
    corners = CanonicalCorners.from_sequence(_corners())
    result = DocumentDetectionResult(
        detected=True,
        corners=corners,
        confidence=0.91,
        confidence_source="mock-keypoint-score",
        detector_version="mock-keypoint",
        model_version="mock-v1",
        pipeline_version="scanner-v2",
        candidate_corners=[
            CandidateCorners(
                points=_corners(),
                source="keypoint",
                coordinate_space="original_pixels",
                confidence=0.91,
            )
        ],
        geometry=GeometrySummary(area_ratio=0.6, is_convex=True),
        evidence={"mask_confidence": 0.88, "component_count": 1},
        timing=DetectionTiming(total_detection_ms=3.5),
    )

    payload = result.model_dump(by_alias=True, exclude_none=True)
    restored = DocumentDetectionResult.model_validate(payload)

    assert payload["candidateCorners"][0]["coordinateSpace"] == "original_pixels"
    assert payload["fallbackUsed"] is False
    assert restored.canonical_points() == _corners()
    assert restored.diagnostics_extension()["documentDetection"]["detectorVersion"] == (
        "mock-keypoint"
    )


def test_canonical_corners_reject_reversed_or_ambiguous_labels():
    with pytest.raises(ValueError, match="TL -> TR -> BR -> BL"):
        CanonicalCorners.from_sequence(list(reversed(_corners())))

    with pytest.raises(ValueError, match="exactly four"):
        CanonicalCorners.from_sequence(_corners()[:3])


def test_structured_no_document_result_does_not_use_exception_for_expected_failure():
    result = DocumentDetectionResult(
        detector_version="mock-empty",
        pipeline_version="scanner-v2",
        failure_code="no_document",
        warnings=["NO_DOCUMENT"],
    )

    assert result.detected is False
    assert result.corners is None
    assert result.failure_code == "no_document"
    assert result.diagnostics_extension()["documentDetection"]["failureCode"] == "no_document"


def test_wire_extension_rejects_raw_mask_fields_and_only_accepts_scalar_metadata():
    with pytest.raises(ValueError, match="raw-mask"):
        DocumentDetectionResult(
            detector_version="mock",
            pipeline_version="scanner-v2",
            metadata={"raw_mask": "internal-debug-reference"},
        )

    result = DocumentDetectionResult(
        detector_version="mock",
        pipeline_version="scanner-v2",
        failure_code="no_document",
        metadata={"provider_status": "empty", "candidate_count": 0},
    )
    payload = result.diagnostics_extension()["documentDetection"]
    assert payload["metadata"] == {"provider_status": "empty", "candidate_count": 0}
    assert "raw_mask" not in payload


def test_document_detector_protocol_accepts_segmentation_keypoint_and_cv_providers(
    tmp_path: Path,
):
    class Provider:
        def __init__(self, source: str) -> None:
            self.source = source

        def detect(self, image):  # type: ignore[no-untyped-def]
            assert image.width == 100
            return DocumentDetectionResult(
                detected=True,
                corners=CanonicalCorners.from_sequence(_corners()),
                detector_version=self.source,
                pipeline_version="scanner-v2",
            )

    image = load_image(_write_image(tmp_path))
    for source in ("segmentation", "keypoint", "cv"):
        provider: DocumentDetector = Provider(source)
        result = provider.detect(image)
        points = canonical_points_for_perspective(result)
        assert points == _corners()
        warped = warp_perspective(image, points)
        assert warped.image.size > 0


def _write_image(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / "fixture.png"
    Image.new("RGB", (100, 100), color=(230, 230, 230)).save(image_path)
    return image_path


def test_v1_adapter_exposes_classical_detector_without_changing_perspective_api(tmp_path: Path):
    image_path = _write_image(tmp_path)
    detector = adapt_v1_detector()
    result = detector.detect(load_image(image_path))

    assert isinstance(result, DocumentDetectionResult)
    assert result.detector_version == "v1_cv"
    assert result.pipeline_version == "scanner-v1"
    assert result.timing.total_detection_ms is not None
    assert result.failure_code in {None, "no_document", "document_clipped", "invalid_geometry"}
