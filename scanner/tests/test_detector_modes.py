"""AS-54 detector mode and provider selection tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from attendance_scanner.cli import create_parser
from attendance_scanner.contracts import ScanBatchRequest
from attendance_scanner.detector import (
    CanonicalCorners,
    DetectionTiming,
    DetectorDecisionTrace,
    DocumentDetectionResult,
)
from attendance_scanner.detector_modes import (
    DEFAULT_DETECTOR_MODE,
    detector_name_for_mode,
    internal_detector_mode,
    normalize_detector_mode,
)
from attendance_scanner.pipeline.orchestrator import scan_one


class StubDetector:
    detector_version = "configured-provider"

    def __init__(self) -> None:
        self.calls = 0

    def detect(self, image):  # type: ignore[no-untyped-def]
        self.calls += 1
        return DocumentDetectionResult(
            detected=True,
            corners=CanonicalCorners(
                tl=(5.0, 5.0), tr=(195.0, 5.0), br=(195.0, 95.0), bl=(5.0, 95.0)
            ),
            confidence=0.98,
            detector_version=self.detector_version,
            model_version="configured-v1",
            pipeline_version="configured-pipeline",
            timing=DetectionTiming(total_detection_ms=1.0),
            decision_trace=DetectorDecisionTrace(
                path="segmentation_first",
                segmentation_state="success",
                candidate_count=1,
                rejected_candidate_count=0,
                ranking_status="selected",
                selected_candidate_id=1,
            ),
        )


def test_product_mode_defaults_and_provider_neutral_mapping() -> None:
    assert DEFAULT_DETECTOR_MODE == "ai_enhanced"
    assert normalize_detector_mode(None) == "ai_enhanced"
    assert normalize_detector_mode("Classic", allow_development=False) == "classic"
    assert internal_detector_mode("ai_enhanced") == "hybrid"
    assert detector_name_for_mode("classic") == "v1_cv"
    with pytest.raises(ValueError, match="Unsupported detector mode"):
        normalize_detector_mode("not-a-mode")


def test_cli_exposes_product_and_development_modes() -> None:
    parser = create_parser()
    args = parser.parse_args(["plan", "--input", "input"])
    assert args.detector_mode == "ai_enhanced"
    debug_args = parser.parse_args(
        [
            "scan-batch",
            "--input",
            "input",
            "--detector-mode",
            "cv_v2",
            "--debug-diagnostics",
            "--reprocess",
        ]
    )
    assert debug_args.detector_mode == "cv_v2"
    assert debug_args.debug_diagnostics is True
    assert debug_args.reprocess is True


def test_scan_batch_request_keeps_enhancement_and_detector_modes_separate() -> None:
    request = ScanBatchRequest(input_root="input", mode="gray", detector_mode="classic")
    assert request.mode.value == "gray"
    assert request.detector_mode == "classic"
    with pytest.raises(ValueError, match="Unsupported detector mode"):
        ScanBatchRequest(input_root="input", detector_mode="unknown")


def test_scan_one_uses_configured_detector_without_model_name_in_product_mode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "page.png"
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    assert cv2.imwrite(str(source), image)
    detector = StubDetector()

    result = scan_one(
        source,
        mode="gray",
        detector=detector,
        detector_mode="ai_enhanced",
        debug_diagnostics=True,
    )

    assert detector.calls == 1
    assert result.document_detected is True
    assert result.detector_name == "configured-provider"
    assert result.detector_mode == "ai_enhanced"
    assert result.detector_model_version == "configured-v1"
    assert result.detection_quality_summary["decisionPath"] is not None
