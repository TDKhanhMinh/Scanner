"""AS-46 hybrid decision-policy tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from attendance_scanner.candidate_scoring import CandidateScoringConfig
from attendance_scanner.cv_candidates import CvCandidate, CvCandidateConfig, CvCandidateSet
from attendance_scanner.detector import (
    CandidateCorners,
    DetectionTiming,
    DetectorEvidence,
    DocumentDetectionResult,
)
from attendance_scanner.hybrid import HybridConfig, HybridDocumentDetector, create_detector
from attendance_scanner.mask_postprocess import MaskPostprocessConfig, postprocess_document_mask
from attendance_scanner.pipeline.load import load_image
from attendance_scanner.segmentation import SegmentationOutput, SegmentationTransform


def _write_document(path: Path) -> Path:
    image = Image.new("RGB", (320, 240), color=(45, 45, 45))
    ImageDraw.Draw(image).rectangle(
        (20, 20, 300, 220), fill=(245, 245, 245), outline=(0, 0, 0), width=4
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def _empty_cv_pool() -> CvCandidateSet:
    return CvCandidateSet(
        source_width=320,
        source_height=240,
        detection_width=320,
        detection_height=240,
        scale_factor=1.0,
        preprocessing=CvCandidateConfig(),
    )


def _cv_pool_with_candidate() -> CvCandidateSet:
    return CvCandidateSet(
        source_width=320,
        source_height=240,
        detection_width=320,
        detection_height=240,
        scale_factor=1.0,
        candidates=[
            CvCandidate(
                candidate_id=0,
                corners=CandidateCorners(
                    points=[(30, 30), (290, 30), (290, 210), (30, 210)],
                    source="cv_contour",
                    confidence=0.8,
                ),
                contour_area_px=46800.0,
                area_ratio=0.6,
                perimeter_px=880.0,
                convex=True,
                confidence=0.8,
                source_contour_index=1,
                approximation_epsilon_ratio=0.02,
            ),
        ],
        preprocessing=CvCandidateConfig(),
    )


def _segmentation_output(*, ambiguous: bool = False) -> SegmentationOutput:
    mask = np.zeros((240, 320), dtype=np.bool_)
    mask[20:221, 20:301] = True
    probability = mask.astype(np.float32)
    postprocess = postprocess_document_mask(
        probability,
        MaskPostprocessConfig(close_kernel_size=0, morphology_iterations=0),
    )
    detection = DocumentDetectionResult(
        detector_version="fake-segmentation",
        model_version="fake-model",
        pipeline_version="segmentation-test",
        evidence=DetectorEvidence(
            mask_confidence=0.95,
            mask_area_ratio=float(mask.mean()),
            component_count=2 if ambiguous else 1,
            model_specific={"componentAmbiguous": ambiguous},
        ),
        timing=DetectionTiming(segmentation_inference_ms=2.0),
        failure_code="mask_only",
    )
    return SegmentationOutput(
        probability_mask=probability,
        binary_mask=mask,
        transform=SegmentationTransform(
            source_width=320,
            source_height=240,
            input_width=320,
            input_height=240,
            scale_x=1.0,
            scale_y=1.0,
            resized_width=320,
            resized_height=240,
        ),
        postprocess=postprocess,
        detection=detection,
    )


class SuccessfulSegmentation:
    def segment(self, _image):  # type: ignore[no-untyped-def]
        return _segmentation_output()


class AmbiguousSegmentation:
    def segment(self, _image):  # type: ignore[no-untyped-def]
        return _segmentation_output(ambiguous=True)


class BrokenSegmentation:
    def segment(self, _image):  # type: ignore[no-untyped-def]
        raise RuntimeError("model unavailable")


def test_segmentation_success_returns_canonical_result_and_trace(tmp_path: Path):
    image = load_image(_write_document(tmp_path / "sheet.png"))
    detector = HybridDocumentDetector(
        SuccessfulSegmentation(),
        config=HybridConfig(
            use_hough=False,
            scoring=CandidateScoringConfig(minimum_final_score=0.0, ambiguity_margin=0.0),
        ),
        cv_provider=lambda _image: _empty_cv_pool(),
    )

    result = detector.detect(image)

    assert result.detected is True
    assert result.corners is not None
    assert result.decision_trace is not None
    assert result.decision_trace.path == "segmentation_first"
    assert result.decision_trace.segmentation_state == "success"
    assert result.decision_trace.ranking_status == "selected"
    assert result.metadata["selectedSource"] == "mask_fit"


def test_segmentation_failure_rescues_with_cv_and_records_warning(tmp_path: Path):
    image = load_image(_write_document(tmp_path / "sheet.png"))
    detector = HybridDocumentDetector(
        BrokenSegmentation(),
        config=HybridConfig(
            use_hough=False,
            scoring=CandidateScoringConfig(minimum_final_score=0.0, ambiguity_margin=0.0),
        ),
        cv_provider=lambda _image: _cv_pool_with_candidate(),
    )

    result = detector.detect(image)

    assert result.detected is True
    assert result.decision_trace is not None
    assert result.decision_trace.path == "cv_fallback"
    assert "SEGMENTATION_FALLBACK" in result.warnings


def test_ambiguous_ranking_does_not_autocrop_and_full_failure_falls_back(tmp_path: Path):
    image = load_image(_write_document(tmp_path / "sheet.png"))
    ambiguous = HybridDocumentDetector(
        AmbiguousSegmentation(),
        config=HybridConfig(
            use_hough=False,
            scoring=CandidateScoringConfig(minimum_final_score=0.0, ambiguity_margin=1.0),
        ),
        cv_provider=lambda _image: _cv_pool_with_candidate(),
    )
    ambiguous_result = ambiguous.detect(image)

    assert ambiguous_result.detected is False
    assert ambiguous_result.corners is None
    assert ambiguous_result.failure_code == "ambiguous"
    assert ambiguous_result.decision_trace is not None
    assert ambiguous_result.decision_trace.path == "ambiguous"
    assert "DETECTION_AMBIGUOUS" in ambiguous_result.warnings

    failed = HybridDocumentDetector(
        BrokenSegmentation(),
        config=HybridConfig(use_hough=False),
        cv_provider=lambda _image: _empty_cv_pool(),
    )
    failed_result = failed.detect(image)
    assert failed_result.detected is False
    assert failed_result.fallback_used is True
    assert failed_result.decision_trace is not None
    assert failed_result.decision_trace.path == "full_image_fallback"
    assert "FALLBACK_FULL_IMAGE" in failed_result.warnings

    disabled = HybridDocumentDetector(
        BrokenSegmentation(),
        config=HybridConfig(use_hough=False, fallback_full_image=False),
        cv_provider=lambda _image: _empty_cv_pool(),
    )
    disabled_result = disabled.detect(image)
    assert disabled_result.detected is False
    assert disabled_result.fallback_used is False
    assert disabled_result.decision_trace is not None
    assert disabled_result.decision_trace.path == "no_candidate"
    assert "FALLBACK_FULL_IMAGE" not in disabled_result.warnings


def test_hybrid_factory_keeps_ab_modes_explicit():
    assert create_detector("v1_cv").__class__.__name__ == "V1CvDocumentDetector"
    assert create_detector("cv_v2").__class__.__name__ == "HybridDocumentDetector"
    with pytest.raises(ValueError, match="requires a segmentation"):
        create_detector("segmentation_only")
