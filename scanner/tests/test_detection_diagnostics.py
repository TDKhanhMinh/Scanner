"""AS-53 detection failure taxonomy tests."""

import pytest

from attendance_scanner.contracts import DetectionFailureReason, DetectionFailureSummary
from attendance_scanner.diagnostics import summarize_detection_failure
from attendance_scanner.events import FileCompletedEvent, deserialize_event, serialize_event
from attendance_scanner.state import ManifestArtifact, ManifestEntry


def test_detection_failure_mapping_is_typed_and_user_safe():
    summary = summarize_detection_failure(
        document_detected=False,
        warning_codes=["DOCUMENT_NOT_DETECTED"],
        fallback_used=True,
    )

    assert isinstance(summary, DetectionFailureSummary)
    assert summary.primary_reason == DetectionFailureReason.CV_NO_CANDIDATE
    assert summary.reason_codes == [
        DetectionFailureReason.CV_NO_CANDIDATE,
        DetectionFailureReason.FALLBACK_FULL_IMAGE,
    ]
    assert summary.user_message
    assert "CV_NO_CANDIDATE" not in summary.user_message


def test_detection_reason_event_fields_are_additive_and_v1_parser_round_trips():
    event = FileCompletedEvent(
        relative_path="NV01/sheet.png",
        employee_name="NV01",
        output_relative_path="NV01/sheet.pdf",
        document_detected=False,
        warning="DOCUMENT_NOT_DETECTED",
        detection_reason=DetectionFailureReason.CV_NO_CANDIDATE,
        detection_reason_codes=[
            DetectionFailureReason.CV_NO_CANDIDATE,
            DetectionFailureReason.FALLBACK_FULL_IMAGE,
        ],
    )

    restored = deserialize_event(serialize_event(event))

    assert isinstance(restored, FileCompletedEvent)
    assert restored.detection_reason == DetectionFailureReason.CV_NO_CANDIDATE
    assert restored.detection_reason_codes[-1] == DetectionFailureReason.FALLBACK_FULL_IMAGE


@pytest.mark.parametrize(
    ("wire_warning", "expected"),
    [
        ("SEGMENTATION_LOW_CONFIDENCE", DetectionFailureReason.SEGMENTATION_LOW_CONFIDENCE),
        ("MASK_INVALID", DetectionFailureReason.MASK_INVALID),
        ("MASK_AMBIGUOUS_COMPONENTS", DetectionFailureReason.MASK_AMBIGUOUS_COMPONENTS),
        ("QUAD_FIT_FAILED", DetectionFailureReason.QUAD_FIT_FAILED),
        ("CV_NO_CANDIDATE", DetectionFailureReason.CV_NO_CANDIDATE),
        ("HYBRID_AMBIGUOUS", DetectionFailureReason.HYBRID_AMBIGUOUS),
        ("REFINEMENT_REJECTED", DetectionFailureReason.REFINEMENT_REJECTED),
        ("PERSPECTIVE_INVALID", DetectionFailureReason.PERSPECTIVE_INVALID),
    ],
)
def test_all_failure_categories_map_to_typed_primary_reasons(wire_warning, expected):
    summary = summarize_detection_failure(
        document_detected=False,
        warning_codes=[wire_warning],
    )

    assert summary.primary_reason == expected


def test_ai_unavailable_warning_maps_to_actionable_segmentation_reason():
    summary = summarize_detection_failure(
        document_detected=False,
        warning_codes=["SEGMENTATION_LOW_CONFIDENCE", "FALLBACK_FULL_IMAGE"],
        fallback_used=True,
    )

    assert summary.primary_reason == DetectionFailureReason.SEGMENTATION_LOW_CONFIDENCE
    assert "AI" in (summary.user_message or "")


@pytest.mark.parametrize("model", [ManifestEntry, ManifestArtifact])
def test_persisted_quality_summary_rejects_raw_masks_and_nested_debug_data(model):
    payload = {"detection_quality_summary": {"raw_mask": [[0, 1], [1, 0]]}}
    if model is ManifestEntry:
        payload.update(
            {
                "relative_path": "NV01/sheet.png",
                "size": 1,
                "mtime_ns": 1,
                "status": "success",
                "processed_at": "2026-09-10T00:00:00+00:00",
            }
        )
    else:
        payload.update(
            {
                "output_relative_path": "NV01/sheet.pdf",
            }
        )

    with pytest.raises(ValueError):
        model.model_validate(payload)
