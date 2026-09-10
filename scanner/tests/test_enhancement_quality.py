"""AS-56 readability comparison and visual artifact tests."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from attendance_scanner.contracts import ScanMode
from attendance_scanner.enhancement_quality import (
    compare_enhancement_modes,
    load_comparison_report,
)
from attendance_scanner.pipeline import orchestrator
from attendance_scanner.pipeline.enhance import enhance_image as real_enhance_image
from tests.test_detector_modes import StubDetector


def _timesheet_fixture() -> np.ndarray:
    image = np.full((600, 900, 3), (220, 222, 218), dtype=np.uint8)
    cv2.rectangle(image, (60, 60), (840, 540), (245, 245, 245), -1)
    for x in range(100, 841, 74):
        cv2.line(image, (x, 130), (x, 500), (70, 70, 70), 2)
    for y in range(130, 501, 46):
        cv2.line(image, (60, y), (840, y), (70, 70, 70), 2)
    cv2.putText(image, "ATTENDANCE", (105, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    cv2.putText(image, "blue", (120, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (150, 50, 30), 2)
    cv2.putText(image, "red", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (30, 40, 180), 2)
    return image


def test_comparison_generates_four_mode_visual_and_metric_artifacts(tmp_path: Path) -> None:
    report = compare_enhancement_modes(_timesheet_fixture(), output_dir=tmp_path)

    assert report.source_width == 900
    assert report.source_height == 600
    assert [metric.mode for metric in report.modes] == [
        ScanMode.GRAY,
        ScanMode.BW,
        ScanMode.COLOR,
        ScanMode.SMART_DOCUMENT,
    ]
    assert report.recommended_mode in set(ScanMode)
    assert "readability" in report.recommendation_reason
    assert "file size" in report.recommendation_reason
    assert set(report.artifacts) == {"gray", "bw", "color", "smart_document", "comparison"}
    assert all(Path(path).is_file() for path in report.artifacts.values())

    restored = load_comparison_report(tmp_path / "comparison.json")
    assert restored.recommended_mode == report.recommended_mode
    assert all(metric.width == 900 and metric.height == 600 for metric in restored.modes)


def test_comparison_can_limit_modes_and_keeps_color_metric_separate() -> None:
    report = compare_enhancement_modes(
        _timesheet_fixture(),
        modes=[ScanMode.GRAY, ScanMode.COLOR],
    )

    assert [metric.mode for metric in report.modes] == [ScanMode.GRAY, ScanMode.COLOR]
    assert report.modes[0].color_ink_retention is None
    assert report.modes[1].color_ink_retention is not None


def test_detector_receives_source_pixels_before_final_enhancement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    detector = StubDetector()
    original_detect = detector.detect

    def capture_detection(image):  # type: ignore[no-untyped-def]
        observed["detector_shape"] = image.image.shape
        observed["detector_channels"] = image.image.shape[2]
        return original_detect(image)

    def capture_enhancement(image, mode, config=None):  # type: ignore[no-untyped-def]
        observed["enhancement_shape"] = image.image.shape
        return real_enhance_image(image, mode=mode, config=config)

    detector.detect = capture_detection  # type: ignore[method-assign]
    monkeypatch.setattr(orchestrator, "enhance_image", capture_enhancement)
    orchestrator.scan_one(
        _timesheet_fixture(),
        mode=ScanMode.GRAY,
        detector=detector,
        detector_mode="ai_enhanced",
    )

    assert observed["detector_shape"] == (600, 900, 3)
    assert observed["detector_channels"] == 3
    assert observed["enhancement_shape"] == (134, 190, 3)
