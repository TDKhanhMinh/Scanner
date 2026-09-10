"""AS-59 full detector comparison and decision tests."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from attendance_scanner.benchmark import BenchmarkManifest, BenchmarkSample, BenchmarkSource
from attendance_scanner.benchmark_decision import (
    REQUIRED_DETECTORS,
    render_decision_markdown,
    run_full_v2_decision,
)
from attendance_scanner.benchmark_runner import (
    BenchmarkThresholds,
    DetectionPrediction,
    PredictionBundle,
    load_benchmark_datasets,
)


def _source() -> BenchmarkSource:
    return BenchmarkSource(
        name="Synthetic AS-59 fixture",
        license="Test-only synthetic data",
        attribution="Attendance Scanner tests",
    )


def _dataset(tmp_path: Path) -> list:
    root = tmp_path / "dataset"
    root.mkdir()
    image_path = root / "sheet.png"
    image = np.full((240, 320, 3), 32, dtype=np.uint8)
    cv2.rectangle(image, (30, 30), (290, 210), (245, 245, 245), -1)
    cv2.rectangle(image, (30, 30), (290, 210), (20, 20, 20), 3)
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(image_path)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        BenchmarkManifest(
            samples=[
                BenchmarkSample(
                    sample_id="timesheet:sheet.png",
                    image_path="sheet.png",
                    dataset="timesheet",
                    split="acceptance",
                    width=320,
                    height=240,
                    tl=(30.0, 30.0),
                    tr=(290.0, 30.0),
                    br=(290.0, 210.0),
                    bl=(30.0, 210.0),
                    scenario_tags=["grid-heavy", "low-contrast"],
                    source=_source(),
                )
            ]
        ).dump_canonical_json(),
        encoding="utf-8",
    )
    return load_benchmark_datasets([manifest_path])


def _write_bundle(
    tmp_path: Path,
    detector: str,
    *,
    correct: bool = True,
    checksum: str | None = "a" * 64,
) -> Path:
    corners = (
        [[30.0, 30.0], [290.0, 30.0], [290.0, 210.0], [30.0, 210.0]]
        if correct
        else [[100.0, 70.0], [220.0, 70.0], [220.0, 170.0], [100.0, 170.0]]
    )
    bundle = PredictionBundle(
        detector=detector,
        model_version=f"{detector}-test-v1",
        model_checksum=checksum,
        predictions=[
            DetectionPrediction(
                sample_id="timesheet:sheet.png",
                detected=True,
                corners=corners,
                polygon=corners,
            )
        ],
    )
    path = tmp_path / f"{detector}.json"
    path.write_text(bundle.model_dump_json(), encoding="utf-8")
    return path


def test_decision_report_blocks_without_full_v2_evidence(tmp_path: Path) -> None:
    report = run_full_v2_decision(_dataset(tmp_path), repo_root=tmp_path)

    assert report.decision_status == "BLOCKED"
    assert report.selected_production_detector is None
    assert set(report.comparisons) == set(REQUIRED_DETECTORS)
    assert report.comparisons["v1_cv"].available is True
    assert report.comparisons["hybrid"].available is False
    assert any("hybrid" in blocker for blocker in report.blockers)
    assert "Blockers" in render_decision_markdown(report)


def test_decision_report_requires_model_checksum_for_v2_adapters(tmp_path: Path) -> None:
    prediction_paths = {
        detector: _write_bundle(tmp_path, detector)
        for detector in ("segmentation_only", "cv_v2", "hybrid")
    }
    prediction_paths["hybrid"] = _write_bundle(tmp_path, "hybrid", checksum=None)

    report = run_full_v2_decision(
        _dataset(tmp_path),
        prediction_bundle_paths=prediction_paths,
        repo_root=tmp_path,
    )

    assert report.decision_status == "BLOCKED"
    assert report.selected_production_detector is None
    assert report.comparisons["hybrid"].available is False
    assert any("checksum" in blocker for blocker in report.blockers)
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        PredictionBundle(
            detector="hybrid",
            model_checksum="not-a-sha256",
            predictions=[DetectionPrediction(sample_id="x", detected=False, failure_reason="none")],
        )


def test_decision_report_never_promotes_silent_wrong_crop(tmp_path: Path) -> None:
    prediction_paths = {
        detector: _write_bundle(tmp_path, detector) for detector in ("segmentation_only", "cv_v2")
    }
    prediction_paths["hybrid"] = _write_bundle(tmp_path, "hybrid", correct=False)

    report = run_full_v2_decision(
        _dataset(tmp_path),
        prediction_bundle_paths=prediction_paths,
        repo_root=tmp_path,
        thresholds=BenchmarkThresholds(
            min_detection_success_rate=1.0,
            min_mean_iou=0.0,
            max_p95_normalized_corner_error=10.0,
            max_false_detection_rate=1.0,
        ),
    )

    assert report.decision_status == "BLOCKED"
    assert report.selected_production_detector is None
    assert report.comparisons["hybrid"].metrics["silent_wrong_crop_count"] == 1
    assert any("silent wrong crop" in blocker for blocker in report.blockers)


def test_decision_report_records_provenance_scenarios_and_hybrid_choice(tmp_path: Path) -> None:
    datasets = _dataset(tmp_path)
    prediction_paths = {
        detector: _write_bundle(tmp_path, detector)
        for detector in ("segmentation_only", "cv_v2", "hybrid")
    }
    report = run_full_v2_decision(
        datasets,
        prediction_bundle_paths=prediction_paths,
        repo_root=tmp_path,
        thresholds=BenchmarkThresholds(
            min_detection_success_rate=1.0,
            min_mean_iou=0.99,
            max_p95_normalized_corner_error=0.01,
            max_false_detection_rate=0.0,
        ),
    )

    assert report.decision_status == "READY"
    assert report.selected_production_detector == "hybrid"
    assert len(report.dataset_manifest_sha256) == 64
    assert report.comparisons["hybrid"].model_checksum == "a" * 64
    assert report.comparisons["hybrid"].prediction_bundle_sha256
    assert set(report.comparisons["hybrid"].scenario_metrics) == {
        "grid-heavy",
        "low-contrast",
    }
    assert report.comparisons["hybrid"].metrics["silent_wrong_crop_count"] == 0
    assert json.loads(report.dump_canonical_json())["decision_status"] == "READY"
