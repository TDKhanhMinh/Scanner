"""AS-35 benchmark metric and report-runner tests."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from attendance_scanner.benchmark import BenchmarkManifest, BenchmarkSample, BenchmarkSource
from attendance_scanner.benchmark_runner import (
    BenchmarkThresholds,
    DetectionPrediction,
    PredictionBundle,
    corner_distance,
    evaluate_thresholds,
    load_benchmark_datasets,
    mask_iou,
    polygon_iou,
    render_csv,
    render_markdown,
    run_benchmark,
    write_report_files,
)


def _source() -> BenchmarkSource:
    return BenchmarkSource(
        name="Synthetic anonymized benchmark fixture",
        license="Test-only synthetic data",
        attribution="Attendance Scanner tests",
    )


def _create_manifest(tmp_path: Path, *, dataset: str = "timesheet-v1", count: int = 2) -> Path:
    root = tmp_path / dataset
    root.mkdir(parents=True)
    samples = []
    for index in range(count):
        image_path = root / f"sheet-{index + 1:03d}.png"
        Image.new("RGB", (100, 100), color=(220, 220, 220)).save(image_path)
        samples.append(
            BenchmarkSample(
                sample_id=f"{dataset}:sheet-{index + 1:03d}.png",
                image_path=image_path.name,
                dataset=dataset,
                split="acceptance",
                width=100,
                height=100,
                tl=(10.0, 10.0),
                tr=(90.0, 10.0),
                br=(90.0, 90.0),
                bl=(10.0, 90.0),
                source=_source(),
            )
        )
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        BenchmarkManifest(samples=samples).dump_canonical_json(), encoding="utf-8"
    )
    return manifest_path


def _load_cli_module():
    script_path = Path(__file__).parents[2] / "scripts" / "run-v2-benchmark.py"
    spec = importlib.util.spec_from_file_location("run_v2_benchmark", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_polygon_iou_corner_distance_and_mask_iou():
    first = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    second = [(1.0, 0.0), (3.0, 0.0), (3.0, 2.0), (1.0, 2.0)]
    assert polygon_iou(first, first) == 1.0
    assert polygon_iou(first, second) == 1.0 / 3.0

    error = corner_distance(
        [(1.0, 1.0), (3.0, 1.0), (3.0, 3.0), (1.0, 3.0)],
        first,
        width=4,
        height=3,
    )
    assert error is not None
    assert error["mean_px"] == 2**0.5
    assert error["mean_normalized"] == (2**0.5) / 5

    expected_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    predicted_mask = np.array([[255, 255], [0, 0]], dtype=np.uint8)
    assert mask_iou(predicted_mask, expected_mask) == 1.0 / 3.0


def test_reference_report_is_perfect_and_release_gate_passes(tmp_path: Path):
    manifest_path = _create_manifest(tmp_path)
    datasets = load_benchmark_datasets([manifest_path])

    report = run_benchmark(
        datasets,
        detector="reference",
        thresholds=BenchmarkThresholds.release_targets(),
        repo_root=tmp_path,
    )

    assert report.metrics["sample_count"] == 2
    assert report.metrics["detection_success_rate"] == 1.0
    assert report.metrics["mean_iou"] == 1.0
    assert report.metrics["p95_normalized_corner_error"] == 0.0
    assert report.threshold_gate["passed"] is True
    assert report.dataset_manifest_sha256


def test_external_predictions_report_missing_and_invalid_geometry(tmp_path: Path):
    manifest_path = _create_manifest(tmp_path)
    datasets = load_benchmark_datasets([manifest_path])
    sample_id = "timesheet-v1:sheet-001.png"
    bundle = PredictionBundle(
        detector="hybrid",
        model_version="hybrid-test",
        predictions=[
            DetectionPrediction(
                sample_id=sample_id,
                detected=True,
                corners=[[10, 10], [91, 10], [90, 90], [10, 90]],
                raw_corners=[[20, 20], [100, 20], [100, 100], [20, 100]],
                timings_ms={"total_detection_ms": 5.0},
            )
        ],
    )

    report = run_benchmark(datasets, detector="hybrid", prediction_bundle=bundle)

    assert report.metrics["detected_count"] == 1
    assert report.metrics["detection_success_rate"] == 0.5
    assert report.metrics["failure_taxonomy"] == {"missing_prediction": 1}
    assert report.metrics["timings_ms"]["total_detection_ms"]["count"] == 1
    assert report.samples[0]["polygon_iou"] is not None
    assert report.metrics["mean_raw_corner_error_px"] > report.metrics["mean_corner_error_px"]

    invalid_bundle = PredictionBundle(
        detector="hybrid",
        predictions=[
            DetectionPrediction(
                sample_id=sample_id,
                detected=True,
                corners=[[10, 10], [90, 90], [90, 10]],
            )
        ],
    )
    invalid_report = run_benchmark(
        datasets,
        detector="hybrid",
        prediction_bundle=invalid_bundle,
    )
    assert invalid_report.metrics["failure_taxonomy"] == {
        "invalid_geometry": 1,
        "missing_prediction": 1,
    }


@pytest.mark.parametrize("detector", ["v1_cv", "reference"])
def test_external_predictions_cannot_override_builtin_detector(tmp_path: Path, detector: str):
    manifest_path = _create_manifest(tmp_path, count=1)
    datasets = load_benchmark_datasets([manifest_path])
    bundle = PredictionBundle(
        detector=detector,
        predictions=[DetectionPrediction(sample_id="timesheet-v1:sheet-001.png")],
    )

    with pytest.raises(ValueError, match="cannot override built-in detector"):
        run_benchmark(datasets, detector=detector, prediction_bundle=bundle)  # type: ignore[arg-type]


def test_report_writers_and_combined_manifests(tmp_path: Path):
    first = _create_manifest(tmp_path, dataset="first", count=1)
    second = _create_manifest(tmp_path, dataset="second", count=1)
    datasets = load_benchmark_datasets([first, second])
    report = run_benchmark(datasets, detector="reference")

    json_path = tmp_path / "reports" / "report.json"
    markdown_path = tmp_path / "reports" / "report.md"
    csv_path = tmp_path / "reports" / "report.csv"
    write_report_files(
        report,
        json_path=json_path,
        markdown_path=markdown_path,
        csv_path=csv_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["sample_count"] == 2
    assert "# V2 Benchmark Report" in render_markdown(report)
    assert "sample_id" in render_csv(report).splitlines()[0]
    assert markdown_path.is_file()
    assert csv_path.is_file()


def test_provider_callback_can_score_mask_only_adapter_evidence(tmp_path: Path):
    manifest_path = _create_manifest(tmp_path, count=1)
    dataset_root = manifest_path.parent
    mask = Image.new("L", (100, 100), color=0)
    ImageDraw.Draw(mask).rectangle((10, 10, 89, 89), fill=255)
    mask.save(dataset_root / "mask.png")
    manifest = BenchmarkManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    manifest = BenchmarkManifest(
        samples=[manifest.samples[0].model_copy(update={"mask_path": "mask.png"})]
    )
    manifest_path.write_text(manifest.dump_canonical_json(), encoding="utf-8")
    datasets = load_benchmark_datasets([manifest_path])

    def provider(sample, _image_root):  # type: ignore[no-untyped-def]
        return DetectionPrediction(
            sample_id=sample.sample_id,
            detected=False,
            failure_reason="mask_only",
            mask_path="mask.png",
        )

    report = run_benchmark(
        datasets,
        detector="segmentation_only",
        prediction_provider=provider,
    )

    assert report.metrics["detection_success_rate"] == 0.0
    assert report.metrics["mask_iou_count"] == 1
    assert report.metrics["mean_mask_iou"] == 1.0
    assert report.samples[0]["failure_reason"] == "mask_only"


def test_cli_runs_reference_report_and_gate(tmp_path: Path):
    module = _load_cli_module()
    manifest_path = _create_manifest(tmp_path, count=1)
    output = tmp_path / "cli" / "report.json"

    result = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--detector",
            "reference",
            "--json-output",
            str(output),
            "--gate",
        ]
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8"))["threshold_gate"]["passed"] is True


def test_threshold_gate_marks_missing_accuracy_metric_failed():
    gate = evaluate_thresholds(
        {
            "detection_success_rate": 1.0,
            "mean_iou": None,
            "p95_normalized_corner_error": 0.0,
            "false_detection_rate": 0.0,
        },
        BenchmarkThresholds(min_mean_iou=0.95),
    )

    assert gate["passed"] is False
    assert gate["checks"]["mean_iou"]["passed"] is False
