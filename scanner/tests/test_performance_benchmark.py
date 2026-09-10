"""AS-58 performance, memory and report-contract tests."""

import importlib.util
import json
from pathlib import Path
from shutil import copyfile

import numpy as np
import onnxruntime as ort
from PIL import Image

from attendance_scanner.onnx_runtime import OnnxInferenceService
from attendance_scanner.performance_benchmark import (
    load_pipeline_report,
    measure_onnx_performance,
    measure_pipeline_performance,
)


def _model() -> Path:
    return Path(ort.__file__).resolve().parent / "datasets" / "mul_1.onnx"


def _write_images(root: Path, count: int = 3) -> list[Path]:
    paths = []
    for index in range(count):
        path = root / f"NV01/card-{index + 1}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 120), color=(128 + index, 128, 128)).save(path)
        paths.append(path)
    return paths


def _load_script():
    script_path = Path(__file__).parents[2] / "scripts" / "benchmark-v2-performance.py"
    spec = importlib.util.spec_from_file_location("benchmark_v2_performance", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_report_streams_one_image_and_compares_baseline(tmp_path: Path) -> None:
    paths = _write_images(tmp_path / "input")
    baseline = measure_pipeline_performance(paths[:1], detector_mode="classic")
    artifact = tmp_path / "sidecar.exe"
    artifact.write_bytes(b"sidecar")
    report = measure_pipeline_performance(
        paths,
        detector_mode="classic",
        baseline=baseline,
        model_path=_model(),
        packaged_artifact=artifact,
    )

    assert report.image_count == 3
    assert report.processed_count == 3
    assert report.max_inflight_images == 1
    assert report.latency_ms.count == 3
    assert report.peak_tracemalloc_bytes is not None
    assert report.model_size_bytes == _model().stat().st_size
    assert report.model_sha256
    assert report.packaged_artifact_size_bytes == artifact.stat().st_size
    assert set(report.baseline_comparison) == {
        "meanLatencyMs",
        "p95LatencyMs",
        "peakRssBytes",
        "peakTracemallocBytes",
    }

    report_path = tmp_path / "report.json"
    report_path.write_text(report.model_dump_json(by_alias=True), encoding="utf-8")
    restored = load_pipeline_report(report_path)
    assert restored.max_inflight_images == 1


def test_onnx_report_measures_cold_start_and_reused_session(tmp_path: Path) -> None:
    model_path = tmp_path / "mul_1.onnx"
    copyfile(_model(), model_path)
    service = OnnxInferenceService(model_path)
    try:
        report = measure_onnx_performance(
            service,
            np.ones((3, 2), dtype=np.float32),
            iterations=3,
        )
    finally:
        service.close()

    assert report.model_size_bytes == model_path.stat().st_size
    assert report.cold_start_ms >= 0
    assert report.first_inference_ms >= 0
    assert report.steady_state_ms.count == 3
    assert report.session_create_count == 1
    assert report.providers == ["CPUExecutionProvider"]


def test_performance_cli_writes_report_without_network_or_model(tmp_path: Path) -> None:
    module = _load_script()
    input_root = tmp_path / "input"
    _write_images(input_root, count=1)
    report_path = tmp_path / "reports" / "performance.json"

    exit_code = module.main(
        [
            "--input-root",
            str(input_root),
            "--output",
            str(report_path),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["measurement"] == {
        "cpuOnlyBaseline": True,
        "imagesAreStreamed": True,
        "internetRequired": False,
        "maxInflightImages": 1,
    }
    assert payload["onnx"] is None
