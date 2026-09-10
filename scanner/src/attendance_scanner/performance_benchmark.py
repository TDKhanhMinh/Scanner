"""Performance and memory reports for the V2 pipeline and ONNX runtime."""

from __future__ import annotations

import gc
import hashlib
import importlib
import os
import platform
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from pydantic import Field

from attendance_scanner.contracts import BaseContract
from attendance_scanner.onnx_runtime import OnnxInferenceService, benchmark_service
from attendance_scanner.pipeline.orchestrator import scan_one

_resource: Any = None
try:
    _resource = importlib.import_module("resource")
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    pass

PERFORMANCE_REPORT_VERSION = "1.0"


class LatencySummary(BaseContract):
    """Distribution summary for one measured stage."""

    count: int = Field(ge=0)
    mean_ms: Optional[float] = Field(default=None, ge=0.0)
    p50_ms: Optional[float] = Field(default=None, ge=0.0)
    p95_ms: Optional[float] = Field(default=None, ge=0.0)
    p99_ms: Optional[float] = Field(default=None, ge=0.0)


class PipelinePerformanceReport(BaseContract):
    """Streaming scan performance report with explicit memory observations."""

    schema_version: str = PERFORMANCE_REPORT_VERSION
    detector_mode: str
    image_count: int = Field(ge=0)
    processed_count: int = Field(ge=0)
    latency_ms: LatencySummary
    stage_latency_ms: Dict[str, LatencySummary] = Field(default_factory=dict)
    peak_rss_bytes: Optional[int] = Field(default=None, ge=0)
    peak_tracemalloc_bytes: Optional[int] = Field(default=None, ge=0)
    max_inflight_images: int = Field(default=1, ge=1)
    model_size_bytes: Optional[int] = Field(default=None, ge=0)
    model_sha256: Optional[str] = None
    packaged_artifact_size_bytes: Optional[int] = Field(default=None, ge=0)
    packaged_artifact_sha256: Optional[str] = None
    runtime: Dict[str, str] = Field(default_factory=dict)
    baseline_comparison: Dict[str, Optional[float]] = Field(default_factory=dict)


class OnnxPerformanceReport(BaseContract):
    """ONNX cold-start and reused-session benchmark report."""

    schema_version: str = PERFORMANCE_REPORT_VERSION
    model_path: str
    model_size_bytes: int = Field(ge=0)
    model_sha256: str
    cold_start_ms: float = Field(ge=0.0)
    first_inference_ms: float = Field(ge=0.0)
    steady_state_ms: LatencySummary
    session_create_count: int = Field(ge=0)
    providers: List[str] = Field(default_factory=list)
    runtime: Dict[str, str] = Field(default_factory=dict)


def _latency(values: Sequence[float]) -> LatencySummary:
    if not values:
        return LatencySummary(count=0)
    array = np.asarray(values, dtype=np.float64)
    return LatencySummary(
        count=len(values),
        mean_ms=float(np.mean(array)),
        p50_ms=float(np.percentile(array, 50)),
        p95_ms=float(np.percentile(array, 95)),
        p99_ms=float(np.percentile(array, 99)),
    )


def _peak_rss_bytes() -> Optional[int]:
    """Return process peak RSS where the current platform exposes it."""
    if platform.system().lower() == "windows":
        try:
            import ctypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("page_fault_count", ctypes.c_ulong),
                    ("peak_working_set_size", ctypes.c_size_t),
                    ("working_set_size", ctypes.c_size_t),
                    ("quota_peak_paged_pool_usage", ctypes.c_size_t),
                    ("quota_paged_pool_usage", ctypes.c_size_t),
                    ("quota_peak_non_paged_pool_usage", ctypes.c_size_t),
                    ("quota_non_paged_pool_usage", ctypes.c_size_t),
                    ("pagefile_usage", ctypes.c_size_t),
                    ("peak_pagefile_usage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(ProcessMemoryCounters)
            process = ctypes.windll.kernel32.GetCurrentProcess()
            succeeded = ctypes.windll.psapi.GetProcessMemoryInfo(
                process,
                ctypes.byref(counters),
                counters.cb,
            )
            return int(counters.peak_working_set_size) if succeeded else None
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    if _resource is None:
        return None
    try:
        value = int(_resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    return value * 1024


def _sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_performance_reports(
    current: PipelinePerformanceReport,
    baseline: Optional[PipelinePerformanceReport],
) -> Dict[str, Optional[float]]:
    """Compare current metrics to a prior report without inventing missing values."""
    if baseline is None:
        return {}
    current_values = {
        "meanLatencyMs": current.latency_ms.mean_ms,
        "p95LatencyMs": current.latency_ms.p95_ms,
        "peakRssBytes": float(current.peak_rss_bytes)
        if current.peak_rss_bytes is not None
        else None,
        "peakTracemallocBytes": float(current.peak_tracemalloc_bytes)
        if current.peak_tracemalloc_bytes is not None
        else None,
    }
    baseline_values = {
        "meanLatencyMs": baseline.latency_ms.mean_ms,
        "p95LatencyMs": baseline.latency_ms.p95_ms,
        "peakRssBytes": float(baseline.peak_rss_bytes)
        if baseline.peak_rss_bytes is not None
        else None,
        "peakTracemallocBytes": float(baseline.peak_tracemalloc_bytes)
        if baseline.peak_tracemalloc_bytes is not None
        else None,
    }

    def difference(
        current_value: Optional[float], baseline_value: Optional[float]
    ) -> Optional[float]:
        if current_value is None or baseline_value is None:
            return None
        return current_value - baseline_value

    return {key: difference(current_values[key], baseline_values[key]) for key in current_values}


def measure_pipeline_performance(
    image_paths: Sequence[Union[str, Path]],
    *,
    detector_mode: str = "classic",
    baseline: Optional[PipelinePerformanceReport] = None,
    model_path: Optional[Union[str, Path]] = None,
    packaged_artifact: Optional[Union[str, Path]] = None,
) -> PipelinePerformanceReport:
    """Measure bounded, one-image-at-a-time scan execution over image paths."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        raise ValueError("At least one image path is required")
    durations: List[float] = []
    stage_values: Dict[str, List[float]] = {}
    processed_count = 0
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    try:
        for path in paths:
            started_at = time.perf_counter()
            result = scan_one(path, detector_mode=detector_mode)
            durations.append((time.perf_counter() - started_at) * 1000.0)
            processed_count += 1
            for stage, value in result.diagnostics.stage_durations_ms.items():
                stage_values.setdefault(stage, []).append(float(value))
            del result
            gc.collect()
        _, peak_tracemalloc = tracemalloc.get_traced_memory()
    finally:
        if not tracing_before:
            tracemalloc.stop()

    model = Path(model_path) if model_path is not None else None
    artifact = Path(packaged_artifact) if packaged_artifact is not None else None
    report = PipelinePerformanceReport(
        detector_mode=detector_mode,
        image_count=len(paths),
        processed_count=processed_count,
        latency_ms=_latency(durations),
        stage_latency_ms={stage: _latency(values) for stage, values in stage_values.items()},
        peak_rss_bytes=_peak_rss_bytes(),
        peak_tracemalloc_bytes=None if tracing_before else peak_tracemalloc,
        model_size_bytes=model.stat().st_size if model is not None and model.is_file() else None,
        model_sha256=_sha256(model) if model is not None and model.is_file() else None,
        packaged_artifact_size_bytes=(
            artifact.stat().st_size if artifact is not None and artifact.is_file() else None
        ),
        packaged_artifact_sha256=(
            _sha256(artifact) if artifact is not None and artifact.is_file() else None
        ),
        runtime={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpuCount": str(os.cpu_count() or 0),
        },
        baseline_comparison={},
    )
    return report.model_copy(
        update={"baseline_comparison": compare_performance_reports(report, baseline)}
    )


def measure_onnx_performance(
    service: OnnxInferenceService,
    input_data: np.ndarray,
    *,
    iterations: int = 20,
) -> OnnxPerformanceReport:
    """Measure ONNX cold start and reused inference on one service lifecycle."""
    started_at = time.perf_counter()
    info = service.load()
    cold_start_ms = (time.perf_counter() - started_at) * 1000.0
    benchmark = benchmark_service(service, input_data, iterations=iterations)
    return OnnxPerformanceReport(
        model_path=info.model_path,
        model_size_bytes=info.model_size_bytes,
        model_sha256=info.model_sha256,
        cold_start_ms=cold_start_ms,
        first_inference_ms=benchmark.first_inference_ms,
        steady_state_ms=_latency(
            benchmark.steady_state_durations_ms or [benchmark.repeated_inference_ms["meanMs"]]
        ),
        session_create_count=benchmark.session_create_count,
        providers=info.providers,
        runtime={"onnxruntime": info.runtime_version},
    )


def load_pipeline_report(path: Union[str, Path]) -> PipelinePerformanceReport:
    """Load a pipeline performance report for before/after comparison."""
    return PipelinePerformanceReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "LatencySummary",
    "OnnxPerformanceReport",
    "PERFORMANCE_REPORT_VERSION",
    "PipelinePerformanceReport",
    "compare_performance_reports",
    "load_pipeline_report",
    "measure_onnx_performance",
    "measure_pipeline_performance",
]
