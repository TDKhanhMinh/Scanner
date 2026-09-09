"""Deterministic V2 benchmark metrics and report generation.

The runner is deliberately independent from the desktop UI. It can execute the
existing classical detector, use an oracle reference adapter, or score prediction
JSON emitted by a future segmentation/CV/hybrid detector through the same schema.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attendance_scanner.annotation import atomic_write_text
from attendance_scanner.benchmark import (
    BenchmarkManifest,
    BenchmarkPoint,
    BenchmarkSample,
    load_benchmark_manifest,
)
from attendance_scanner.pipeline.detect import detect_document_boundary
from attendance_scanner.pipeline.load import load_image

BENCHMARK_RUNNER_VERSION: Literal["1.0"] = "1.0"
DetectorName = Literal["v1_cv", "segmentation_only", "cv_v2", "hybrid", "reference"]
TIMING_STAGES = (
    "segmentation_inference_ms",
    "mask_postprocess_ms",
    "quadrilateral_fitting_ms",
    "cv_refinement_ms",
    "total_detection_ms",
)
PointList = List[BenchmarkPoint]


class DetectionPrediction(BaseModel):
    """Common prediction payload for built-in and external detector adapters."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    detected: bool = True
    corners: Any = None
    polygon: Any = None
    mask_path: Optional[str] = None
    timings_ms: Dict[str, float] = Field(default_factory=dict)
    memory_peak_bytes: Optional[int] = Field(default=None, ge=0)
    failure_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_measurements(self) -> "DetectionPrediction":
        if any(not math.isfinite(value) or value < 0.0 for value in self.timings_ms.values()):
            raise ValueError("timings_ms values must be finite and non-negative")
        return self


class PredictionBundle(BaseModel):
    """Versioned JSON input produced by a segmentation/CV/hybrid adapter."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    detector: Optional[str] = None
    model_version: str = "external"
    pipeline_version: Optional[str] = None
    predictions: List[DetectionPrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prediction_identity(self) -> "PredictionBundle":
        sample_ids = [prediction.sample_id for prediction in self.predictions]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("prediction sample_id values must be unique")
        return self


class BenchmarkThresholds(BaseModel):
    """Optional quality gate thresholds for one benchmark report."""

    model_config = ConfigDict(extra="forbid")

    min_detection_success_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_mean_iou: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_p95_normalized_corner_error: Optional[float] = Field(default=None, ge=0.0)
    max_false_detection_rate: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @classmethod
    def release_targets(cls) -> "BenchmarkThresholds":
        """Return the release targets tracked by AS-35."""
        return cls(
            min_detection_success_rate=0.98,
            min_mean_iou=0.95,
            max_p95_normalized_corner_error=0.01,
            max_false_detection_rate=0.01,
        )


class BenchmarkReport(BaseModel):
    """Machine-readable benchmark report with per-sample evidence."""

    model_config = ConfigDict(extra="forbid")

    report_version: Literal["1.0"] = BENCHMARK_RUNNER_VERSION
    detector: DetectorName
    model_version: str
    pipeline_version: str
    git_commit: str
    dataset_manifest_sha256: str
    runtime_environment: Dict[str, Any]
    metrics: Dict[str, Any]
    threshold_gate: Dict[str, Any]
    samples: List[Dict[str, Any]]

    def dump_canonical_json(self) -> str:
        """Serialize the report without timestamps or nondeterministic ordering."""
        return (
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True)
class BenchmarkDataset:
    """One manifest and its corresponding local image root."""

    manifest_path: Path
    image_root: Path
    manifest: BenchmarkManifest


@dataclass(frozen=True)
class BenchmarkEntry:
    """A sample paired with the root needed to resolve its references."""

    sample: BenchmarkSample
    image_root: Path


def load_benchmark_datasets(
    manifest_paths: Sequence[Union[str, Path]],
    image_roots: Optional[Sequence[Union[str, Path]]] = None,
) -> List[BenchmarkDataset]:
    """Load one or more manifests and validate all referenced images/masks."""
    paths = [Path(path) for path in manifest_paths]
    if not paths:
        raise ValueError("At least one benchmark manifest is required")
    roots = [Path(root) for root in image_roots or ()]
    if roots and len(roots) != len(paths):
        raise ValueError("Provide exactly one --image-root for each --manifest")

    datasets: List[BenchmarkDataset] = []
    for index, manifest_path in enumerate(paths):
        image_root = (roots[index] if roots else manifest_path.parent).resolve()
        manifest = load_benchmark_manifest(manifest_path, image_root=image_root)
        datasets.append(
            BenchmarkDataset(
                manifest_path=manifest_path.resolve(),
                image_root=image_root,
                manifest=manifest,
            )
        )

    entries = _flatten_entries(datasets)
    sample_ids = [entry.sample.sample_id for entry in entries]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Combined benchmark manifests contain duplicate sample_id values")
    return datasets


def load_prediction_bundle(path: Union[str, Path]) -> PredictionBundle:
    """Load a prediction JSON object or a bare prediction list."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"predictions": payload}
    if not isinstance(payload, dict):
        raise ValueError("Prediction file must contain an object or prediction array")
    return PredictionBundle.model_validate(payload)


def _flatten_entries(datasets: Sequence[BenchmarkDataset]) -> List[BenchmarkEntry]:
    entries = [
        BenchmarkEntry(sample=sample, image_root=dataset.image_root)
        for dataset in datasets
        for sample in dataset.manifest.samples
    ]
    return sorted(entries, key=lambda entry: (entry.sample.dataset, entry.sample.sample_id))


def _combined_manifest_hash(datasets: Sequence[BenchmarkDataset]) -> str:
    canonical = "".join(
        sorted(dataset.manifest.dump_canonical_json() for dataset in datasets)
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _git_commit(repo_root: Optional[Union[str, Path]]) -> str:
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def runtime_environment() -> Dict[str, Any]:
    """Return stable environment facts useful for comparing benchmark runs."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpuCount": os.cpu_count(),
        "opencv": cv2.__version__,
        "opencvThreads": cv2.getNumThreads(),
        "numpy": np.__version__,
    }


def _coerce_points(
    value: Any, *, minimum: int = 4, maximum: Optional[int] = None
) -> Optional[PointList]:
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        return None
    if maximum is not None and len(value) > maximum:
        return None
    points: PointList = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        x, y = point
        if isinstance(x, bool) or isinstance(y, bool):
            return None
        try:
            normalized = (float(x), float(y))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in normalized):
            return None
        points.append(normalized)
    return points


def _signed_area(points: Sequence[BenchmarkPoint]) -> float:
    return (
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def _polygon_area(points: Sequence[BenchmarkPoint]) -> float:
    return abs(_signed_area(points))


def _cross(a: BenchmarkPoint, b: BenchmarkPoint, c: BenchmarkPoint) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _is_convex(points: Sequence[BenchmarkPoint]) -> bool:
    if len(points) < 3:
        return False
    crosses = [
        _cross(points[index], points[(index + 1) % len(points)], points[(index + 2) % len(points)])
        for index in range(len(points))
    ]
    return all(value > 1e-6 for value in crosses) or all(value < -1e-6 for value in crosses)


def _valid_quad(value: Any) -> Optional[PointList]:
    points = _coerce_points(value, minimum=4, maximum=4)
    if points is None or not _is_convex(points):
        return None
    if len({(round(x, 6), round(y, 6)) for x, y in points}) != 4:
        return None
    if _polygon_area(points) < 1.0:
        return None
    crosses = [
        _cross(points[index], points[(index + 1) % 4], points[(index + 2) % 4])
        for index in range(4)
    ]
    if any(value <= 1e-6 for value in crosses):
        return None
    sums = [point[0] + point[1] for point in points]
    differences = [point[1] - point[0] for point in points]
    expected = [
        min(range(4), key=lambda index: (sums[index], index)),
        min(range(4), key=lambda index: (differences[index], index)),
        max(range(4), key=lambda index: (sums[index], -index)),
        max(range(4), key=lambda index: (differences[index], -index)),
    ]
    if len(set(expected)) != 4 or expected != [0, 1, 2, 3]:
        return None
    return points


def _line_intersection(
    start: BenchmarkPoint,
    end: BenchmarkPoint,
    clip_start: BenchmarkPoint,
    clip_end: BenchmarkPoint,
) -> BenchmarkPoint:
    direction = (end[0] - start[0], end[1] - start[1])
    clip_direction = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denominator) < 1e-9:
        return end
    offset = (clip_start[0] - start[0], clip_start[1] - start[1])
    parameter = (offset[0] * clip_direction[1] - offset[1] * clip_direction[0]) / denominator
    return (start[0] + parameter * direction[0], start[1] + parameter * direction[1])


def _convex_intersection(
    subject: Sequence[BenchmarkPoint], clip: Sequence[BenchmarkPoint]
) -> PointList:
    clip_points = list(clip)
    if _signed_area(clip_points) < 0.0:
        clip_points.reverse()
    output = list(subject)
    for index, clip_start in enumerate(clip_points):
        if not output:
            break
        clip_end = clip_points[(index + 1) % len(clip_points)]
        previous = output[-1]
        next_output: PointList = []
        previous_inside = _cross(clip_start, clip_end, previous) >= -1e-7
        for current in output:
            current_inside = _cross(clip_start, clip_end, current) >= -1e-7
            if current_inside:
                if not previous_inside:
                    next_output.append(_line_intersection(previous, current, clip_start, clip_end))
                next_output.append(current)
            elif previous_inside:
                next_output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
            previous_inside = current_inside
        output = next_output
    return output


def polygon_iou(predicted: Any, expected: Any) -> Optional[float]:
    """Compute IoU for two convex polygons without rasterization."""
    first = _coerce_points(predicted, minimum=3)
    second = _coerce_points(expected, minimum=3)
    if first is None or second is None or not _is_convex(first) or not _is_convex(second):
        return None
    first_area = _polygon_area(first)
    second_area = _polygon_area(second)
    intersection = _polygon_area(_convex_intersection(first, second))
    union = first_area + second_area - intersection
    if union <= 1e-9:
        return None
    return max(0.0, min(1.0, intersection / union))


def corner_distance(
    predicted: Any,
    expected: Sequence[BenchmarkPoint],
    width: int,
    height: int,
) -> Optional[Dict[str, Any]]:
    """Return per-corner pixel errors and image-diagonal normalization."""
    first = _valid_quad(predicted)
    if first is None or len(expected) != 4:
        return None
    distances = [
        math.hypot(predicted_point[0] - expected_point[0], predicted_point[1] - expected_point[1])
        for predicted_point, expected_point in zip(first, expected, strict=True)
    ]
    diagonal = math.hypot(width, height)
    normalized = [distance / diagonal for distance in distances]
    return {
        "per_corner_px": dict(zip(("tl", "tr", "br", "bl"), distances, strict=True)),
        "per_corner_normalized": dict(zip(("tl", "tr", "br", "bl"), normalized, strict=True)),
        "mean_px": float(np.mean(distances)),
        "mean_normalized": float(np.mean(normalized)),
    }


def mask_iou(predicted: np.ndarray, expected: np.ndarray) -> Optional[float]:
    """Compute binary mask IoU."""
    if predicted.shape != expected.shape:
        return None
    predicted_mask = predicted.astype(bool)
    expected_mask = expected.astype(bool)
    union = np.logical_or(predicted_mask, expected_mask).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(predicted_mask, expected_mask).sum() / union)


def _resolve_relative_path(root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} must be a relative path inside the dataset root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes dataset root") from exc
    return resolved


def _load_mask(root: Path, path: str, width: int, height: int, *, label: str) -> np.ndarray:
    resolved = _resolve_relative_path(root, path, label=label)
    if not resolved.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    with Image.open(resolved) as image:
        image.load()
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError(
            f"{label} dimensions {mask.shape[1]}x{mask.shape[0]} do not match {width}x{height}"
        )
    return mask > 0


def _normalize_timing_key(key: str) -> Optional[str]:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    aliases = {
        "segmentationinference": "segmentation_inference_ms",
        "segmentationinferencems": "segmentation_inference_ms",
        "maskpostprocess": "mask_postprocess_ms",
        "maskpostprocessms": "mask_postprocess_ms",
        "quadrilateralfitting": "quadrilateral_fitting_ms",
        "quadrilateralfittingms": "quadrilateral_fitting_ms",
        "cvrefinement": "cv_refinement_ms",
        "cvrefinementms": "cv_refinement_ms",
        "detect": "total_detection_ms",
        "detectms": "total_detection_ms",
        "totaldetection": "total_detection_ms",
        "totaldetectionms": "total_detection_ms",
    }
    if key in TIMING_STAGES:
        return key
    return aliases.get(normalized)


def _normalized_timings(values: Mapping[str, float]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in values.items():
        target = _normalize_timing_key(key)
        if target is not None:
            normalized[target] = float(value)
    return normalized


def _run_reference_prediction(entry: BenchmarkEntry) -> DetectionPrediction:
    sample = entry.sample
    polygon = sample.document_polygon or sample.corners
    return DetectionPrediction(
        sample_id=sample.sample_id,
        detected=True,
        corners=sample.corners,
        polygon=polygon,
        mask_path=sample.mask_path,
        metadata={"adapter": "oracle_reference"},
    )


def _run_v1_prediction(entry: BenchmarkEntry) -> DetectionPrediction:
    tracing_before = tracemalloc.is_tracing()
    if not tracing_before:
        tracemalloc.start()
    try:
        loaded = load_image(entry.image_root / entry.sample.image_path)
        started_at = time.perf_counter()
        result = detect_document_boundary(loaded)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if not tracing_before:
            tracemalloc.stop()

    timings = {"total_detection_ms": elapsed_ms}
    memory = None if tracing_before else peak_bytes
    if result is None:
        return DetectionPrediction(
            sample_id=entry.sample.sample_id,
            detected=False,
            timings_ms=timings,
            memory_peak_bytes=memory,
            failure_reason="not_detected",
            metadata={"adapter": "v1_cv"},
        )
    if not result.accepted or result.clipped:
        reason = (
            result.rejection_reason.value.lower() if result.rejection_reason else "invalid_geometry"
        )
        return DetectionPrediction(
            sample_id=entry.sample.sample_id,
            detected=False,
            corners=result.corners,
            timings_ms=timings,
            memory_peak_bytes=memory,
            failure_reason=reason,
            metadata={"adapter": "v1_cv"},
        )
    return DetectionPrediction(
        sample_id=entry.sample.sample_id,
        detected=True,
        corners=result.corners,
        polygon=result.corners,
        timings_ms=timings,
        memory_peak_bytes=memory,
        metadata={"adapter": "v1_cv"},
    )


def _failure_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized or "unknown_failure"


def _score_sample(
    entry: BenchmarkEntry, prediction: Optional[DetectionPrediction]
) -> Dict[str, Any]:
    sample = entry.sample
    result: Dict[str, Any] = {
        "sample_id": sample.sample_id,
        "dataset": sample.dataset,
        "split": sample.split,
        "detected": False,
        "valid_geometry": False,
        "polygon_iou": None,
        "corner_error": None,
        "failure_reason": None,
        "mask_iou": None,
        "mask_coverage": None,
        "mask_component_count": None,
        "component_ambiguity": None,
        "timings_ms": {},
        "memory_peak_bytes": None,
    }
    if prediction is None:
        result["failure_reason"] = "missing_prediction"
        return result

    result["timings_ms"] = _normalized_timings(prediction.timings_ms)
    result["memory_peak_bytes"] = prediction.memory_peak_bytes
    if not prediction.detected:
        result["failure_reason"] = _failure_name(prediction.failure_reason or "not_detected")
        return result

    predicted_corners = _valid_quad(prediction.corners)
    if predicted_corners is None:
        result["failure_reason"] = _failure_name(prediction.failure_reason or "invalid_geometry")
        return result
    result["detected"] = True
    result["valid_geometry"] = True
    expected_polygon = sample.document_polygon or sample.corners
    predicted_polygon = _coerce_points(prediction.polygon, minimum=3)
    if predicted_polygon is None:
        predicted_polygon = predicted_corners
    result["polygon_iou"] = polygon_iou(predicted_polygon, expected_polygon)
    result["corner_error"] = corner_distance(
        predicted_corners,
        sample.corners,
        sample.width,
        sample.height,
    )

    if prediction.mask_path is not None:
        try:
            predicted_mask = _load_mask(
                entry.image_root,
                prediction.mask_path,
                sample.width,
                sample.height,
                label="prediction mask",
            )
            result["mask_coverage"] = float(predicted_mask.mean())
            components, _ = cv2.connectedComponents(predicted_mask.astype(np.uint8), connectivity=8)
            component_count = max(0, int(components) - 1)
            result["mask_component_count"] = component_count
            result["component_ambiguity"] = component_count > 1
            if sample.mask_path is not None:
                expected_mask = _load_mask(
                    entry.image_root,
                    sample.mask_path,
                    sample.width,
                    sample.height,
                    label="ground-truth mask",
                )
                result["mask_iou"] = mask_iou(predicted_mask, expected_mask)
        except ValueError as exc:
            result["mask_failure"] = _failure_name(str(exc))
    return result


def _percentile_summary(values: Sequence[float], suffix: str = "") -> Dict[str, Optional[float]]:
    if not values:
        return {
            f"mean{suffix}": None,
            f"p50{suffix}": None,
            f"p95{suffix}": None,
            f"p99{suffix}": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        f"mean{suffix}": float(np.mean(array)),
        f"p50{suffix}": float(np.percentile(array, 50)),
        f"p95{suffix}": float(np.percentile(array, 95)),
        f"p99{suffix}": float(np.percentile(array, 99)),
    }


def _timing_summary(samples: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for stage in TIMING_STAGES:
        values = [
            float(sample["timings_ms"][stage])
            for sample in samples
            if stage in sample["timings_ms"]
        ]
        percentiles = _percentile_summary(values, suffix="_ms")
        summary[stage] = {"count": len(values), **percentiles}
    return summary


def _memory_summary(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    values = [
        float(sample["memory_peak_bytes"])
        for sample in samples
        if sample["memory_peak_bytes"] is not None
    ]
    percentiles = _percentile_summary(values, suffix="_bytes")
    return {"count": len(values), **percentiles, "max_bytes": max(values) if values else None}


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def _metrics(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(samples)
    detected_count = sum(
        bool(sample["detected"]) and bool(sample["valid_geometry"]) for sample in samples
    )
    invalid_geometry_count = sum(
        sample["failure_reason"] == "invalid_geometry" for sample in samples
    )
    ious = [sample["polygon_iou"] for sample in samples if sample["polygon_iou"] is not None]
    corner_px = [
        value
        for sample in samples
        if sample["corner_error"] is not None
        for value in sample["corner_error"]["per_corner_px"].values()
    ]
    corner_normalized = [
        value
        for sample in samples
        if sample["corner_error"] is not None
        for value in sample["corner_error"]["per_corner_normalized"].values()
    ]
    mask_ious = [sample["mask_iou"] for sample in samples if sample["mask_iou"] is not None]
    mask_coverages = [
        sample["mask_coverage"] for sample in samples if sample["mask_coverage"] is not None
    ]
    ambiguities = [
        sample["component_ambiguity"]
        for sample in samples
        if sample["component_ambiguity"] is not None
    ]
    failures = Counter(
        sample["failure_reason"] for sample in samples if sample["failure_reason"] is not None
    )
    pixel_percentiles = _percentile_summary(corner_px, suffix="_px")
    normalized_percentiles = _percentile_summary(corner_normalized, suffix="_normalized")
    return {
        "sample_count": total,
        "detected_count": detected_count,
        "valid_geometry_count": sum(bool(sample["valid_geometry"]) for sample in samples),
        "detection_success_rate": detected_count / total if total else 0.0,
        "failure_count": total - detected_count,
        "false_detection_invalid_geometry_count": invalid_geometry_count,
        "false_detection_rate": invalid_geometry_count / total if total else 0.0,
        "polygon_iou_count": len(ious),
        "mean_iou": _mean(ious),
        "corner_error_count": len(corner_px),
        "mean_corner_error_px": pixel_percentiles["mean_px"],
        "p50_corner_error_px": pixel_percentiles["p50_px"],
        "p95_corner_error_px": pixel_percentiles["p95_px"],
        "p99_corner_error_px": pixel_percentiles["p99_px"],
        "mean_normalized_corner_error": normalized_percentiles["mean_normalized"],
        "p50_normalized_corner_error": normalized_percentiles["p50_normalized"],
        "p95_normalized_corner_error": normalized_percentiles["p95_normalized"],
        "p99_normalized_corner_error": normalized_percentiles["p99_normalized"],
        "mask_iou_count": len(mask_ious),
        "mean_mask_iou": _mean(mask_ious),
        "mask_coverage_count": len(mask_coverages),
        "mean_mask_coverage": _mean(mask_coverages),
        "component_count": len(ambiguities),
        "component_ambiguity_count": sum(bool(value) for value in ambiguities),
        "component_ambiguity_rate": (
            sum(bool(value) for value in ambiguities) / len(ambiguities) if ambiguities else None
        ),
        "failure_taxonomy": dict(sorted(failures.items())),
        "timings_ms": _timing_summary(samples),
        "memory": _memory_summary(samples),
    }


def evaluate_thresholds(
    metrics: Mapping[str, Any], thresholds: Optional[BenchmarkThresholds]
) -> Dict[str, Any]:
    """Evaluate configured thresholds without turning an unconfigured run into a failure."""
    if thresholds is None:
        return {"enabled": False, "passed": None, "thresholds": {}, "checks": {}}
    checks: Dict[str, Dict[str, Any]] = {}
    comparisons = (
        ("min_detection_success_rate", "detection_success_rate", ">="),
        ("min_mean_iou", "mean_iou", ">="),
        ("max_p95_normalized_corner_error", "p95_normalized_corner_error", "<="),
        ("max_false_detection_rate", "false_detection_rate", "<="),
    )
    for threshold_name, metric_name, operator in comparisons:
        target = getattr(thresholds, threshold_name)
        if target is None:
            continue
        value = metrics.get(metric_name)
        passed = value is not None and (value >= target if operator == ">=" else value <= target)
        checks[metric_name] = {
            "value": value,
            "target": target,
            "operator": operator,
            "passed": passed,
        }
    return {
        "enabled": True,
        "passed": all(check["passed"] for check in checks.values()),
        "thresholds": thresholds.model_dump(mode="json"),
        "checks": checks,
    }


def run_benchmark(
    datasets: Sequence[BenchmarkDataset],
    *,
    detector: DetectorName,
    prediction_bundle: Optional[PredictionBundle] = None,
    thresholds: Optional[BenchmarkThresholds] = None,
    model_version: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    repo_root: Optional[Union[str, Path]] = None,
) -> BenchmarkReport:
    """Run one detector adapter and return deterministic per-sample metrics."""
    entries = _flatten_entries(datasets)
    external_predictions: Optional[Dict[str, DetectionPrediction]] = None
    if prediction_bundle is not None:
        if detector in {"v1_cv", "reference"}:
            raise ValueError(
                f"Prediction JSON cannot override built-in detector {detector!r}; "
                "use a V2 adapter name for external predictions"
            )
        if prediction_bundle.detector not in (None, detector):
            raise ValueError(
                f"Prediction detector {prediction_bundle.detector!r} does not match {detector!r}"
            )
        external_predictions = {
            prediction.sample_id: prediction for prediction in prediction_bundle.predictions
        }
        known_ids = {entry.sample.sample_id for entry in entries}
        unknown_ids = sorted(set(external_predictions) - known_ids)
        if unknown_ids:
            raise ValueError(f"Prediction file contains unknown sample_id: {unknown_ids[0]}")
    elif detector in {"segmentation_only", "cv_v2", "hybrid"}:
        raise ValueError(f"Detector {detector!r} requires a prediction JSON adapter input")

    scored: List[Dict[str, Any]] = []
    for entry in entries:
        if external_predictions is not None:
            prediction = external_predictions.get(entry.sample.sample_id)
        elif detector == "reference":
            prediction = _run_reference_prediction(entry)
        else:
            prediction = _run_v1_prediction(entry)
        scored.append(_score_sample(entry, prediction))

    effective_model_version = (
        model_version
        or (prediction_bundle.model_version if prediction_bundle is not None else None)
        or ("oracle-reference" if detector == "reference" else detector)
    )
    effective_pipeline_version = (
        pipeline_version
        or (prediction_bundle.pipeline_version if prediction_bundle is not None else None)
        or "benchmark-runner-1.0"
    )
    metric_values = _metrics(scored)
    return BenchmarkReport(
        detector=detector,
        model_version=effective_model_version,
        pipeline_version=effective_pipeline_version,
        git_commit=_git_commit(repo_root),
        dataset_manifest_sha256=_combined_manifest_hash(datasets),
        runtime_environment=runtime_environment(),
        metrics=metric_values,
        threshold_gate=evaluate_thresholds(metric_values, thresholds),
        samples=scored,
    )


def render_markdown(report: BenchmarkReport) -> str:
    """Render a compact human-readable report from the canonical report model."""
    metrics = report.metrics
    lines = [
        "# V2 Benchmark Report",
        "",
        f"- Detector: `{report.detector}`",
        f"- Model version: `{report.model_version}`",
        f"- Pipeline version: `{report.pipeline_version}`",
        f"- Git commit: `{report.git_commit}`",
        f"- Dataset manifest SHA-256: `{report.dataset_manifest_sha256}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key in (
        "sample_count",
        "detection_success_rate",
        "mean_iou",
        "mean_corner_error_px",
        "p50_corner_error_px",
        "p95_corner_error_px",
        "p99_corner_error_px",
        "mean_normalized_corner_error",
        "p95_normalized_corner_error",
        "false_detection_rate",
        "mean_mask_iou",
        "mean_mask_coverage",
        "component_ambiguity_rate",
    ):
        lines.append(f"| `{key}` | {metrics.get(key)} |")
    lines.extend(
        [
            "",
            "## Failure taxonomy",
            "",
            "| Failure | Count |",
            "| --- | ---: |",
        ]
    )
    for key, value in metrics["failure_taxonomy"].items():
        lines.append(f"| `{key}` | {value} |")
    gate = report.threshold_gate
    lines.extend(
        [
            "",
            "## Threshold gate",
            "",
            f"Enabled: `{gate['enabled']}`; passed: `{gate['passed']}`.",
            "",
            "## Per-sample evidence",
            "",
            "| Sample | Detected | IoU | Mean corner px | Normalized mean | Failure |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for sample in report.samples:
        corner = sample["corner_error"] or {}
        lines.append(
            (
                "| `{sample_id}` | {detected} | {iou} | {corner_px} | {corner_norm} | {failure} |"
            ).format(
                sample_id=sample["sample_id"],
                detected=sample["detected"],
                iou=sample["polygon_iou"],
                corner_px=corner.get("mean_px"),
                corner_norm=corner.get("mean_normalized"),
                failure=sample["failure_reason"] or "",
            )
        )
    return "\n".join(lines) + "\n"


def render_csv(report: BenchmarkReport) -> str:
    """Render per-sample evidence as a stable CSV summary."""
    columns = [
        "sample_id",
        "dataset",
        "split",
        "detected",
        "valid_geometry",
        "polygon_iou",
        "mean_corner_error_px",
        "mean_corner_error_normalized",
        "mask_iou",
        "mask_coverage",
        "mask_component_count",
        "component_ambiguity",
        "failure_reason",
    ]
    rows: List[Dict[str, Any]] = []
    for sample in report.samples:
        corner = sample["corner_error"] or {}
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "split": sample["split"],
                "detected": sample["detected"],
                "valid_geometry": sample["valid_geometry"],
                "polygon_iou": sample["polygon_iou"],
                "mean_corner_error_px": corner.get("mean_px"),
                "mean_corner_error_normalized": corner.get("mean_normalized"),
                "mask_iou": sample["mask_iou"],
                "mask_coverage": sample["mask_coverage"],
                "mask_component_count": sample["mask_component_count"],
                "component_ambiguity": sample["component_ambiguity"],
                "failure_reason": sample["failure_reason"],
            }
        )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_report_files(
    report: BenchmarkReport,
    *,
    json_path: Union[str, Path],
    markdown_path: Optional[Union[str, Path]] = None,
    csv_path: Optional[Union[str, Path]] = None,
) -> None:
    """Write requested report artifacts atomically."""
    atomic_write_text(json_path, report.dump_canonical_json())
    if markdown_path is not None:
        atomic_write_text(markdown_path, render_markdown(report))
    if csv_path is not None:
        atomic_write_text(csv_path, render_csv(report))


__all__ = [
    "BENCHMARK_RUNNER_VERSION",
    "BenchmarkDataset",
    "BenchmarkReport",
    "BenchmarkThresholds",
    "DetectionPrediction",
    "PredictionBundle",
    "corner_distance",
    "evaluate_thresholds",
    "load_benchmark_datasets",
    "load_prediction_bundle",
    "mask_iou",
    "polygon_iou",
    "render_csv",
    "render_markdown",
    "run_benchmark",
    "runtime_environment",
    "write_report_files",
]
