"""Full V2 detector comparison and production-candidate decision report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from pydantic import Field

from attendance_scanner.benchmark_runner import (
    BenchmarkDataset,
    BenchmarkReport,
    BenchmarkThresholds,
    combined_manifest_sha256,
    load_prediction_bundle,
    run_benchmark,
)
from attendance_scanner.contracts import BaseContract

REQUIRED_DETECTORS = (
    "v1_cv",
    "segmentation_only",
    "cv_v2",
    "hybrid",
    "reference",
)


class DetectorComparison(BaseContract):
    """One adapter's availability, metrics, provenance and optional performance facts."""

    detector: str
    available: bool
    model_version: Optional[str] = None
    model_checksum: Optional[str] = None
    prediction_bundle_sha256: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    threshold_gate: Dict[str, Any] = Field(default_factory=dict)
    scenario_metrics: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    performance: Optional[Dict[str, Any]] = None
    unavailable_reason: Optional[str] = None


class BenchmarkDecisionReport(BaseContract):
    """Reproducible comparison report with explicit blocked/ready decision state."""

    schema_version: str = "1.0"
    dataset_manifest_sha256: str
    git_commit: str
    required_detectors: List[str]
    comparisons: Dict[str, DetectorComparison]
    decision_status: str
    selected_production_detector: Optional[str] = None
    decision_reason: str = Field(min_length=1)
    blockers: List[str] = Field(default_factory=list)
    remediation: List[str] = Field(default_factory=list)

    def dump_canonical_json(self) -> str:
        return (
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario_metrics(report: BenchmarkReport) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in report.samples:
        tags = sample.get("scenario_tags") or ["untagged"]
        for tag in tags:
            grouped.setdefault(str(tag), []).append(sample)

    result: Dict[str, Dict[str, Any]] = {}
    for tag, samples in sorted(grouped.items()):
        ious = [
            float(sample["polygon_iou"])
            for sample in samples
            if sample.get("polygon_iou") is not None
        ]
        errors = [
            float(sample["corner_error"]["mean_normalized"])
            for sample in samples
            if sample.get("corner_error") is not None
        ]
        result[tag] = {
            "sample_count": len(samples),
            "detection_success_rate": sum(
                bool(sample.get("detected")) and bool(sample.get("valid_geometry"))
                for sample in samples
            )
            / len(samples),
            "mean_iou": sum(ious) / len(ious) if ious else None,
            "p95_normalized_corner_error": (
                sorted(errors)[max(0, round(0.95 * len(errors)) - 1)] if errors else None
            ),
            "silent_wrong_crop_count": sum(
                bool(sample.get("detected"))
                and bool(sample.get("valid_geometry"))
                and sample.get("polygon_iou") is not None
                and float(sample["polygon_iou"]) < 0.5
                for sample in samples
            ),
        }
    return result


def _git_commit(repo_root: Optional[Union[str, Path]]) -> str:
    from attendance_scanner.benchmark_runner import _git_commit as read_git_commit

    return read_git_commit(repo_root)


def run_full_v2_decision(
    datasets: Sequence[BenchmarkDataset],
    *,
    prediction_bundle_paths: Optional[Mapping[str, Union[str, Path]]] = None,
    performance_reports: Optional[Mapping[str, Union[str, Path]]] = None,
    repo_root: Optional[Union[str, Path]] = None,
    thresholds: Optional[BenchmarkThresholds] = None,
) -> BenchmarkDecisionReport:
    """Run all available adapters and block selection when evidence is missing."""
    effective_thresholds = thresholds or BenchmarkThresholds.release_targets()
    bundle_paths = prediction_bundle_paths or {}
    bundles: Dict[str, Any] = {}
    comparisons: Dict[str, DetectorComparison] = {}
    reports: Dict[str, BenchmarkReport] = {}
    blockers: List[str] = []
    for detector, path in bundle_paths.items():
        try:
            bundles[detector] = load_prediction_bundle(path)
        except (OSError, ValueError) as exc:
            comparisons[detector] = DetectorComparison(
                detector=detector,
                available=False,
                unavailable_reason=f"Invalid prediction bundle: {exc}",
            )
            blockers.append(f"Invalid prediction bundle for {detector}")

    for detector in REQUIRED_DETECTORS:
        bundle = bundles.get(detector)
        if detector in {"v1_cv", "reference"}:
            report = run_benchmark(
                datasets,
                detector=detector,  # type: ignore[arg-type]
                thresholds=effective_thresholds,
                repo_root=repo_root,
            )
        elif bundle is None:
            reason = f"Missing prediction bundle for {detector}"
            comparisons[detector] = DetectorComparison(
                detector=detector,
                available=False,
                unavailable_reason=reason,
            )
            blockers.append(reason)
            continue
        elif detector in {"segmentation_only", "cv_v2", "hybrid"} and bundle.model_checksum is None:
            reason = f"Missing model checksum for {detector}"
            comparisons[detector] = DetectorComparison(
                detector=detector,
                available=False,
                model_version=bundle.model_version,
                unavailable_reason=reason,
            )
            blockers.append(reason)
            continue
        else:
            report = run_benchmark(
                datasets,
                detector=detector,  # type: ignore[arg-type]
                prediction_bundle=bundle,
                thresholds=effective_thresholds,
                repo_root=repo_root,
            )

        reports[detector] = report
        bundle_path = bundle_paths.get(detector)
        comparisons[detector] = DetectorComparison(
            detector=detector,
            available=True,
            model_version=report.model_version,
            model_checksum=bundle.model_checksum if bundle is not None else None,
            prediction_bundle_sha256=(
                _file_sha256(Path(bundle_path))
                if bundle_path is not None and Path(bundle_path).is_file()
                else None
            ),
            metrics=report.metrics,
            threshold_gate=report.threshold_gate,
            scenario_metrics=_scenario_metrics(report),
            performance=(
                json.loads(Path(performance_reports[detector]).read_text(encoding="utf-8"))
                if performance_reports
                and detector in performance_reports
                and Path(performance_reports[detector]).is_file()
                else None
            ),
        )

    baseline = reports.get("v1_cv")
    hybrid = reports.get("hybrid")
    selected: Optional[str] = None
    if not blockers and baseline is not None and hybrid is not None:
        baseline_iou = baseline.metrics.get("mean_iou") or 0.0
        hybrid_iou = hybrid.metrics.get("mean_iou") or 0.0
        baseline_success = baseline.metrics.get("detection_success_rate") or 0.0
        hybrid_success = hybrid.metrics.get("detection_success_rate") or 0.0
        baseline_corner = baseline.metrics.get("p95_normalized_corner_error")
        hybrid_corner = hybrid.metrics.get("p95_normalized_corner_error")
        hybrid_silent_wrong = hybrid.metrics.get("silent_wrong_crop_count") or 0
        if hybrid.threshold_gate.get("passed") is True and (
            hybrid_silent_wrong == 0
            and hybrid_success >= baseline_success
            and hybrid_iou >= baseline_iou
            and (
                baseline_corner is None or hybrid_corner is None or hybrid_corner <= baseline_corner
            )
            and (
                hybrid_iou > baseline_iou
                or hybrid_success > baseline_success
                or (
                    baseline_corner is not None
                    and hybrid_corner is not None
                    and hybrid_corner < baseline_corner
                )
            )
        ):
            selected = "hybrid"
            reason = "Hybrid vượt baseline V1 và quality gate trên đầy đủ benchmark."
        else:
            blockers.append(
                "Hybrid chưa chứng minh accuracy/reliability vượt V1 mà không có "
                "silent wrong crop hoặc regression metric."
            )
            reason = "Chưa đủ bằng chứng để thay đổi production detector mặc định."
    else:
        reason = "Chưa thể quyết định vì thiếu prediction bundle/model evidence bắt buộc."

    status = "READY" if selected is not None and not blockers else "BLOCKED"
    remediation = [
        "Cung cấp bundle prediction và model checksum cho segmentation_only, cv_v2 và hybrid.",
        (
            "Chạy lại trên corpus có tag grid-heavy, low-contrast, paper-overlap, "
            "border-touching, shadow, perspective và blur."
        ),
        (
            "Đối chiếu accuracy với AS-35 và performance report AS-58 trước khi "
            "chọn production provider."
        ),
    ]
    git_commit = next(iter(reports.values())).git_commit if reports else _git_commit(repo_root)
    return BenchmarkDecisionReport(
        dataset_manifest_sha256=combined_manifest_sha256(datasets),
        git_commit=git_commit,
        required_detectors=list(REQUIRED_DETECTORS),
        comparisons=comparisons,
        decision_status=status,
        selected_production_detector=selected,
        decision_reason=reason,
        blockers=sorted(set(blockers)),
        remediation=remediation if status == "BLOCKED" else [],
    )


def render_decision_markdown(report: BenchmarkDecisionReport) -> str:
    lines = [
        "# V2 Detector Decision Report",
        "",
        f"- Status: `{report.decision_status}`",
        f"- Selected production detector: `{report.selected_production_detector or 'none'}`",
        f"- Git commit: `{report.git_commit}`",
        f"- Dataset manifest SHA-256: `{report.dataset_manifest_sha256}`",
        "",
        "| Detector | Available | Success rate | Mean IoU | Silent wrong crop |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for detector in report.required_detectors:
        comparison = report.comparisons[detector]
        metrics = comparison.metrics
        lines.append(
            f"| `{detector}` | {comparison.available} | "
            f"{metrics.get('detection_success_rate')} | {metrics.get('mean_iou')} | "
            f"{metrics.get('silent_wrong_crop_count')} |"
        )
    lines.extend(["", "## Decision", "", report.decision_reason, ""])
    if report.blockers:
        lines.extend(["## Blockers", ""])
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    if report.remediation:
        lines.extend(["", "## Remediation", ""])
        lines.extend(f"- {item}" for item in report.remediation)
    return "\n".join(lines) + "\n"


__all__ = [
    "BenchmarkDecisionReport",
    "DetectorComparison",
    "REQUIRED_DETECTORS",
    "render_decision_markdown",
    "run_full_v2_decision",
]
