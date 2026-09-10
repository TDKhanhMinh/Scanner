"""Resumable, bounded-concurrency batch processing for scanner source files."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Union

from attendance_scanner.contracts import (
    BatchPeriod,
    BatchSummary,
    CompletenessStatus,
    DiscoveredFile,
    DiscoveryResult,
    DocumentGroupKey,
    ExportMode,
    FileProcessingStatus,
    FileResult,
    ImageProcessError,
    InvalidInputRootError,
    PageIdentity,
    ScanMode,
    ScannerErrorCode,
)
from attendance_scanner.diagnostics import describe_scanner_error, log_scanner_error
from attendance_scanner.discovery import (
    DEFAULT_PIPELINE_VERSION,
    GroupAwareScanPlan,
    pipeline_version_for_mode,
)
from attendance_scanner.events import (
    BaseEvent,
    FileCompletedEvent,
    FileFailedEvent,
    FileStartedEvent,
    ScanCompletedEvent,
    serialize_event,
)
from attendance_scanner.export import (
    ExportPage,
    ExportReviewRequiredError,
    export_grouped,
)
from attendance_scanner.fingerprint import compute_fast_fingerprint, compute_sha256
from attendance_scanner.pdf_export import PdfExportConfig, export_single_page_pdf
from attendance_scanner.pipeline.orchestrator import (
    PipelineConfig,
    SingleScanResult,
    scan_one,
)
from attendance_scanner.state import (
    Manifest,
    ManifestArtifact,
    ManifestEntry,
    ManifestGroup,
    ManifestStore,
    assign_entry_context,
    set_manual_group_order,
)

logger = logging.getLogger(__name__)

EventEmitter = Callable[[BaseEvent], None]


def _group_id(key: DocumentGroupKey) -> str:
    """Return the stable UI/CLI identifier for an employee-period group."""
    return f"{key.employee_relative_dir}:{key.year:04d}-{key.month:02d}"


def _aggregate_detection_metadata(entries: List[ManifestEntry]) -> Dict[str, Any]:
    """Aggregate per-page detector provenance without misrepresenting mixed groups."""

    def distinct_or_none(values: List[Optional[str]]) -> Optional[str]:
        distinct = {value for value in values if value is not None}
        return next(iter(distinct)) if len(distinct) == 1 else None

    statuses = {entry.detection_status for entry in entries if entry.detection_status is not None}
    if "failed" in statuses:
        status: Optional[str] = "failed"
    elif "fallback" in statuses:
        status = "fallback"
    elif statuses == {"detected"}:
        status = "detected"
    else:
        status = next(iter(statuses)) if len(statuses) == 1 else None
    fallback_values = [
        entry.detection_fallback_used
        for entry in entries
        if entry.detection_fallback_used is not None
    ]
    return {
        "pipeline_version": distinct_or_none([entry.pipeline_version for entry in entries]),
        "detector_name": distinct_or_none([entry.detector_name for entry in entries]),
        "detector_mode": distinct_or_none([entry.detector_mode for entry in entries]),
        "detector_model_version": distinct_or_none(
            [entry.detector_model_version for entry in entries]
        ),
        "detector_model_checksum": distinct_or_none(
            [entry.detector_model_checksum for entry in entries]
        ),
        "detection_status": status,
        "detection_fallback_used": (any(fallback_values) if fallback_values else None),
        "detection_quality_summary": {
            "pageCount": len(entries),
            "fallbackPageCount": sum(value is True for value in fallback_values),
            "failedPageCount": sum(entry.detection_status == "failed" for entry in entries),
            "detectorVariantCount": len(
                {(entry.detector_name, entry.detector_model_version) for entry in entries}
            ),
        },
    }


def _persist_manual_orders(
    manifest: Manifest,
    manifest_store: ManifestStore,
    discovery: DiscoveryResult,
    period: Optional[BatchPeriod],
    manual_orders: Optional[Dict[str, List[str]]],
) -> None:
    """Persist resolved UI orders after all current source entries are available."""
    if period is None or not manual_orders:
        return
    discovered_by_employee: Dict[str, List[str]] = {}
    for file in discovery.files:
        discovered_by_employee.setdefault(file.employee_name, []).append(
            file.relative_path.replace("\\", "/")
        )
    changed = False
    for group_id, ordered_paths in manual_orders.items():
        if ":" not in group_id:
            continue
        employee, period_text = group_id.rsplit(":", 1)
        try:
            year_text, month_text = period_text.split("-", 1)
            group_period = BatchPeriod(year=int(year_text), month=int(month_text))
        except (ValueError, TypeError):
            continue
        if group_period != period or employee not in discovered_by_employee:
            continue
        source_paths = discovered_by_employee[employee]
        source_entries = [manifest.get_entry(path) for path in source_paths]
        if any(entry is None for entry in source_entries):
            continue
        set_manual_group_order(
            manifest,
            group_key=DocumentGroupKey(
                employee_relative_dir=employee,
                year=group_period.year,
                month=group_period.month,
            ),
            ordered_source_paths=ordered_paths,
            source_entries=[entry for entry in source_entries if entry is not None],
        )
        changed = True
    if changed:
        manifest_store.save_manifest(manifest)


def _validate_grouped_run(
    group_plan: GroupAwareScanPlan,
    manual_orders: Optional[Dict[str, List[str]]],
    skip_groups: Optional[Set[str]],
) -> None:
    """Reject grouped execution unless every review group has an explicit decision."""
    orders = manual_orders or {}
    skipped = skip_groups or set()
    for review_group in group_plan.review_groups:
        group_id = _group_id(review_group.key)
        if group_id not in skipped and group_id not in orders:
            raise ExportReviewRequiredError(
                f"Grouped export requires page-order review for group {group_id!r}"
            )
    for group in group_plan.affected_groups:
        group_id = _group_id(group.key)
        if group_id in skipped or group_id not in orders:
            continue
        normalized_order = [path.replace("\\", "/") for path in orders[group_id]]
        expected_paths = {path.replace("\\", "/") for path in group.source_relative_paths}
        if (
            len(normalized_order) != len(set(normalized_order))
            or set(normalized_order) != expected_paths
        ):
            raise ExportReviewRequiredError(
                f"Manual page order does not match current sources for group {group_id!r}"
            )


_GROUP_EXPORT_BLOCKED_MESSAGE = (
    "Grouped artifact was not committed because another source page in this group failed"
)


def _block_failed_groups(
    outcomes: Dict[int, _FileOutcome],
    selected_files: List[DiscoveredFile],
    group_plan: GroupAwareScanPlan,
    manifest: Manifest,
    manifest_store: ManifestStore,
) -> None:
    """Keep file results and manifest state consistent when a group cannot commit."""
    outcome_by_path = {
        file.relative_path.replace("\\", "/"): outcomes[index]
        for index, file in enumerate(selected_files, start=1)
    }
    changed = False
    for group in group_plan.affected_groups:
        group_outcomes = [
            outcome_by_path.get(path.replace("\\", "/")) for path in group.source_relative_paths
        ]
        if not any(
            outcome is not None and outcome.file_result.status == FileProcessingStatus.FAILED
            for outcome in group_outcomes
        ):
            continue
        for outcome in group_outcomes:
            if outcome is None:
                continue
            outcome.file_result.target_relative_pdf = ""
            if outcome.file_result.status != FileProcessingStatus.FAILED:
                outcome.file_result.status = FileProcessingStatus.FAILED
                outcome.file_result.error_code = ScannerErrorCode.GROUP_EXPORT_BLOCKED
                outcome.file_result.error_message = _GROUP_EXPORT_BLOCKED_MESSAGE
                outcome.completed_event = None
                outcome.failed_event = FileFailedEvent(
                    relative_path=outcome.file_result.relative_path,
                    employee_name=outcome.file_result.employee_name,
                    error_code=ScannerErrorCode.GROUP_EXPORT_BLOCKED,
                    message=_GROUP_EXPORT_BLOCKED_MESSAGE,
                )
            entry = manifest.get_entry(outcome.file_result.relative_path)
            if entry is not None:
                entry.status = FileProcessingStatus.FAILED
                entry.output_relative_path = None
                entry.output_relative_paths = []
                entry.artifact_dependencies = []
                manifest.set_entry(entry)
                changed = True
        persisted_group = manifest.groups.get(_group_id(group.key))
        if persisted_group is not None:
            persisted_group.artifact_stale = True
            persisted_group.completeness_status = CompletenessStatus.AMBIGUOUS
            persisted_group.review_required = True
            manifest.set_group(persisted_group)
            for artifact_path in persisted_group.artifact_relative_paths:
                artifact = manifest.artifacts.get(artifact_path)
                if artifact is not None:
                    artifact.stale = True
                    manifest.set_artifact(artifact)
            changed = True
    if changed:
        manifest_store.save_manifest(manifest)


def _export_grouped_results(
    outcomes: Dict[int, _FileOutcome],
    selected_files: List[DiscoveredFile],
    group_plan: GroupAwareScanPlan,
    manifest: Manifest,
    manifest_store: ManifestStore,
    output_root: Path,
    pdf_config: Optional[PdfExportConfig],
    manual_orders: Optional[Dict[str, List[str]]],
    skip_groups: Set[str],
) -> Dict[str, str]:
    """Build grouped PDFs and persist group/artifact dependencies atomically."""
    outcome_by_path = {
        file.relative_path.replace("\\", "/"): outcomes[index]
        for index, file in enumerate(selected_files, start=1)
    }
    exported_path_by_source: Dict[str, str] = {}
    changed = False
    for group in group_plan.affected_groups:
        group_id = _group_id(group.key)
        if group_id in skip_groups:
            continue
        group_pages: List[ExportPage] = []
        source_entries: List[ManifestEntry] = []
        group_failed = False
        for source_path in group.source_relative_paths:
            normalized_path = source_path.replace("\\", "/")
            outcome = outcome_by_path.get(normalized_path)
            entry = manifest.get_entry(normalized_path)
            if (
                outcome is None
                or outcome.scan_result is None
                or outcome.file_result.status == FileProcessingStatus.FAILED
                or entry is None
            ):
                group_failed = True
                break
            source_entries.append(entry)
            identity = entry.page_identity
            group_pages.append(
                ExportPage(
                    image=outcome.scan_result.image,
                    source_relative_path=normalized_path,
                    employee_name=group.key.employee_relative_dir,
                    page_type=identity.page_type,
                    page_order=identity.page_order,
                    confidence=identity.confidence,
                    detection_method=identity.detection_method,
                )
            )
        if group_failed:
            continue

        persisted_group = manifest.groups.get(group_id)
        effective_orders: Dict[str, List[str]] = {}
        if manual_orders and group_id in manual_orders:
            effective_orders[group_id] = list(manual_orders[group_id])
        elif persisted_group and persisted_group.manual_order:
            effective_orders[group_id] = list(persisted_group.manual_order)
        artifacts = export_grouped(
            group_pages,
            output_root,
            group_plan.period,
            config=pdf_config,
            manual_orders=effective_orders or None,
        )
        if len(artifacts) != 1:
            raise ValueError(f"Expected one grouped artifact for group {group_id!r}")
        artifact = artifacts[0]
        artifact_relative_path = artifact.output_path.resolve().relative_to(output_root.resolve())
        artifact_relative_path_text = artifact_relative_path.as_posix()
        for source_path in artifact.source_relative_paths:
            exported_path_by_source[source_path.replace("\\", "/")] = artifact_relative_path_text

        explicit_order = effective_orders.get(group_id)
        if explicit_order:
            set_manual_group_order(
                manifest,
                group_key=group.key,
                ordered_source_paths=artifact.source_relative_paths,
                source_entries=source_entries,
            )
            persisted_group = manifest.groups[group_id]
        else:
            persisted_group = ManifestGroup(
                key=group.key,
                source_relative_paths=list(artifact.source_relative_paths),
                artifact_relative_paths=[artifact_relative_path_text],
                completeness_status=CompletenessStatus.COMPLETE,
                review_required=False,
                manual_order=(list(persisted_group.manual_order) if persisted_group else []),
                manual_order_fingerprint=(
                    persisted_group.manual_order_fingerprint if persisted_group else None
                ),
                artifact_stale=False,
            )
        persisted_group.source_relative_paths = list(artifact.source_relative_paths)
        persisted_group.artifact_relative_paths = [artifact_relative_path_text]
        persisted_group.completeness_status = CompletenessStatus.COMPLETE
        persisted_group.review_required = False
        persisted_group.artifact_stale = False
        manifest.set_group(persisted_group)
        detection_metadata = _aggregate_detection_metadata(source_entries)
        manifest.set_artifact(
            ManifestArtifact(
                export_mode=ExportMode.GROUPED,
                output_relative_path=artifact_relative_path_text,
                source_relative_paths=list(artifact.source_relative_paths),
                artifact_version=DEFAULT_PIPELINE_VERSION,
                pipeline_version=detection_metadata["pipeline_version"],
                detector_name=detection_metadata["detector_name"],
                detector_mode=detection_metadata["detector_mode"],
                detector_model_version=detection_metadata["detector_model_version"],
                detector_model_checksum=detection_metadata["detector_model_checksum"],
                detection_status=detection_metadata["detection_status"],
                detection_fallback_used=detection_metadata["detection_fallback_used"],
                detection_quality_summary=detection_metadata["detection_quality_summary"],
                stale=False,
            )
        )
        for entry in source_entries:
            entry.output_relative_path = artifact_relative_path_text
            entry.output_relative_paths = [artifact_relative_path_text]
            entry.artifact_dependencies = [artifact_relative_path_text]
            manifest.set_entry(entry)
        changed = True
    if changed:
        manifest_store.save_manifest(manifest)
    return exported_path_by_source


def default_worker_count() -> int:
    """Return the default worker count required by the system design."""
    return min(3, max(1, (os.cpu_count() or 1) - 1))


def clamp_worker_count(workers: Optional[int]) -> int:
    """Clamp an optional worker count to the supported inclusive range 1..4."""
    requested = default_worker_count() if workers is None else workers
    if isinstance(requested, bool):
        raise ValueError("workers must be an integer between 1 and 4")
    try:
        normalized = int(requested)
    except (TypeError, ValueError) as exc:
        raise ValueError("workers must be an integer between 1 and 4") from exc
    return min(4, max(1, normalized))


def _normalize_scan_mode(mode: Union[ScanMode, str]) -> ScanMode:
    """Normalize CLI aliases to the canonical scan mode enum."""
    if isinstance(mode, ScanMode):
        return mode
    normalized = mode.strip().lower()
    if normalized in ("color_enhanced", "colored"):
        normalized = ScanMode.COLOR.value
    elif normalized in ("smart", "smart-document"):
        normalized = ScanMode.SMART_DOCUMENT.value
    try:
        return ScanMode(normalized)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Unsupported scan mode: {mode!r}") from exc


def _resolve_output_path(output_root: Union[str, Path], relative_path: str) -> Path:
    """Resolve a manifest/discovery output path without allowing root escape."""
    root = Path(output_root).resolve()
    normalized = str(relative_path).replace("\\", "/")
    candidate = (root / Path(normalized)).resolve()

    def comparison_path(path: Path) -> str:
        """Normalize Windows extended paths before comparing containment."""
        value = str(path)
        if value.startswith("\\\\?\\UNC\\"):
            value = "\\\\" + value[8:]
        elif value.startswith("\\\\?\\"):
            value = value[4:]
        return os.path.normcase(value)

    try:
        root_comparison = comparison_path(root)
        candidate_comparison = comparison_path(candidate)
        if os.path.commonpath([root_comparison, candidate_comparison]) != root_comparison:
            raise ValueError("resolved candidate is outside output root")
    except ValueError as exc:
        raise ValueError(f"Output path escapes output root: {relative_path!r}") from exc
    return candidate


def _error_details(
    exc: BaseException,
    *,
    relative_path: Optional[str] = None,
) -> tuple[ScannerErrorCode, str]:
    """Map and log worker exceptions without exposing diagnostic details to users."""
    info = describe_scanner_error(exc, operation="file", relative_path=relative_path)
    log_scanner_error(info, exc, operation="file")
    return info.code, info.user_message


def _persist_manifest_entry(
    manifest: Manifest,
    manifest_store: ManifestStore,
    entry: ManifestEntry,
    manifest_lock: Lock,
) -> None:
    """Commit one terminal file state while serializing manifest writes."""
    with manifest_lock:
        manifest.set_entry(entry)
        manifest_store.save_manifest(manifest)


def _best_effort_persist_failure(
    manifest: Manifest,
    manifest_store: ManifestStore,
    entry: ManifestEntry,
    manifest_lock: Lock,
) -> None:
    """Persist failure metadata without hiding the original file failure."""
    try:
        _persist_manifest_entry(manifest, manifest_store, entry, manifest_lock)
    except Exception:
        logger.exception("Failed to persist failure state for %s", entry.relative_path)


def _make_failure_entry(
    file_relative_path: str,
    file_size: int,
    file_mtime_ns: int,
    file_sha256: Optional[str],
    target_relative_pdf: str,
    manifest: Manifest,
    pipeline_version: str,
    batch_period: Optional[BatchPeriod] = None,
    employee_relative_dir: Optional[str] = None,
    detector_name: Optional[str] = "v1_cv",
    detector_mode: Optional[str] = None,
    detector_model_version: Optional[str] = "opencv-classical",
    detector_model_checksum: Optional[str] = None,
) -> ManifestEntry:
    """Build retryable failure metadata while preserving prior artifacts."""
    previous = manifest.get_entry(file_relative_path)
    if previous is None:
        output_relative_paths = [target_relative_pdf]
        output_relative_path: Optional[str] = target_relative_pdf
        stored_pipeline_version = pipeline_version
    else:
        output_relative_paths = list(previous.output_relative_paths)
        output_relative_path = previous.output_relative_path
        if not output_relative_paths and output_relative_path is None:
            output_relative_paths = [target_relative_pdf]
            output_relative_path = target_relative_pdf
        stored_pipeline_version = pipeline_version

    failure_entry = ManifestEntry(
        relative_path=file_relative_path,
        size=file_size,
        mtime_ns=file_mtime_ns,
        sha256=file_sha256,
        output_relative_path=output_relative_path,
        output_relative_paths=output_relative_paths,
        status=FileProcessingStatus.FAILED,
        processed_at=datetime.now(timezone.utc).isoformat(),
        pipeline_version=stored_pipeline_version,
        period=previous.period if previous is not None else None,
        group_key=previous.group_key if previous is not None else None,
        page_identity=previous.page_identity if previous is not None else PageIdentity(),
        artifact_dependencies=(
            list(previous.artifact_dependencies) if previous is not None else output_relative_paths
        ),
        detector_name=detector_name,
        detector_mode=detector_mode,
        detector_model_version=detector_model_version,
        detector_model_checksum=detector_model_checksum,
        detection_status="failed",
        detection_fallback_used=None,
        detection_quality_summary={"error": "file_processing_failed"},
    )
    if batch_period is not None and employee_relative_dir is not None:
        assign_entry_context(
            failure_entry,
            employee_relative_dir=employee_relative_dir,
            batch_period=batch_period,
        )
    return failure_entry


@dataclass
class _FileOutcome:
    """Internal result returned by one isolated worker."""

    file_result: FileResult
    completed_event: Optional[FileCompletedEvent] = None
    failed_event: Optional[FileFailedEvent] = None
    scan_result: Optional[SingleScanResult] = None


@dataclass
class BatchRunResult:
    """Batch result with stable file ordering and emitted protocol events."""

    summary: BatchSummary
    file_results: List[FileResult]
    events: List[BaseEvent]
    exit_code: int

    def jsonl(self) -> str:
        """Serialize all batch events as newline-delimited JSON."""
        return "\n".join(serialize_event(event) for event in self.events)


def _process_one_file(
    file: DiscoveredFile,
    index: int,
    total: int,
    output_root: Union[str, Path],
    mode: ScanMode,
    pipeline_config: Optional[PipelineConfig],
    pdf_config: Optional[PdfExportConfig],
    manifest: Manifest,
    manifest_store: ManifestStore,
    manifest_lock: Lock,
    event_lock: Lock,
    pipeline_version: str,
    events: List[BaseEvent],
    emit: Optional[EventEmitter],
    batch_period: Optional[BatchPeriod],
    export_mode: ExportMode,
) -> _FileOutcome:
    """Process one file; all exceptions become a file-level failure outcome."""
    started_at = time.perf_counter()
    source_path = Path(file.absolute_path)

    _emit_event(
        FileStartedEvent(
            relative_path=file.relative_path,
            employee_name=file.employee_name,
            index=index,
            total=total,
        ),
        events,
        emit,
        event_lock,
    )

    try:
        source_size, source_mtime_ns = compute_fast_fingerprint(source_path)
        source_hash = file.sha256
        if source_hash is None:
            previous = manifest.get_entry(file.relative_path)
            source_hash = previous.sha256 if previous is not None else None
        if source_hash is None:
            source_hash = compute_sha256(source_path)

        scan_result: SingleScanResult = scan_one(
            source_path,
            mode=mode,
            config=pipeline_config,
        )
        if export_mode == ExportMode.PER_IMAGE:
            target_path = _resolve_output_path(output_root, file.target_relative_pdf)
            export_single_page_pdf(scan_result, target_path, config=pdf_config)

        final_size, final_mtime_ns = compute_fast_fingerprint(source_path)
        if (final_size, final_mtime_ns) != (source_size, source_mtime_ns):
            raise ImageProcessError(
                "Source changed while it was being processed; output will be retried",
                path=str(source_path),
            )

        status = (
            FileProcessingStatus.WARNING
            if scan_result.has_warning
            else FileProcessingStatus.SUCCESS
        )
        previous = manifest.get_entry(file.relative_path)
        page_identity = scan_result.page_identity
        output_relative_path: Optional[str] = file.target_relative_pdf
        output_relative_paths = (
            list(previous.output_relative_paths)
            if previous is not None
            else [file.target_relative_pdf]
        )
        artifact_dependencies = (
            list(previous.artifact_dependencies)
            if previous is not None
            else [file.target_relative_pdf]
        )
        if export_mode == ExportMode.GROUPED:
            output_relative_path = previous.output_relative_path if previous is not None else None
            output_relative_paths = (
                list(previous.output_relative_paths) if previous is not None else []
            )
            artifact_dependencies = (
                list(previous.artifact_dependencies) if previous is not None else []
            )
        manifest_entry = ManifestEntry(
            relative_path=file.relative_path,
            size=source_size,
            mtime_ns=source_mtime_ns,
            sha256=source_hash,
            output_relative_path=output_relative_path,
            output_relative_paths=output_relative_paths,
            status=status,
            processed_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version=pipeline_version,
            period=previous.period if previous is not None else batch_period,
            group_key=previous.group_key if previous is not None else None,
            page_identity=page_identity,
            artifact_dependencies=artifact_dependencies,
            detector_name=scan_result.detector_name,
            detector_mode=scan_result.detector_mode or mode.value,
            detector_model_version=scan_result.detector_model_version,
            detector_model_checksum=scan_result.detector_model_checksum,
            detection_status=("detected" if scan_result.document_detected else "fallback"),
            detection_fallback_used=scan_result.detection_fallback_used,
            detection_quality_summary=dict(scan_result.detection_quality_summary),
        )
        if batch_period is not None:
            assign_entry_context(
                manifest_entry,
                employee_relative_dir=file.employee_name,
                batch_period=batch_period,
            )
        _persist_manifest_entry(manifest, manifest_store, manifest_entry, manifest_lock)

        duration_ms = int(round((time.perf_counter() - started_at) * 1000.0))
        file_result = FileResult(
            relative_path=file.relative_path,
            employee_name=file.employee_name,
            target_relative_pdf=file.target_relative_pdf,
            status=status,
            document_detected=scan_result.document_detected,
            warning=scan_result.warning,
            duration_ms=duration_ms,
        )
        return _FileOutcome(
            file_result=file_result,
            completed_event=FileCompletedEvent(
                relative_path=file.relative_path,
                employee_name=file.employee_name,
                output_relative_path=file.target_relative_pdf,
                document_detected=scan_result.document_detected,
                warning=scan_result.warning,
                duration_ms=duration_ms,
            ),
            scan_result=scan_result,
        )
    except Exception as exc:
        error_code, error_message = _error_details(exc, relative_path=file.relative_path)
        try:
            current_size, current_mtime_ns = compute_fast_fingerprint(source_path)
        except OSError:
            current_size, current_mtime_ns = file.size, file.mtime_ns

        previous = manifest.get_entry(file.relative_path)
        failure_hash = file.sha256 or (previous.sha256 if previous is not None else None)
        failure_entry = _make_failure_entry(
            file_relative_path=file.relative_path,
            file_size=current_size,
            file_mtime_ns=current_mtime_ns,
            file_sha256=failure_hash,
            target_relative_pdf=file.target_relative_pdf,
            manifest=manifest,
            pipeline_version=pipeline_version,
            batch_period=batch_period,
            employee_relative_dir=file.employee_name,
            detector_mode=mode.value,
        )
        _best_effort_persist_failure(manifest, manifest_store, failure_entry, manifest_lock)

        duration_ms = int(round((time.perf_counter() - started_at) * 1000.0))
        return _FileOutcome(
            file_result=FileResult(
                relative_path=file.relative_path,
                employee_name=file.employee_name,
                target_relative_pdf=file.target_relative_pdf,
                status=FileProcessingStatus.FAILED,
                error_code=error_code,
                error_message=error_message,
                duration_ms=duration_ms,
            ),
            failed_event=FileFailedEvent(
                relative_path=file.relative_path,
                employee_name=file.employee_name,
                error_code=error_code,
                message=error_message,
            ),
        )


def _emit_event(
    event: BaseEvent,
    events: List[BaseEvent],
    emitter: Optional[EventEmitter],
    event_lock: Lock,
) -> None:
    """Record an event and optionally forward it to a live JSONL sink."""
    with event_lock:
        events.append(event)
        if emitter is not None:
            emitter(event)


def run_batch(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]] = None,
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    workers: Optional[int] = None,
    *,
    pipeline_config: Optional[PipelineConfig] = None,
    pdf_config: Optional[PdfExportConfig] = None,
    manifest_store: Optional[ManifestStore] = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    emit: Optional[EventEmitter] = None,
    batch_period: Optional[BatchPeriod] = None,
    process_files: Optional[List[DiscoveredFile]] = None,
    manual_orders: Optional[Dict[str, List[str]]] = None,
    export_mode: Union[ExportMode, str] = ExportMode.PER_IMAGE,
    group_plan: Optional[GroupAwareScanPlan] = None,
    skip_groups: Optional[Set[str]] = None,
) -> BatchRunResult:
    """Run the classified process set with resumable per-file isolation.

    Images are opened only inside worker calls, so at most ``workers`` processed
    pixel buffers are live. Every terminal file state is persisted under a lock
    before its completed/failed event is emitted. Per-file failures never abort
    the remaining futures; the returned exit code is ``0`` for a clean batch and
    ``2`` when one or more files failed. Configuration errors raise before any
    worker is submitted and are mapped to CLI exit code ``1`` by the caller.
    """
    if Path(manifest.input_root).resolve() != Path(discovery.input_root).resolve():
        raise InvalidInputRootError(
            discovery.input_root,
            "Manifest input root does not match the discovered input root",
        )

    effective_output_root = Path(output_root or manifest.output_root).resolve()
    if effective_output_root.exists() and not effective_output_root.is_dir():
        raise ValueError(f"Output root is not a directory: {effective_output_root}")

    scan_mode = _normalize_scan_mode(mode)
    effective_pipeline_version = pipeline_version_for_mode(pipeline_version, scan_mode)
    if isinstance(export_mode, ExportMode):
        scan_export_mode = export_mode
    else:
        try:
            scan_export_mode = ExportMode(str(export_mode).replace("-", "_").upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported export mode: {export_mode!r}") from exc
    worker_count = clamp_worker_count(workers)
    store = manifest_store or ManifestStore()
    if process_files is not None:
        candidates = list(process_files)
    elif scan_export_mode == ExportMode.GROUPED:
        candidates = list(discovery.files)
    else:
        candidates = list(discovery.files_to_process)
    effective_skip_groups = set(skip_groups or set())
    if scan_export_mode == ExportMode.GROUPED:
        if group_plan is None:
            raise ValueError("GROUPED execution requires a group-aware scan plan")
        _validate_grouped_run(group_plan, manual_orders, effective_skip_groups)
        process_paths = {
            path.replace("\\", "/")
            for group in group_plan.affected_groups
            if _group_id(group.key) not in effective_skip_groups
            for path in group.process_relative_paths
        }
        selected_files = [
            file for file in candidates if file.relative_path.replace("\\", "/") in process_paths
        ]
    else:
        selected_files = candidates
    total = len(selected_files)
    events: List[BaseEvent] = []
    outcomes: Dict[int, _FileOutcome] = {}
    manifest_lock = Lock()
    event_lock = Lock()
    batch_started_at = time.perf_counter()

    futures: Dict[Future[_FileOutcome], int] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scanner") as executor:
        for index, file in enumerate(selected_files, start=1):
            future = executor.submit(
                _process_one_file,
                file,
                index,
                total,
                effective_output_root,
                scan_mode,
                pipeline_config,
                pdf_config,
                manifest,
                store,
                manifest_lock,
                event_lock,
                effective_pipeline_version,
                events,
                emit,
                batch_period,
                scan_export_mode,
            )
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]
            outcome = future.result()
            outcomes[index] = outcome
            if scan_export_mode == ExportMode.PER_IMAGE and outcome.completed_event is not None:
                _emit_event(outcome.completed_event, events, emit, event_lock)
            elif scan_export_mode == ExportMode.PER_IMAGE and outcome.failed_event is not None:
                _emit_event(outcome.failed_event, events, emit, event_lock)

    if scan_export_mode == ExportMode.GROUPED and group_plan is not None:
        _block_failed_groups(outcomes, selected_files, group_plan, manifest, store)
    ordered_outcomes = [outcomes[index] for index in sorted(outcomes)]
    if scan_export_mode == ExportMode.GROUPED and group_plan is not None:
        grouped_outputs = _export_grouped_results(
            outcomes,
            selected_files,
            group_plan,
            manifest,
            store,
            effective_output_root,
            pdf_config,
            manual_orders,
            effective_skip_groups,
        )
        for outcome in ordered_outcomes:
            relative_path = outcome.file_result.relative_path.replace("\\", "/")
            grouped_output = grouped_outputs.get(relative_path)
            if grouped_output is not None:
                outcome.file_result.target_relative_pdf = grouped_output
                if outcome.completed_event is not None:
                    outcome.completed_event.output_relative_path = grouped_output
            if outcome.completed_event is not None:
                _emit_event(outcome.completed_event, events, emit, event_lock)
            elif outcome.failed_event is not None:
                _emit_event(outcome.failed_event, events, emit, event_lock)
    file_results = [outcome.file_result for outcome in ordered_outcomes]
    success_count = sum(
        1 for result in file_results if result.status == FileProcessingStatus.SUCCESS
    )
    warning_count = sum(
        1 for result in file_results if result.status == FileProcessingStatus.WARNING
    )
    failed_count = sum(1 for result in file_results if result.status == FileProcessingStatus.FAILED)
    summary = BatchSummary(
        total_images=total,
        success=success_count,
        failed=failed_count,
        warning=warning_count,
        skipped=max(0, discovery.image_count - total),
        duration_ms=int(round((time.perf_counter() - batch_started_at) * 1000.0)),
    )
    if scan_export_mode == ExportMode.PER_IMAGE:
        _persist_manual_orders(manifest, store, discovery, batch_period, manual_orders)
    _emit_event(ScanCompletedEvent.from_summary(summary), events, emit, event_lock)

    return BatchRunResult(
        summary=summary,
        file_results=file_results,
        events=events,
        exit_code=2 if failed_count else 0,
    )
