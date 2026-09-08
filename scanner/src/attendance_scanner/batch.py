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
from typing import Callable, Dict, List, Optional, Union

from attendance_scanner.contracts import (
    BatchSummary,
    DiscoveredFile,
    DiscoveryResult,
    FileProcessingStatus,
    FileResult,
    ImageProcessError,
    InvalidInputRootError,
    ScanMode,
    ScannerErrorCode,
)
from attendance_scanner.diagnostics import describe_scanner_error, log_scanner_error
from attendance_scanner.discovery import DEFAULT_PIPELINE_VERSION
from attendance_scanner.events import (
    BaseEvent,
    FileCompletedEvent,
    FileFailedEvent,
    FileStartedEvent,
    ScanCompletedEvent,
    serialize_event,
)
from attendance_scanner.fingerprint import compute_fast_fingerprint, compute_sha256
from attendance_scanner.pdf_export import PdfExportConfig, export_single_page_pdf
from attendance_scanner.pipeline.orchestrator import (
    PipelineConfig,
    SingleScanResult,
    scan_one,
)
from attendance_scanner.state import Manifest, ManifestEntry, ManifestStore

logger = logging.getLogger(__name__)

EventEmitter = Callable[[BaseEvent], None]


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
    try:
        return ScanMode(normalized)
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Unsupported scan mode: {mode!r}") from exc


def _resolve_output_path(output_root: Union[str, Path], relative_path: str) -> Path:
    """Resolve a manifest/discovery output path without allowing root escape."""
    root = Path(output_root).resolve()
    normalized = str(relative_path).replace("\\", "/")
    candidate = (root / Path(normalized)).resolve()
    try:
        candidate.relative_to(root)
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
        stored_pipeline_version = previous.pipeline_version

    return ManifestEntry(
        relative_path=file_relative_path,
        size=file_size,
        mtime_ns=file_mtime_ns,
        sha256=file_sha256,
        output_relative_path=output_relative_path,
        output_relative_paths=output_relative_paths,
        status=FileProcessingStatus.FAILED,
        processed_at=datetime.now(timezone.utc).isoformat(),
        pipeline_version=stored_pipeline_version,
    )


@dataclass
class _FileOutcome:
    """Internal result returned by one isolated worker."""

    file_result: FileResult
    completed_event: Optional[FileCompletedEvent] = None
    failed_event: Optional[FileFailedEvent] = None


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
        manifest_entry = ManifestEntry(
            relative_path=file.relative_path,
            size=source_size,
            mtime_ns=source_mtime_ns,
            sha256=source_hash,
            output_relative_path=file.target_relative_pdf,
            status=status,
            processed_at=datetime.now(timezone.utc).isoformat(),
            pipeline_version=pipeline_version,
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
    worker_count = clamp_worker_count(workers)
    store = manifest_store or ManifestStore()
    process_files = list(discovery.files_to_process)
    total = len(process_files)
    events: List[BaseEvent] = []
    outcomes: Dict[int, _FileOutcome] = {}
    manifest_lock = Lock()
    event_lock = Lock()
    batch_started_at = time.perf_counter()

    futures: Dict[Future[_FileOutcome], int] = {}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="scanner") as executor:
        for index, file in enumerate(process_files, start=1):
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
                pipeline_version,
                events,
                emit,
            )
            futures[future] = index

        for future in as_completed(futures):
            index = futures[future]
            outcome = future.result()
            outcomes[index] = outcome
            if outcome.completed_event is not None:
                _emit_event(outcome.completed_event, events, emit, event_lock)
            elif outcome.failed_event is not None:
                _emit_event(outcome.failed_event, events, emit, event_lock)

    ordered_outcomes = [outcomes[index] for index in sorted(outcomes)]
    file_results = [outcome.file_result for outcome in ordered_outcomes]
    success_count = sum(
        1 for result in file_results if result.status == FileProcessingStatus.SUCCESS
    )
    warning_count = sum(
        1 for result in file_results if result.status == FileProcessingStatus.WARNING
    )
    failed_count = sum(
        1 for result in file_results if result.status == FileProcessingStatus.FAILED
    )
    summary = BatchSummary(
        total_images=total,
        success=success_count,
        failed=failed_count,
        warning=warning_count,
        skipped=max(0, discovery.image_count - total),
        duration_ms=int(round((time.perf_counter() - batch_started_at) * 1000.0)),
    )
    _emit_event(ScanCompletedEvent.from_summary(summary), events, emit, event_lock)

    return BatchRunResult(
        summary=summary,
        file_results=file_results,
        events=events,
        exit_code=2 if failed_count else 0,
    )
