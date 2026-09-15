"""Incremental flat-folder scanning and isolated PDF export workflow."""

from __future__ import annotations

import os
import shutil
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Set, Union, cast

from attendance_scanner.contracts import (
    BaseContract,
    DocumentOrientation,
    FileProcessingStatus,
    FlatArtifactStatus,
    FlatDiscoveredFile,
    FlatDiscoveryResult,
    FlatExportMode,
    ScanMode,
    ScannerError,
    ScannerErrorCode,
)
from attendance_scanner.detector_modes import detector_name_for_mode, normalize_detector_mode
from attendance_scanner.discovery import DEFAULT_PIPELINE_VERSION, discover_flat_folder
from attendance_scanner.events import (
    BaseEvent,
    FlatFileCompletedEvent,
    FlatScanCompletedEvent,
    FlatScanPlanEvent,
    FlatScanProgressEvent,
)
from attendance_scanner.fingerprint import compute_sha256
from attendance_scanner.flat_state import (
    FlatManifest,
    FlatManifestEntry,
    FlatManifestStore,
    output_fingerprint,
    resolve_external_collision,
)
from attendance_scanner.pdf_export import PdfExportConfig, export_pdf_pages, export_single_page_pdf
from attendance_scanner.pipeline.orchestrator import SingleScanResult, scan_one
from attendance_scanner.segmentation import get_default_segmentation_model_path

FlatEventEmitter = Callable[[BaseEvent], None]
FLAT_PIPELINE_VERSION = f"{DEFAULT_PIPELINE_VERSION}:flat-v2"


@dataclass(frozen=True)
class _FlatWorkItem:
    source: FlatDiscoveredFile
    output_relative_path: str


@dataclass
class _ProcessedFlatFile:
    work: _FlatWorkItem
    result: Optional[SingleScanResult]
    error_code: Optional[ScannerErrorCode] = None
    message: Optional[str] = None


class FlatScanRunResult(BaseContract):
    """Summary returned to CLI/Tauri callers after flat execution."""

    exit_code: int = 0
    total_processed: int = 0
    success: int = 0
    warning: int = 0
    failed: int = 0
    skipped: int = 0
    unsupported_count: int = 0
    artifact_status: FlatArtifactStatus = FlatArtifactStatus.NOT_REQUIRED
    artifact_relative_path: Optional[str] = None
    artifact_error_code: Optional[ScannerErrorCode] = None
    artifact_message: Optional[str] = None
    duration_ms: int = 0


FlatPipelineOrientation = Literal["natural", "landscape", "portrait"]


def _orientation_value(orientation: Union[DocumentOrientation, str]) -> FlatPipelineOrientation:
    value = orientation.value if isinstance(orientation, DocumentOrientation) else str(orientation)
    if value not in {"auto", "landscape", "portrait", "natural"}:
        raise ValueError(f"Unsupported flat-folder orientation: {orientation}")
    return cast(FlatPipelineOrientation, "natural" if value == "auto" else value)


def _pdf_config(result: SingleScanResult) -> PdfExportConfig:
    height, width = result.image.shape[:2]
    return PdfExportConfig(page_orientation="PORTRAIT" if height > width else "LANDSCAPE")


def _safe_error_message(error_code: ScannerErrorCode) -> str:
    return {
        ScannerErrorCode.IMAGE_DECODE_FAILED: "Không thể đọc ảnh đầu vào.",
        ScannerErrorCode.OUTPUT_NOT_WRITABLE: "Không thể ghi file PDF.",
        ScannerErrorCode.PDF_WRITE_FAILED: "Không thể tạo file PDF.",
        ScannerErrorCode.INVALID_REQUEST: "Yêu cầu quét không hợp lệ.",
    }.get(error_code, "Không thể xử lý file ảnh.")


def _expected_detector_provenance(
    detector_mode: str,
) -> tuple[str, Optional[str], Optional[str]]:
    """Resolve stable detector metadata used to decide whether state is reusable."""
    normalized = normalize_detector_mode(detector_mode)
    detector_name = detector_name_for_mode(normalized)
    if detector_name == "v1_cv":
        return detector_name, "opencv-classical", None
    model_path = get_default_segmentation_model_path()
    if model_path is None:
        return detector_name, None, None
    return detector_name, "deeplabv3-mobilenetv3-large-384", compute_sha256(model_path)


def _safe_target_path(output_root: Path, relative_path: str) -> Path:
    root = output_root.resolve()
    target = (root / relative_path.replace("\\", "/")).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Flat output path escapes output root: {relative_path}") from exc
    return target


def _build_work_items(
    discovery: FlatDiscoveryResult,
    manifest: FlatManifest,
    output_root: Path,
    export_mode: FlatExportMode,
) -> List[_FlatWorkItem]:
    if export_mode == FlatExportMode.MERGED:
        merged_name = manifest.merged_output_relative_path or (
            f"{Path(discovery.input_root).name}_merged.pdf"
        )
        if manifest.merged_output_relative_path is None:
            merged_name = resolve_external_collision(
                output_root,
                merged_name,
                owned_names=set(),
            )
        return [
            _FlatWorkItem(source=source, output_relative_path=merged_name)
            for source in discovery.files
        ]

    assigned_names: Set[str] = set()
    items: List[_FlatWorkItem] = []
    for source in discovery.files:
        preferred = source.target_relative_pdf
        existing = manifest.entries.get(source.relative_path)
        if (
            existing is not None
            and existing.export_mode == export_mode
            and existing.output_relative_path
        ):
            preferred = existing.output_relative_path
            if export_mode == FlatExportMode.PER_IMAGE:
                assigned_names.add(preferred)
        target_name = resolve_external_collision(
            output_root,
            preferred,
            owned_names=assigned_names,
        )
        assigned_names.add(target_name)
        items.append(
            _FlatWorkItem(
                source=source,
                output_relative_path=(
                    target_name
                    if export_mode == FlatExportMode.PER_IMAGE
                    else f"{Path(discovery.input_root).name}_merged.pdf"
                ),
            )
        )
    return items


def _entry_matches(
    entry: Optional[FlatManifestEntry],
    item: _FlatWorkItem,
    *,
    output_root: Path,
    scan_mode: ScanMode,
    detector_mode: str,
    detector_name: str,
    detector_model_version: Optional[str],
    detector_model_checksum: Optional[str],
    orientation: str,
    export_mode: FlatExportMode,
) -> bool:
    if entry is None:
        return False
    if (
        entry.content_fingerprint != item.source.content_fingerprint
        or entry.source_relative_path != item.source.relative_path
        or entry.pipeline_version != FLAT_PIPELINE_VERSION
        or entry.scan_mode != scan_mode
        or entry.detector_mode != detector_mode
        or entry.detector_name != detector_name
        or entry.detector_model_version != detector_model_version
        or entry.detector_model_checksum != detector_model_checksum
        or entry.preferred_orientation != orientation
        or entry.export_mode != export_mode
        or entry.output_relative_path != item.output_relative_path
    ):
        return False
    output_path = _safe_target_path(output_root, entry.output_relative_path)
    current_output_hash = output_fingerprint(output_path)
    return current_output_hash is not None and current_output_hash == entry.output_fingerprint


def _manifest_entry(
    *,
    item: _FlatWorkItem,
    result: SingleScanResult,
    output_hash: Optional[str],
    scan_mode: ScanMode,
    detector_mode: str,
    orientation: FlatPipelineOrientation,
    export_mode: FlatExportMode,
) -> FlatManifestEntry:
    """Create a compact, auditable manifest entry from the pipeline result."""
    return FlatManifestEntry(
        source_relative_path=item.source.relative_path,
        content_fingerprint=item.source.content_fingerprint,
        mtime_ns=item.source.mtime_ns,
        size=item.source.size,
        pipeline_version=FLAT_PIPELINE_VERSION,
        scan_mode=scan_mode,
        detector_mode=detector_mode,
        preferred_orientation=orientation,
        export_mode=export_mode,
        output_relative_path=item.output_relative_path,
        output_fingerprint=output_hash,
        detector_name=result.detector_name,
        detector_model_version=result.detector_model_version,
        detector_model_checksum=result.detector_model_checksum,
        detection_status=(
            "detected"
            if result.document_detected
            else "fallback"
            if result.detection_fallback_used
            else "failed"
        ),
        detection_fallback_used=result.detection_fallback_used,
        detection_quality_summary=dict(result.detection_quality_summary),
        detection_reason_codes=list(result.detection_reason_codes),
    )


def _emit_progress(
    emitter: Optional[FlatEventEmitter],
    *,
    item: _FlatWorkItem,
    processed: _ProcessedFlatFile,
    completed_sources: int,
    total_sources: int,
) -> None:
    """Emit source progress without implying that a merged artifact is committed."""
    if emitter is None:
        return
    result = processed.result
    status: Literal["success", "warning", "failed"] = (
        "failed"
        if processed.error_code is not None or result is None
        else "warning"
        if result.has_warning
        else "success"
    )
    emitter(
        FlatScanProgressEvent(
            relative_path=item.source.relative_path,
            completed_sources=completed_sources,
            total_sources=total_sources,
            status=status,
            message=processed.message or (result.warning if result is not None else None),
            duration_ms=(round(result.diagnostics.total_duration_ms) if result is not None else 0),
        )
    )


def _emit_file_event(
    emitter: Optional[FlatEventEmitter],
    *,
    item: _FlatWorkItem,
    processed: _ProcessedFlatFile,
) -> None:
    result = processed.result
    warning = result.warning if result is not None else None
    status = (
        FileProcessingStatus.FAILED
        if processed.error_code is not None
        else FileProcessingStatus.WARNING
        if warning
        else FileProcessingStatus.SUCCESS
    )
    if emitter is not None:
        emitter(
            FlatFileCompletedEvent(
                relative_path=item.source.relative_path,
                output_relative_path=(
                    item.output_relative_path if processed.error_code is None else None
                ),
                status=status,
                document_detected=result.document_detected if result is not None else False,
                warning=warning,
                error_code=processed.error_code,
                message=processed.message,
                duration_ms=(
                    round(result.diagnostics.total_duration_ms) if result is not None else 0
                ),
                detection_preview=result.detection_preview if result is not None else None,
            )
        )


def run_flat_scan(
    input_root: Union[str, Path],
    output_root: Optional[Union[str, Path]] = None,
    *,
    export_mode: Union[FlatExportMode, str] = FlatExportMode.PER_IMAGE,
    mode: Union[ScanMode, str] = ScanMode.GRAY,
    detector_mode: Optional[str] = "ai_enhanced",
    orientation: Union[DocumentOrientation, str] = DocumentOrientation.AUTO,
    workers: int = 3,
    emit: Optional[FlatEventEmitter] = None,
) -> FlatScanRunResult:
    """Plan and execute an incremental flat-folder scan."""
    started_at = time.perf_counter()
    scan_export_mode = (
        export_mode
        if isinstance(export_mode, FlatExportMode)
        else FlatExportMode(str(export_mode).replace("-", "_").upper())
    )
    scan_mode = mode if isinstance(mode, ScanMode) else ScanMode(str(mode).lower())
    normalized_detector_mode = normalize_detector_mode(detector_mode)
    expected_detector_name, expected_model_version, expected_model_checksum = (
        _expected_detector_provenance(normalized_detector_mode)
    )
    preferred_orientation = _orientation_value(orientation)
    discovery = discover_flat_folder(input_root, output_root)
    output_dir = Path(output_root).resolve() if output_root else Path(f"{discovery.input_root}_pdf")
    output_dir.mkdir(parents=True, exist_ok=True)
    store = FlatManifestStore(output_dir)
    manifest = store.load(discovery.input_root)
    previous_entries = dict(manifest.entries)
    previous_merged_output = manifest.merged_output_relative_path
    items = _build_work_items(discovery, manifest, output_dir, scan_export_mode)

    current_paths = {item.source.relative_path for item in items}
    deleted_paths = set(manifest.entries) - current_paths
    for deleted_path in deleted_paths:
        old_entry = manifest.entries.pop(deleted_path)
        if scan_export_mode == FlatExportMode.PER_IMAGE:
            old_output = _safe_target_path(output_dir, old_entry.output_relative_path)
            if old_output.is_file():
                old_output.unlink()

    changed_items = [
        item
        for item in items
        if not _entry_matches(
            manifest.entries.get(item.source.relative_path),
            item,
            output_root=output_dir,
            scan_mode=scan_mode,
            detector_mode=normalized_detector_mode,
            detector_name=expected_detector_name,
            detector_model_version=expected_model_version,
            detector_model_checksum=expected_model_checksum,
            orientation=preferred_orientation,
            export_mode=scan_export_mode,
        )
    ]
    merged_output = _safe_target_path(output_dir, items[0].output_relative_path) if items else None
    if scan_export_mode == FlatExportMode.MERGED:
        merged_needs_rebuild = bool(deleted_paths or changed_items)
        if merged_output is None or not merged_output.is_file():
            merged_needs_rebuild = bool(items)
        elif manifest.merged_artifact_fingerprint != output_fingerprint(merged_output):
            merged_needs_rebuild = bool(items)
        changed_items = items if merged_needs_rebuild else []

    unchanged_count = len(items) - len(changed_items)
    if emit is not None:
        emit(
            FlatScanPlanEvent(
                input_root=discovery.input_root,
                output_root=str(output_dir),
                export_mode=scan_export_mode,
                orientation=(
                    "auto" if preferred_orientation == "natural" else preferred_orientation
                ),
                total_files=len(items),
                files_to_process=len(changed_items),
                unchanged_files=unchanged_count,
                expected_artifacts=(
                    1 if scan_export_mode == FlatExportMode.MERGED and items else len(items)
                ),
                unsupported_count=discovery.unsupported_count,
            )
        )

    if not changed_items:
        if deleted_paths:
            if previous_merged_output and not items:
                old_merged = _safe_target_path(output_dir, previous_merged_output)
                if old_merged.is_file():
                    old_merged.unlink()
                manifest.merged_output_relative_path = None
                manifest.merged_artifact_fingerprint = None
        store.save(manifest)
        duration = round((time.perf_counter() - started_at) * 1000.0)
        artifact_status = (
            FlatArtifactStatus.NOT_REQUIRED if not items else FlatArtifactStatus.COMMITTED
        )
        empty_message = None
        if not items:
            empty_message = (
                f"Không tìm thấy ảnh được hỗ trợ trong thư mục. Đã bỏ qua "
                f"{discovery.unsupported_count} file không hỗ trợ."
                if discovery.unsupported_count
                else "Thư mục không có ảnh để quét."
            )
        completed = FlatScanCompletedEvent(
            input_root=discovery.input_root,
            output_root=str(output_dir),
            export_mode=scan_export_mode,
            total_processed=0,
            success=0,
            warning=0,
            failed=0,
            skipped=unchanged_count,
            unsupported_count=discovery.unsupported_count,
            duration_ms=duration,
            exit_code=0,
            artifact_status=artifact_status,
            artifact_relative_path=(items[0].output_relative_path if items else None),
            artifact_message=empty_message,
        )
        if emit is not None:
            emit(completed)
        return FlatScanRunResult(
            exit_code=0,
            skipped=unchanged_count,
            unsupported_count=discovery.unsupported_count,
            artifact_status=artifact_status,
            artifact_relative_path=(items[0].output_relative_path if items else None),
            artifact_message=empty_message,
            duration_ms=duration,
        )

    def process(item: _FlatWorkItem) -> _ProcessedFlatFile:
        try:
            result = scan_one(
                item.source.absolute_path,
                mode=scan_mode,
                detector_mode=normalized_detector_mode,
                preferred_orientation=preferred_orientation,
            )
            return _ProcessedFlatFile(work=item, result=result)
        except ScannerError as exc:
            return _ProcessedFlatFile(
                work=item,
                result=None,
                error_code=exc.code,
                message=_safe_error_message(exc.code),
            )
        except Exception:
            return _ProcessedFlatFile(
                work=item,
                result=None,
                error_code=ScannerErrorCode.UNEXPECTED_ERROR,
                message="Không thể xử lý file ảnh.",
            )

    processed: List[_ProcessedFlatFile] = []
    worker_count = max(1, min(4, int(workers)))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="flat-scanner") as pool:
        futures: Dict[Future[_ProcessedFlatFile], _FlatWorkItem] = {
            pool.submit(process, item): item for item in changed_items
        }
        for future in as_completed(futures):
            result = future.result()
            processed.append(result)
            if scan_export_mode == FlatExportMode.PER_IMAGE:
                if result.result is not None:
                    target = _safe_target_path(output_dir, result.work.output_relative_path)
                    try:
                        export_single_page_pdf(
                            result.result.image,
                            target,
                            config=_pdf_config(result.result),
                        )
                        manifest.entries[result.work.source.relative_path] = _manifest_entry(
                            item=result.work,
                            result=result.result,
                            output_hash=output_fingerprint(target),
                            scan_mode=scan_mode,
                            detector_mode=normalized_detector_mode,
                            orientation=preferred_orientation,
                            export_mode=scan_export_mode,
                        )
                    except ScannerError as exc:
                        result.error_code = exc.code
                        result.message = _safe_error_message(exc.code)
                _emit_file_event(emit, item=result.work, processed=result)
            else:
                _emit_progress(
                    emit,
                    item=result.work,
                    processed=result,
                    completed_sources=len(processed),
                    total_sources=len(changed_items),
                )

    processed.sort(
        key=lambda value: (
            value.work.source.relative_path.casefold(),
            value.work.source.relative_path,
        )
    )
    failed_results = [
        value for value in processed if value.error_code is not None or value.result is None
    ]
    success_results = [
        value for value in processed if value.result is not None and value.error_code is None
    ]
    artifact_status = FlatArtifactStatus.COMMITTED
    artifact_relative_path: Optional[str] = None
    artifact_error_code: Optional[ScannerErrorCode] = None
    artifact_message: Optional[str] = None

    if scan_export_mode == FlatExportMode.PER_IMAGE:
        manifest.merged_output_relative_path = None
        manifest.merged_artifact_fingerprint = None
        if deleted_paths or processed:
            try:
                store.save(manifest)
            except ScannerError as exc:
                artifact_status = FlatArtifactStatus.NOT_COMMITTED
                artifact_error_code = exc.code
                artifact_message = "Không thể lưu trạng thái quét thư mục."
        if previous_merged_output:
            old_merged = _safe_target_path(output_dir, previous_merged_output)
            if (
                items
                and old_merged.name != Path(items[0].output_relative_path).name
                and old_merged.is_file()
            ):
                try:
                    old_merged.unlink()
                except OSError:
                    pass
    else:
        artifact_relative_path = items[0].output_relative_path
        merged_target = _safe_target_path(output_dir, artifact_relative_path)
        if failed_results:
            artifact_status = FlatArtifactStatus.NOT_COMMITTED
            artifact_error_code = ScannerErrorCode.GROUP_EXPORT_BLOCKED
            artifact_message = "Không thể tạo PDF gộp vì có file nguồn bị lỗi."
            for value in processed:
                if value.error_code is None:
                    value.error_code = artifact_error_code
                    value.message = artifact_message
        else:
            successful_scan_results = [
                value.result for value in success_results if value.result is not None
            ]
            previous_target_exists = merged_target.is_file()
            backup_target = merged_target.with_name(
                f".{merged_target.name}.backup.{uuid.uuid4().hex}"
            )
            staging_target = merged_target.with_name(
                f".{merged_target.name}.stage.{uuid.uuid4().hex}"
            )
            replaced_target = False
            merged_commit_succeeded = False
            try:
                if previous_target_exists:
                    shutil.copy2(merged_target, backup_target)
                export_pdf_pages(
                    [result.image for result in successful_scan_results],
                    staging_target,
                    config=_pdf_config(successful_scan_results[0]),
                )
                os.replace(staging_target, merged_target)
                replaced_target = True
                merged_hash = output_fingerprint(merged_target)
                if merged_hash is None:
                    raise ScannerError(
                        code=ScannerErrorCode.PDF_WRITE_FAILED,
                        message="Merged PDF fingerprint is unavailable",
                    )
                candidate_manifest = manifest.model_copy(deep=True)
                candidate_manifest.merged_artifact_fingerprint = merged_hash
                candidate_manifest.merged_output_relative_path = artifact_relative_path
                candidate_manifest.entries = {
                    value.work.source.relative_path: _manifest_entry(
                        item=value.work,
                        result=value.result,
                        output_hash=merged_hash,
                        scan_mode=scan_mode,
                        detector_mode=normalized_detector_mode,
                        orientation=preferred_orientation,
                        export_mode=scan_export_mode,
                    )
                    for value in success_results
                    if value.result is not None
                }
                store.save(candidate_manifest)
                manifest = candidate_manifest
                artifact_status = FlatArtifactStatus.COMMITTED
                merged_commit_succeeded = True
                for old_entry in previous_entries.values():
                    if old_entry.export_mode != FlatExportMode.MERGED:
                        old_output = _safe_target_path(output_dir, old_entry.output_relative_path)
                        if old_output != merged_target and old_output.is_file():
                            try:
                                old_output.unlink()
                            except OSError:
                                pass
            except ScannerError as exc:
                artifact_status = FlatArtifactStatus.NOT_COMMITTED
                artifact_error_code = exc.code
                artifact_message = "Không thể commit PDF gộp; artifact trước đó được giữ nguyên."
                for value in processed:
                    if value.error_code is None:
                        value.error_code = exc.code
                        value.message = artifact_message
            except Exception:
                artifact_status = FlatArtifactStatus.NOT_COMMITTED
                artifact_error_code = ScannerErrorCode.PDF_WRITE_FAILED
                artifact_message = "Không thể commit PDF gộp; artifact trước đó được giữ nguyên."
                for value in processed:
                    if value.error_code is None:
                        value.error_code = ScannerErrorCode.PDF_WRITE_FAILED
                        value.message = artifact_message
            finally:
                if replaced_target and not merged_commit_succeeded:
                    if previous_target_exists and backup_target.is_file():
                        try:
                            os.replace(backup_target, merged_target)
                        except OSError:
                            pass
                    else:
                        merged_target.unlink(missing_ok=True)
                staging_target.unlink(missing_ok=True)
                backup_target.unlink(missing_ok=True)
        for value in processed:
            _emit_file_event(emit, item=value.work, processed=value)

    failed_results = [
        value for value in processed if value.error_code is not None or value.result is None
    ]
    success_results = [
        value for value in processed if value.result is not None and value.error_code is None
    ]
    successful_scan_results = [
        value.result for value in success_results if value.result is not None
    ]
    success_count = sum(1 for result in successful_scan_results if not result.has_warning)
    warning_count = sum(1 for result in successful_scan_results if result.has_warning)
    failed_count = len(failed_results)
    if (
        artifact_status == FlatArtifactStatus.NOT_COMMITTED
        and scan_export_mode == FlatExportMode.PER_IMAGE
    ):
        artifact_relative_path = None
    effective_exit_code = (
        2 if failed_count or artifact_status == FlatArtifactStatus.NOT_COMMITTED else 0
    )
    duration = round((time.perf_counter() - started_at) * 1000.0)
    completed = FlatScanCompletedEvent(
        input_root=discovery.input_root,
        output_root=str(output_dir),
        export_mode=scan_export_mode,
        total_processed=len(processed),
        success=success_count,
        warning=warning_count,
        failed=failed_count,
        skipped=unchanged_count,
        unsupported_count=discovery.unsupported_count,
        duration_ms=duration,
        exit_code=effective_exit_code,
        artifact_status=artifact_status,
        artifact_relative_path=artifact_relative_path,
        artifact_error_code=artifact_error_code,
        artifact_message=artifact_message,
    )
    if emit is not None:
        emit(completed)
    return FlatScanRunResult(
        exit_code=effective_exit_code,
        total_processed=len(processed),
        success=success_count,
        warning=warning_count,
        failed=failed_count,
        skipped=unchanged_count,
        unsupported_count=discovery.unsupported_count,
        artifact_status=artifact_status,
        artifact_relative_path=artifact_relative_path,
        artifact_error_code=artifact_error_code,
        artifact_message=artifact_message,
        duration_ms=duration,
    )


__all__ = ["FLAT_PIPELINE_VERSION", "FlatScanRunResult", "run_flat_scan"]
