"""Employee folder discovery and deterministic scan planning for Attendance Scanner."""

import re
import stat
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

from pydantic import Field, model_validator

from attendance_scanner.contracts import (
    BaseContract,
    BatchPeriod,
    CompletenessStatus,
    DiscoveredFile,
    DiscoveryResult,
    DocumentGroupKey,
    ExportMode,
    FileClassification,
    FileProcessingStatus,
    InvalidInputRootError,
    PageIdentity,
    PageType,
    ReviewGroup,
    ScanMode,
    ScanPlan,
    SourcePage,
)
from attendance_scanner.fingerprint import compute_sha256
from attendance_scanner.state import (
    Manifest,
    ManifestEntry,
    source_order_fingerprint,
)

# Increment whenever scan output semantics change so existing manifests are rebuilt.
DEFAULT_PIPELINE_VERSION = "0.3.0"


def pipeline_version_for_mode(
    pipeline_version: str,
    mode: Union[ScanMode, str],
) -> str:
    """Return a manifest version that separates output enhancement modes."""
    mode_value = (mode.value if isinstance(mode, ScanMode) else str(mode).strip()).lower()
    if mode_value in ("color_enhanced", "colored"):
        mode_value = ScanMode.COLOR.value
    elif mode_value in ("smart", "smart-document"):
        mode_value = ScanMode.SMART_DOCUMENT.value
    if mode_value == ScanMode.GRAY.value:
        return pipeline_version
    return f"{pipeline_version}:{mode_value}"


@dataclass(frozen=True)
class GroupRebuildPlan:
    """Affected document-group plan layered on top of file classifications."""

    key: DocumentGroupKey
    source_relative_paths: List[str]
    process_relative_paths: List[str]
    artifact_relative_paths: List[str]
    reasons: List[str]
    completeness_status: CompletenessStatus
    review_required: bool


@dataclass(frozen=True)
class GroupAwareScanPlan:
    """Combined file plan and affected-group plan for one export mode/period."""

    export_mode: ExportMode
    period: BatchPeriod
    source_plan: ScanPlan
    affected_groups: List[GroupRebuildPlan]
    process_relative_paths: List[str]
    review_groups: List[ReviewGroup]

    @property
    def affected_group_count(self) -> int:
        return len(self.affected_groups)


class ReprocessPolicy(BaseContract):
    """Explicit detector-version policy for non-destructive incremental planning."""

    pipeline_version: str = Field(min_length=1)
    detector_name: Optional[str] = None
    detector_mode: Optional[str] = None
    detector_model_version: Optional[str] = None
    detector_model_checksum: Optional[str] = None
    opt_in: bool = False

    @model_validator(mode="after")
    def require_model_pair(self) -> "ReprocessPolicy":
        if self.detector_model_checksum and not self.detector_model_version:
            raise ValueError("detector_model_checksum requires detector_model_version")
        return self


def _version_change_reasons(entry: ManifestEntry, policy: ReprocessPolicy) -> List[str]:
    reasons: List[str] = []
    if entry.pipeline_version != policy.pipeline_version:
        reasons.append("PIPELINE_VERSION_CHANGED")
    if (
        policy.detector_name is not None
        and entry.detector_name is not None
        and entry.detector_name != policy.detector_name
    ):
        reasons.append("DETECTOR_NAME_CHANGED")
    if (
        policy.detector_mode is not None
        and entry.detector_mode is not None
        and entry.detector_mode != policy.detector_mode
    ):
        reasons.append("DETECTOR_MODE_CHANGED")
    if policy.detector_model_version is not None:
        if (
            entry.detector_model_version is not None
            and entry.detector_model_version != policy.detector_model_version
        ):
            reasons.append("MODEL_VERSION_CHANGED")
    if (
        policy.detector_model_checksum is not None
        and entry.detector_model_checksum is not None
        and (entry.detector_model_checksum != policy.detector_model_checksum)
    ):
        reasons.append("MODEL_CHECKSUM_CHANGED")
    return reasons


def _classify_version_change(
    file: DiscoveredFile,
    policy: Optional[ReprocessPolicy],
    reasons: List[str],
) -> FileClassification:
    if not reasons or policy is None:
        return FileClassification.UNCHANGED
    if policy.opt_in:
        return FileClassification.REBUILD
    file.reprocess_reason = ",".join(reasons)
    return FileClassification.NEEDS_REPROCESS


def _entry_group_key(
    entry: Optional[ManifestEntry],
    *,
    employee_name: str,
    period: BatchPeriod,
) -> DocumentGroupKey:
    """Resolve existing group identity or the context for a new source."""
    if entry is not None and entry.group_key is not None:
        return entry.group_key
    effective_period = entry.period if entry is not None and entry.period else period
    return DocumentGroupKey(
        employee_relative_dir=employee_name,
        year=effective_period.year,
        month=effective_period.month,
    )


def _group_storage_key(key: DocumentGroupKey) -> str:
    """Return the same stable group storage key used by manifest state."""
    employee = key.employee_relative_dir.replace("\\", "/")
    return f"{employee}:{key.year:04d}-{key.month:02d}"


_COMPLETED_STATUSES = {
    FileProcessingStatus.SUCCESS,
    FileProcessingStatus.WARNING,
}

# Supported image file extensions (case-insensitive)
SUPPORTED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp"}

# Filenames to ignore (system/temp files)
IGNORE_FILENAMES: Set[str] = {
    "thumbs.db",
    "desktop.ini",
    ".ds_store",
}

# Windows attribute flags (safely resolved for cross-platform support)
FILE_ATTRIBUTE_HIDDEN: int = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x02)
FILE_ATTRIBUTE_SYSTEM: int = getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 0x04)


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """Generate a deterministic key for natural sorting (e.g., '2.jpg' before '10.jpg').

    Splits string by digit sequences and lowercases text for case-insensitive comparison.
    """
    return [int(text) if text.isdigit() else text.casefold() for text in re.split(r"(\d+)", s)]


def is_hidden_or_system(entry: Union[str, Path]) -> bool:
    """Check if a file or folder is considered hidden or system metadata.

    Checks:
    1. Name-based conventions (starts with '.' or '~$', or matches known ignore names).
    2. OS file attributes (Windows FILE_ATTRIBUTE_HIDDEN and FILE_ATTRIBUTE_SYSTEM).
    """
    path = Path(entry) if not isinstance(entry, Path) else entry
    name = path.name
    lowered = name.lower()
    if name.startswith(".") or name.startswith("~$") or lowered in IGNORE_FILENAMES:
        return True

    # Windows file attributes check
    try:
        attrs = getattr(path.stat(), "st_file_attributes", 0)
        if attrs & (FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM):
            return True
    except (OSError, ValueError):
        pass

    return False


def resolve_target_pdf(file_name: str, is_collision: bool) -> str:
    """Resolve target PDF file name according to collision policy.

    - Unique stem: '09.jpg' -> '09.pdf'
    - Collision group member: '09.jpg' -> '09__jpg.pdf', '09.png' -> '09__png.pdf'
    """
    path = Path(file_name)
    stem = path.stem
    if is_collision:
        ext = path.suffix.lstrip(".").lower()
        return f"{stem}__{ext}.pdf"
    return f"{stem}.pdf"


def discover_employee_folders(input_root: Union[str, Path]) -> DiscoveryResult:
    """Scan input root, discover direct employee folders and image files deterministically.

    Raises:
        InvalidInputRootError: If input_root does not exist or is not a directory.
    """
    raw_path = Path(input_root)
    try:
        resolved_root = raw_path.resolve()
    except Exception as exc:
        raise InvalidInputRootError(str(input_root), f"Cannot resolve path: {exc}") from exc

    if not resolved_root.exists():
        raise InvalidInputRootError(str(input_root), "Path does not exist")
    if not resolved_root.is_dir():
        raise InvalidInputRootError(str(input_root), "Path is not a directory")

    # Discover direct child directories (each represents an employee)
    # Ignore hidden directories by name and OS attributes
    employee_dirs: List[Path] = []
    for entry in resolved_root.iterdir():
        if entry.is_dir() and not is_hidden_or_system(entry):
            employee_dirs.append(entry)

    # Deterministic natural sort of employee directories
    employee_dirs.sort(key=lambda p: (natural_sort_key(p.name), p.name))

    discovered_files: List[DiscoveredFile] = []
    all_collisions: List[str] = []
    unsupported_count = 0

    for emp_dir in employee_dirs:
        emp_name = emp_dir.name

        # Gather direct child files; ignore nested directories and hidden/system files
        valid_images: List[Path] = []
        for child in emp_dir.iterdir():
            if is_hidden_or_system(child):
                continue
            if child.is_dir():
                # Nested directories are explicitly ignored in MVP
                continue
            if child.is_file():
                ext = child.suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    valid_images.append(child)
                else:
                    unsupported_count += 1

        # Detect same-stem collisions within this employee folder
        # Group by case-insensitive stem
        stem_groups = defaultdict(list)
        for img in valid_images:
            stem_groups[img.stem.lower()].append(img)

        # Sort collision stem keys deterministically
        collision_stems: Set[str] = set()
        for stem_key in sorted(stem_groups.keys(), key=lambda s: (natural_sort_key(s), s)):
            group = stem_groups[stem_key]
            if len(group) > 1:
                collision_stems.add(stem_key)
                all_collisions.append(f"{emp_name}/{stem_key}")

        # Deterministic natural sort of valid images
        valid_images.sort(key=lambda p: (natural_sort_key(p.name), p.name))

        for img in valid_images:
            is_collision = img.stem.lower() in collision_stems
            target_pdf_name = resolve_target_pdf(img.name, is_collision)
            rel_pdf = f"{emp_name}/{target_pdf_name}"
            rel_path = f"{emp_name}/{img.name}"

            stat_res = img.stat()
            discovered_files.append(
                DiscoveredFile(
                    employee_name=emp_name,
                    file_name=img.name,
                    relative_path=rel_path,
                    absolute_path=str(img),
                    size=stat_res.st_size,
                    mtime_ns=stat_res.st_mtime_ns,
                    classification=FileClassification.NEW,
                    target_relative_pdf=rel_pdf,
                )
            )

    # Ensure overall collisions list is sorted deterministically
    all_collisions.sort(key=lambda c: (natural_sort_key(c), c))

    return DiscoveryResult(
        input_root=str(resolved_root),
        employees=[d.name for d in employee_dirs],
        files=discovered_files,
        employee_count=len(employee_dirs),
        image_count=len(discovered_files),
        unsupported_count=unsupported_count,
        collisions=all_collisions,
    )


def _output_paths_exist(output_root: Union[str, Path], entry: ManifestEntry) -> bool:
    """Return whether every artifact recorded by an entry still exists as a file.

    Manifest paths are treated as relative paths rooted at ``output_root``. Any
    absolute or escaping path is considered missing so a corrupt/stale manifest
    cannot make the planner accept an output outside the selected directory.
    """
    recorded_paths = list(entry.output_relative_paths)
    if not recorded_paths and entry.output_relative_path:
        recorded_paths = [entry.output_relative_path]
    if not recorded_paths:
        return False

    root = Path(output_root).resolve()
    for relative_path in recorded_paths:
        normalized = str(relative_path).replace("\\", "/")
        candidate = (root / Path(normalized)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
    return True


def _classify_file(
    file: DiscoveredFile,
    entry: Optional[ManifestEntry],
    output_root: Union[str, Path],
    manifest: Manifest,
    pipeline_version: str,
    required_output_mode: Optional[ExportMode] = None,
    reprocess_policy: Optional[ReprocessPolicy] = None,
) -> FileClassification:
    """Classify one discovered file according to the AS-11 state rules."""
    if entry is None:
        return FileClassification.NEW

    version_reasons = (
        _version_change_reasons(entry, reprocess_policy) if reprocess_policy is not None else []
    )
    if reprocess_policy is None and entry.pipeline_version != pipeline_version:
        return FileClassification.REBUILD

    if required_output_mode == ExportMode.PER_IMAGE:
        expected_output = file.target_relative_pdf.replace("\\", "/")
        recorded_outputs = {path.replace("\\", "/") for path in entry.output_relative_paths}
        if entry.output_relative_path:
            recorded_outputs.add(entry.output_relative_path.replace("\\", "/"))
        if expected_output not in recorded_outputs or not _output_paths_exist(
            output_root,
            ManifestEntry(
                relative_path=entry.relative_path,
                size=entry.size,
                mtime_ns=entry.mtime_ns,
                sha256=entry.sha256,
                output_relative_path=expected_output,
                status=entry.status,
                processed_at=entry.processed_at,
                pipeline_version=entry.pipeline_version,
            ),
        ):
            return FileClassification.REBUILD

    output_exists = _output_paths_exist(output_root, entry)
    if not output_exists:
        return FileClassification.REBUILD

    # Warning means the file was exported successfully with a non-fatal scan
    # warning. All other non-completed statuses are retryable after an
    # interrupted/failed batch. If an artifact exists, modified is the most
    # useful retry classification; if it did not exist, the rule above returned
    # rebuild.
    if entry.status not in _COMPLETED_STATUSES:
        return FileClassification.MODIFIED

    if entry.size == file.size and entry.mtime_ns == file.mtime_ns:
        return _classify_version_change(file, reprocess_policy, version_reasons)

    # Metadata changed: hash the source before deciding whether it needs work.
    current_hash = compute_sha256(file.absolute_path)
    file.sha256 = current_hash
    if entry.sha256 is not None and current_hash == entry.sha256:
        # Keep the successful artifact and advance only the source metadata. The
        # caller may persist this in-memory update atomically after planning.
        entry.size = file.size
        entry.mtime_ns = file.mtime_ns
        entry.sha256 = current_hash
        manifest.set_entry(entry)
        return _classify_version_change(file, reprocess_policy, version_reasons)

    return FileClassification.MODIFIED


def classify_discovered_files(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]] = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    required_output_mode: Optional[ExportMode] = None,
    reprocess_policy: Optional[ReprocessPolicy] = None,
) -> DiscoveryResult:
    """Apply manifest-backed incremental classifications to a discovery result.

    The returned object is the same discovery result with each file annotated by
    ``new``, ``modified``, ``unchanged`` or ``rebuild``. New files are not added
    to the manifest here; their state is committed only after a later batch
    successfully exports them. Existing manifest entries for deleted sources are
    intentionally retained and never cause PDF deletion in the MVP.

    A pipeline-version mismatch forces a rebuild even when the source and recorded
    outputs are unchanged, because the output semantics may have changed.
    """
    effective_output_root = output_root or manifest.output_root
    outdated_count = 0
    discovery.needs_reprocess_count = 0
    discovery.reprocess_reasons = {}
    effective_pipeline_version = (
        reprocess_policy.pipeline_version if reprocess_policy is not None else pipeline_version
    )

    for file in discovery.files:
        entry = manifest.get_entry(file.relative_path)
        if entry is not None and entry.pipeline_version != effective_pipeline_version:
            outdated_count += 1

        file.classification = _classify_file(
            file=file,
            entry=entry,
            output_root=effective_output_root,
            manifest=manifest,
            pipeline_version=effective_pipeline_version,
            required_output_mode=required_output_mode,
            reprocess_policy=reprocess_policy,
        )
        if file.classification == FileClassification.NEEDS_REPROCESS:
            discovery.needs_reprocess_count += 1
            for reason in (file.reprocess_reason or "NEEDS_REPROCESS").split(","):
                discovery.reprocess_reasons[reason] = discovery.reprocess_reasons.get(reason, 0) + 1

    discovery.outdated_pipeline_count = outdated_count
    return discovery


def build_incremental_scan_plan(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]] = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    required_output_mode: Optional[ExportMode] = None,
    reprocess_policy: Optional[ReprocessPolicy] = None,
) -> ScanPlan:
    """Classify a discovery inventory and return its aggregate scan plan.

    The exact process set remains available as ``discovery.files_to_process``;
    the returned ``ScanPlan`` carries the counters used by JSONL/UI consumers.
    """
    classify_discovered_files(
        discovery=discovery,
        manifest=manifest,
        output_root=output_root,
        pipeline_version=pipeline_version,
        required_output_mode=required_output_mode,
        reprocess_policy=reprocess_policy,
    )
    effective_output_root = output_root or manifest.output_root
    return discovery.to_scan_plan(output_root=str(effective_output_root))


def _group_artifacts_exist(output_root: Union[str, Path], artifact_paths: List[str]) -> bool:
    """Check persisted artifact paths without treating an empty list as valid."""
    if not artifact_paths:
        return False
    root = Path(output_root).resolve()
    for relative_path in artifact_paths:
        candidate = (root / Path(relative_path.replace("\\", "/"))).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False
        if not candidate.is_file():
            return False
    return True


def build_group_aware_scan_plan(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]],
    period: BatchPeriod,
    *,
    export_mode: ExportMode = ExportMode.GROUPED,
    scan_mode: Union[ScanMode, str] = ScanMode.GRAY,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
    reprocess_policy: Optional[ReprocessPolicy] = None,
) -> GroupAwareScanPlan:
    """Build file and affected-group plans for PER_IMAGE or GROUPED export.

    Grouped mode expands an affected group's process set to all currently
    discovered siblings because the MVP has no reusable processed-page cache.
    The expansion is scoped to the same employee/period and never changes other
    groups. Removed sources remain stale/incomplete and are never auto-deleted.
    """
    source_plan = build_incremental_scan_plan(
        discovery=discovery,
        manifest=manifest,
        output_root=output_root,
        pipeline_version=pipeline_version_for_mode(pipeline_version, scan_mode),
        reprocess_policy=reprocess_policy,
    )
    effective_output_root = output_root or manifest.output_root
    discovered_paths = {file.relative_path.replace("\\", "/") for file in discovery.files}
    grouped_files: Dict[str, List[DiscoveredFile]] = {}
    group_reasons: Dict[str, List[str]] = {}
    group_keys: Dict[str, DocumentGroupKey] = {}

    for file in discovery.files:
        normalized_path = file.relative_path.replace("\\", "/")
        entry = manifest.get_entry(normalized_path)
        key = _entry_group_key(entry, employee_name=file.employee_name, period=period)
        storage_key = _group_storage_key(key)
        group_keys[storage_key] = key
        grouped_files.setdefault(storage_key, []).append(file)
        if file.classification in {
            FileClassification.NEW,
            FileClassification.MODIFIED,
            FileClassification.REBUILD,
        }:
            group_reasons.setdefault(storage_key, []).append(f"source_{file.classification.value}")

    affected_groups: List[GroupRebuildPlan] = []
    all_process_paths: List[str] = []
    for storage_key, files in grouped_files.items():
        key = group_keys[storage_key]
        persisted_group = manifest.groups.get(storage_key)
        artifact_paths = list(persisted_group.artifact_relative_paths) if persisted_group else []
        reasons = list(dict.fromkeys(group_reasons.get(storage_key, [])))
        missing_sources = (
            [
                path
                for path in persisted_group.source_relative_paths
                if path.replace("\\", "/") not in discovered_paths
            ]
            if persisted_group
            else []
        )
        if missing_sources:
            reasons.append("source_removed")
        entries = [manifest.get_entry(file.relative_path) for file in files]
        if export_mode == ExportMode.GROUPED:
            current_entries = [
                entry.model_copy(
                    update={
                        "size": file.size,
                        "mtime_ns": file.mtime_ns,
                        "sha256": file.sha256 or entry.sha256,
                    }
                )
                for file, entry in zip(files, entries, strict=True)
                if entry is not None
            ]
            current_paths = {entry.relative_path.replace("\\", "/") for entry in current_entries}
            persisted_order = list(persisted_group.manual_order) if persisted_group else []
            manual_order_valid = bool(
                persisted_group
                and persisted_order
                and persisted_group.manual_order_fingerprint
                and len(current_entries) == len(entries)
                and len(persisted_order) == len(set(persisted_order))
                and set(persisted_order) == current_paths
                and source_order_fingerprint(current_entries)
                == persisted_group.manual_order_fingerprint
            )
            if persisted_order and not manual_order_valid:
                reasons.append("manual_order_invalidated")
            if not manual_order_valid:
                if any(
                    entry is None or entry.page_identity.page_type == PageType.UNKNOWN
                    for entry in entries
                ):
                    reasons.append("unknown_page_identity")
                known_orders = [
                    entry.page_identity.page_order
                    for entry in entries
                    if entry is not None and entry.page_identity.page_order is not None
                ]
                if len(known_orders) != len(set(known_orders)):
                    reasons.append("duplicate_page_order")
                if len(entries) < 2:
                    reasons.append("missing_expected_page")
        if persisted_group and not _group_artifacts_exist(effective_output_root, artifact_paths):
            reasons.append("missing_grouped_output")
        if persisted_group and persisted_group.artifact_stale:
            reasons.append("grouped_artifact_stale")
        if not reasons:
            continue

        source_paths = [file.relative_path.replace("\\", "/") for file in files]
        process_paths = source_paths
        if export_mode == ExportMode.PER_IMAGE:
            process_paths = [
                file.relative_path.replace("\\", "/")
                for file in files
                if file.classification
                in {
                    FileClassification.NEW,
                    FileClassification.MODIFIED,
                    FileClassification.REBUILD,
                }
            ]
        status = (
            persisted_group.completeness_status if persisted_group else CompletenessStatus.AMBIGUOUS
        )
        if missing_sources or "missing_expected_page" in reasons:
            status = CompletenessStatus.INCOMPLETE
        review_required = persisted_group.review_required if persisted_group else False
        review_required = review_required or bool(missing_sources)
        review_required = review_required or any(
            reason
            in {
                "unknown_page_identity",
                "duplicate_page_order",
                "manual_order_invalidated",
                "missing_expected_page",
                "grouped_artifact_stale",
            }
            for reason in reasons
        )
        if review_required and status == CompletenessStatus.COMPLETE:
            status = CompletenessStatus.AMBIGUOUS
        plan = GroupRebuildPlan(
            key=key,
            source_relative_paths=source_paths,
            process_relative_paths=process_paths,
            artifact_relative_paths=artifact_paths,
            reasons=list(dict.fromkeys(reasons)),
            completeness_status=status,
            review_required=review_required,
        )
        affected_groups.append(plan)
        for path in process_paths:
            if path not in all_process_paths:
                all_process_paths.append(path)

    for storage_key, persisted_group in manifest.groups.items():
        missing_sources = [
            path
            for path in persisted_group.source_relative_paths
            if path.replace("\\", "/") not in discovered_paths
        ]
        if not missing_sources or storage_key in grouped_files:
            continue
        affected_groups.append(
            GroupRebuildPlan(
                key=persisted_group.key,
                source_relative_paths=list(persisted_group.source_relative_paths),
                process_relative_paths=[],
                artifact_relative_paths=list(persisted_group.artifact_relative_paths),
                reasons=["source_removed"],
                completeness_status=CompletenessStatus.INCOMPLETE,
                review_required=True,
            )
        )

    if export_mode == ExportMode.PER_IMAGE:
        process_paths = [
            file.relative_path.replace("\\", "/")
            for file in discovery.files
            if file.classification
            in {
                FileClassification.NEW,
                FileClassification.MODIFIED,
                FileClassification.REBUILD,
            }
        ]
    else:
        process_paths = all_process_paths

    review_groups: List[ReviewGroup] = []
    for group in affected_groups:
        if not group.review_required and group.completeness_status != CompletenessStatus.AMBIGUOUS:
            continue
        source_pages: List[SourcePage] = []
        for relative_path in group.source_relative_paths:
            entry = manifest.get_entry(relative_path)
            source_pages.append(
                SourcePage(
                    source_relative_path=relative_path,
                    identity=entry.page_identity if entry is not None else PageIdentity(),
                )
            )
        persisted_group = manifest.groups.get(_group_storage_key(group.key))
        review_groups.append(
            ReviewGroup(
                key=group.key,
                source_pages=source_pages,
                reasons=list(group.reasons),
                completeness_status=group.completeness_status,
                review_required=True,
                manual_order=(list(persisted_group.manual_order) if persisted_group else []),
            )
        )

    return GroupAwareScanPlan(
        export_mode=export_mode,
        period=period,
        source_plan=source_plan,
        affected_groups=affected_groups,
        process_relative_paths=process_paths,
        review_groups=review_groups,
    )


build_affected_document_groups = build_group_aware_scan_plan
