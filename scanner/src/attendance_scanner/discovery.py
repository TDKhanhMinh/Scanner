"""Employee folder discovery and deterministic scan planning for Attendance Scanner."""

import re
import stat
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Set, Union

from attendance_scanner.contracts import (
    DiscoveredFile,
    DiscoveryResult,
    FileClassification,
    FileProcessingStatus,
    InvalidInputRootError,
    ScanPlan,
)
from attendance_scanner.fingerprint import compute_sha256
from attendance_scanner.state import Manifest, ManifestEntry

DEFAULT_PIPELINE_VERSION = "0.1.0"

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
) -> FileClassification:
    """Classify one discovered file according to the AS-11 state rules."""
    if entry is None:
        return FileClassification.NEW

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
        return FileClassification.UNCHANGED

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
        return FileClassification.UNCHANGED

    return FileClassification.MODIFIED


def classify_discovered_files(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]] = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
) -> DiscoveryResult:
    """Apply manifest-backed incremental classifications to a discovery result.

    The returned object is the same discovery result with each file annotated by
    ``new``, ``modified``, ``unchanged`` or ``rebuild``. New files are not added
    to the manifest here; their state is committed only after a later batch
    successfully exports them. Existing manifest entries for deleted sources are
    intentionally retained and never cause PDF deletion in the MVP.

    ``pipeline_version`` is counted through ``outdated_pipeline_count`` but does
    not force reprocessing when the source and all recorded outputs are unchanged.
    """
    effective_output_root = output_root or manifest.output_root
    outdated_count = 0

    for file in discovery.files:
        entry = manifest.get_entry(file.relative_path)
        if entry is not None and entry.pipeline_version != pipeline_version:
            outdated_count += 1

        file.classification = _classify_file(
            file=file,
            entry=entry,
            output_root=effective_output_root,
            manifest=manifest,
        )

    discovery.outdated_pipeline_count = outdated_count
    return discovery


def build_incremental_scan_plan(
    discovery: DiscoveryResult,
    manifest: Manifest,
    output_root: Optional[Union[str, Path]] = None,
    pipeline_version: str = DEFAULT_PIPELINE_VERSION,
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
    )
    effective_output_root = output_root or manifest.output_root
    return discovery.to_scan_plan(output_root=str(effective_output_root))
