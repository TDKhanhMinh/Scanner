"""Employee folder discovery and deterministic scan planning for Attendance Scanner."""

import re
import stat
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Union

from attendance_scanner.contracts import (
    DiscoveredFile,
    DiscoveryResult,
    FileClassification,
    InvalidInputRootError,
)

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
