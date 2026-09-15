"""Isolated, atomic manifest state for the flat-folder workflow."""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Union

from pydantic import Field

from attendance_scanner.contracts import (
    BaseContract,
    DetectionFailureReason,
    FlatExportMode,
    ScanMode,
    ScannerErrorCode,
    StateError,
)
from attendance_scanner.fingerprint import compute_sha256

FLAT_MANIFEST_SCHEMA_VERSION = 2
FLAT_MANIFEST_FILENAME = ".flat_scanner_manifest.json"


class FlatManifestEntry(BaseContract):
    """Persisted source/output provenance for one flat-folder image."""

    source_relative_path: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=64, max_length=64)
    mtime_ns: int = Field(ge=0)
    size: int = Field(ge=0)
    pipeline_version: str = Field(min_length=1)
    scan_mode: ScanMode
    detector_mode: Optional[str] = None
    preferred_orientation: Literal["natural", "landscape", "portrait"]
    export_mode: FlatExportMode
    output_relative_path: str = Field(min_length=1)
    output_fingerprint: Optional[str] = Field(default=None, min_length=64, max_length=64)
    detector_name: Optional[str] = None
    detector_model_version: Optional[str] = None
    detector_model_checksum: Optional[str] = None
    detection_status: Optional[Literal["detected", "fallback", "failed", "not_run"]] = None
    detection_fallback_used: Optional[bool] = None
    detection_quality_summary: Dict[str, Union[str, int, float, bool, None]] = Field(
        default_factory=dict
    )
    detection_reason_codes: List[DetectionFailureReason] = Field(default_factory=list)


class FlatManifest(BaseContract):
    """Versioned flat-folder state, independent from Attendance manifests."""

    schema_version: Literal[2] = 2
    input_root: str = Field(min_length=1)
    output_root: str = Field(min_length=1)
    entries: Dict[str, FlatManifestEntry] = Field(default_factory=dict)
    merged_artifact_fingerprint: Optional[str] = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    merged_output_relative_path: Optional[str] = None
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class FlatManifestStore:
    """Load and atomically save a manifest beneath the selected output root."""

    def __init__(self, output_root: Union[str, Path]) -> None:
        self.output_root = Path(output_root).resolve()
        self.manifest_path = self.output_root / FLAT_MANIFEST_FILENAME

    def empty_manifest(self, input_root: Union[str, Path]) -> FlatManifest:
        return FlatManifest(
            input_root=str(Path(input_root).resolve()),
            output_root=str(self.output_root),
        )

    def load(self, input_root: Union[str, Path]) -> FlatManifest:
        """Load valid state or return empty state so the next run rebuilds safely."""
        if not self.manifest_path.is_file():
            return self.empty_manifest(input_root)
        try:
            manifest = FlatManifest.model_validate_json(
                self.manifest_path.read_text(encoding="utf-8")
            )
            if (
                Path(manifest.input_root).resolve() != Path(input_root).resolve()
                or Path(manifest.output_root).resolve() != self.output_root
            ):
                raise ValueError("flat manifest root identity does not match selected paths")
            return manifest
        except Exception:
            # Keep the corrupt file for diagnosis and force a clean rebuild.
            return self.empty_manifest(input_root)

    def save(self, manifest: FlatManifest) -> Path:
        """Persist state with flush/fsync and same-directory atomic replacement."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_root / f"{FLAT_MANIFEST_FILENAME}.tmp.{uuid.uuid4().hex}"
        try:
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            serialized = manifest.model_dump_json(by_alias=True, indent=2)
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.manifest_path)
            return self.manifest_path
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise StateError(
                code=ScannerErrorCode.STATE_WRITE_FAILED,
                message=f"Failed to persist flat manifest: {exc}",
                path=str(self.manifest_path),
            ) from exc


def output_fingerprint(path: Union[str, Path]) -> Optional[str]:
    """Return the current output hash, or None when the artifact is missing."""
    target = Path(path)
    if not target.is_file():
        return None
    return compute_sha256(target)


def resolve_external_collision(
    output_root: Union[str, Path],
    preferred_name: str,
    *,
    owned_names: Set[str],
) -> str:
    """Choose a deterministic suffix when an untracked output already exists."""
    root = Path(output_root)
    candidate = preferred_name
    stem = Path(preferred_name).stem
    suffix = Path(preferred_name).suffix
    counter = 1
    while candidate not in owned_names and (root / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


__all__ = [
    "FLAT_MANIFEST_FILENAME",
    "FLAT_MANIFEST_SCHEMA_VERSION",
    "FlatManifest",
    "FlatManifestEntry",
    "FlatManifestStore",
    "output_fingerprint",
    "resolve_external_collision",
]
