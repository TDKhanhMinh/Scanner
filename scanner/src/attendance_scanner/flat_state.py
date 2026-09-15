"""Isolated, atomic manifest state for the flat-folder workflow."""

from __future__ import annotations

import hashlib
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
from attendance_scanner.state import get_default_state_dir

FLAT_MANIFEST_SCHEMA_VERSION = 2
FLAT_MANIFEST_FILENAME = ".flat_scanner_manifest.json"


def flat_manifest_id(
    input_root: Union[str, Path],
    output_root: Union[str, Path],
) -> str:
    """Return a stable opaque id for one flat input/output pairing."""
    normalized = "\n".join(
        os.path.normcase(os.path.normpath(str(Path(value).resolve())))
        for value in (input_root, output_root)
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


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
    """Load/save flat state outside the user-facing PDF output directory."""

    def __init__(
        self,
        output_root: Union[str, Path],
        *,
        state_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.output_root = Path(output_root).resolve()
        self.state_root = (
            Path(state_root).resolve()
            if state_root is not None
            else get_default_state_dir().resolve()
        )
        self.manifest_path = self.output_root / FLAT_MANIFEST_FILENAME

    def get_manifest_path(self, input_root: Union[str, Path]) -> Path:
        """Return the internal state path for one input/output pairing."""
        return (
            self.state_root
            / "flat"
            / flat_manifest_id(input_root, self.output_root)
            / FLAT_MANIFEST_FILENAME
        )

    @property
    def legacy_manifest_path(self) -> Path:
        """Return the pre-v4 manifest path eligible for one-time migration."""
        return self.output_root / FLAT_MANIFEST_FILENAME

    def empty_manifest(self, input_root: Union[str, Path]) -> FlatManifest:
        return FlatManifest(
            input_root=str(Path(input_root).resolve()),
            output_root=str(self.output_root),
        )

    def load(self, input_root: Union[str, Path]) -> FlatManifest:
        """Load valid state or return empty state so the next run rebuilds safely."""
        state_path = self.get_manifest_path(input_root)
        self.manifest_path = state_path
        source_path = state_path
        migrating_legacy = False
        if not source_path.is_file() and self.legacy_manifest_path.is_file():
            source_path = self.legacy_manifest_path
            migrating_legacy = True
        if not source_path.is_file():
            return self.empty_manifest(input_root)
        try:
            manifest = FlatManifest.model_validate_json(source_path.read_text(encoding="utf-8"))
            if (
                Path(manifest.input_root).resolve() != Path(input_root).resolve()
                or Path(manifest.output_root).resolve() != self.output_root
            ):
                raise ValueError("flat manifest root identity does not match selected paths")
            if migrating_legacy:
                try:
                    self.save(manifest)
                    self.legacy_manifest_path.unlink(missing_ok=True)
                except Exception:
                    # Keep the legacy copy if state migration cannot be committed.
                    pass
            return manifest
        except Exception:
            # Keep the corrupt file for diagnosis and force a clean rebuild.
            return self.empty_manifest(input_root)

    def save(self, manifest: FlatManifest) -> Path:
        """Persist state with flush/fsync and same-directory atomic replacement."""
        manifest_path = self.get_manifest_path(manifest.input_root)
        self.manifest_path = manifest_path
        temp_path = manifest_path.parent / f"{FLAT_MANIFEST_FILENAME}.tmp.{uuid.uuid4().hex}"
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            serialized = manifest.model_dump_json(by_alias=True, indent=2)
            with open(temp_path, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, manifest_path)
            return manifest_path
        except Exception as exc:
            temp_path.unlink(missing_ok=True)
            raise StateError(
                code=ScannerErrorCode.STATE_WRITE_FAILED,
                message=f"Failed to persist flat manifest: {exc}",
                path=str(manifest_path),
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
    "flat_manifest_id",
    "FlatManifest",
    "FlatManifestEntry",
    "FlatManifestStore",
    "output_fingerprint",
    "resolve_external_collision",
]
