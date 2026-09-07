"""Manifest state store and deterministic root identity management."""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import Field, model_validator

from .contracts import (
    BaseContract,
    FileProcessingStatus,
    ScannerErrorCode,
    StateError,
)

logger = logging.getLogger(__name__)

# Canonical schema version for manifest files
CURRENT_SCHEMA_VERSION = 1


def compute_root_id(canonical_input_root: Union[str, Path]) -> str:
    """Compute a deterministic, collision-resistant 24-char root identity string.

    Calculates the first 24 hexadecimal characters of the SHA-256 digest of
    the normalized, canonical absolute path.

    Args:
        canonical_input_root: Root path of the employee input directory.

    Returns:
        24-character hexadecimal root identifier.
    """
    resolved = Path(canonical_input_root).resolve()
    # Normalize separators and case (Windows-safe)
    normalized = os.path.normcase(os.path.normpath(str(resolved))).replace("\\", "/")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:24]


def get_default_state_dir() -> Path:
    """Resolve default application state directory in system AppData.

    - Windows: %APPDATA%/attendance-scanner/state
    - macOS: ~/Library/Application Support/attendance-scanner/state
    - Linux/Other: ~/.attendance-scanner/state (or $XDG_DATA_HOME)

    Returns:
        Path to state storage directory.
    """
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "attendance-scanner" / "state"

    xdg_data = os.environ.get("XDG_DATA_HOME")
    if xdg_data:
        return Path(xdg_data) / "attendance-scanner" / "state"

    return Path.home() / ".attendance-scanner" / "state"


class ManifestEntry(BaseContract):
    """Manifest state entry for a single discovered or processed source file."""

    relative_path: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None
    output_relative_path: Optional[str] = None
    output_relative_paths: List[str] = Field(default_factory=list)
    status: FileProcessingStatus = FileProcessingStatus.SUCCESS
    processed_at: str
    pipeline_version: str = "0.1.0"
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _sync_output_paths(self) -> "ManifestEntry":
        """Maintain bidirectional sync between single and multi-artifact paths."""
        if (
            self.output_relative_path
            and self.output_relative_path not in self.output_relative_paths
        ):
            self.output_relative_paths.append(self.output_relative_path)
        elif not self.output_relative_path and self.output_relative_paths:
            self.output_relative_path = self.output_relative_paths[0]
        return self


class Manifest(BaseContract):
    """Top-level versioned manifest tracking incremental processing state."""

    schema_version: int = CURRENT_SCHEMA_VERSION
    root_id: str
    input_root: str
    output_root: str
    created_at: str
    updated_at: str
    entries: Dict[str, ManifestEntry] = Field(default_factory=dict)

    def get_entry(self, relative_path: str) -> Optional[ManifestEntry]:
        """Look up an entry by relative path with separator normalization."""
        key = relative_path.replace("\\", "/")
        return self.entries.get(key)

    def set_entry(self, entry: ManifestEntry) -> None:
        """Add or update an entry, updating manifest timestamp."""
        key = entry.relative_path.replace("\\", "/")
        self.entries[key] = entry
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def remove_entry(self, relative_path: str) -> Optional[ManifestEntry]:
        """Remove an entry by relative path, updating manifest timestamp."""
        key = relative_path.replace("\\", "/")
        removed = self.entries.pop(key, None)
        if removed is not None:
            self.updated_at = datetime.now(timezone.utc).isoformat()
        return removed


class ManifestStore:
    """Persistent storage manager for versioned manifests in AppData."""

    def __init__(self, state_dir: Optional[Union[str, Path]] = None) -> None:
        """Initialize manifest store with an optional injected state directory.

        Args:
            state_dir: Custom state storage directory. If None, uses default AppData.
        """
        if state_dir is not None:
            self.state_dir = Path(state_dir).resolve()
        else:
            self.state_dir = get_default_state_dir().resolve()

    def get_manifest_path(self, input_root: Union[str, Path]) -> Path:
        """Get the filesystem path for the manifest corresponding to an input root."""
        root_id = compute_root_id(input_root)
        return self.state_dir / f"{root_id}.json"

    def load_manifest(
        self,
        input_root: Union[str, Path],
        output_root: Optional[Union[str, Path]] = None,
        raise_on_corrupt: bool = False,
    ) -> Manifest:
        """Load an existing manifest, or create an initial empty one.

        If the manifest file exists but is corrupted (malformed JSON, invalid schema,
        or unsupported version), it is quarantined to `{root_id}.json.corrupt.{timestamp}`
        and a fresh manifest is returned (or `StateError` is raised if `raise_on_corrupt=True`).

        Args:
            input_root: Root directory of employee input images.
            output_root: Expected root directory of scanned output PDFs.
            raise_on_corrupt: If True, raises StateError on corrupt state instead of quarantining.

        Returns:
            Manifest instance.

        Raises:
            StateError: If reading fails and `raise_on_corrupt` is True.
        """
        canonical_input = str(Path(input_root).resolve())
        canonical_output = (
            str(Path(output_root).resolve()) if output_root else f"{canonical_input}_pdf"
        )
        root_id = compute_root_id(input_root)
        manifest_path = self.get_manifest_path(input_root)

        if not manifest_path.exists():
            now_iso = datetime.now(timezone.utc).isoformat()
            return Manifest(
                schema_version=CURRENT_SCHEMA_VERSION,
                root_id=root_id,
                input_root=canonical_input,
                output_root=canonical_output,
                created_at=now_iso,
                updated_at=now_iso,
                entries={},
            )

        try:
            content = manifest_path.read_text(encoding="utf-8")
            raw_data = json.loads(content)

            # Check schema version compatibility
            schema_ver = raw_data.get("schemaVersion", raw_data.get("schema_version"))
            if schema_ver != CURRENT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema version {schema_ver} (expected {CURRENT_SCHEMA_VERSION})"
                )

            manifest = Manifest.model_validate(raw_data)
            return manifest

        except Exception as err:
            if raise_on_corrupt:
                raise StateError(
                    ScannerErrorCode.STATE_READ_FAILED,
                    f"Failed to load manifest '{manifest_path}': {err}",
                    path=str(manifest_path),
                ) from err

            # Deterministic corrupt state quarantine
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            corrupt_path = self.state_dir / f"{root_id}.json.corrupt.{timestamp}"
            try:
                os.replace(manifest_path, corrupt_path)
                logger.warning(
                    "Quarantined corrupted manifest from '%s' to '%s': %s",
                    manifest_path,
                    corrupt_path,
                    err,
                )
            except OSError as q_err:
                logger.error(
                    "Failed to quarantine corrupted manifest '%s': %s",
                    manifest_path,
                    q_err,
                )

            # Return fresh empty manifest
            now_iso = datetime.now(timezone.utc).isoformat()
            return Manifest(
                schema_version=CURRENT_SCHEMA_VERSION,
                root_id=root_id,
                input_root=canonical_input,
                output_root=canonical_output,
                created_at=now_iso,
                updated_at=now_iso,
                entries={},
            )

    def save_manifest(self, manifest: Manifest) -> Path:
        """Atomically persist manifest to disk in the state directory.

        Writes data to a temporary file in the same directory, flushes, syncs,
        and atomically renames to the final target file via `os.replace`.

        Args:
            manifest: Manifest instance to persist.

        Returns:
            Path to the saved manifest file.

        Raises:
            StateError: If writing or atomic replacement fails.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.state_dir / f"{manifest.root_id}.json"
        temp_path = self.state_dir / f"{manifest.root_id}.json.tmp.{uuid.uuid4().hex}"

        try:
            manifest.updated_at = datetime.now(timezone.utc).isoformat()
            serialized = manifest.model_dump_json(by_alias=True, indent=2)

            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_path, target_path)
            return target_path

        except Exception as err:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise StateError(
                ScannerErrorCode.STATE_WRITE_FAILED,
                f"Failed to persist manifest to '{target_path}': {err}",
                path=str(target_path),
            ) from err
