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
    BatchPeriod,
    CompletenessStatus,
    DocumentGroupKey,
    ExportMode,
    FileProcessingStatus,
    PageIdentity,
    ScannerErrorCode,
    StateError,
)

logger = logging.getLogger(__name__)

# Canonical schema version for manifest files
CURRENT_SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1


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
    period: Optional[BatchPeriod] = None
    group_key: Optional[DocumentGroupKey] = None
    page_identity: PageIdentity = Field(default_factory=PageIdentity)
    artifact_dependencies: List[str] = Field(default_factory=list)
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


class ManifestArtifact(BaseContract):
    """One output artifact and its ordered source dependencies."""

    export_mode: ExportMode = ExportMode.PER_IMAGE
    output_relative_path: str
    source_relative_paths: List[str] = Field(default_factory=list)
    artifact_version: Optional[str] = None
    artifact_hash: Optional[str] = None
    stale: bool = False


class ManifestGroup(BaseContract):
    """Persisted document group state for one employee and period."""

    key: DocumentGroupKey
    source_relative_paths: List[str] = Field(default_factory=list)
    artifact_relative_paths: List[str] = Field(default_factory=list)
    completeness_status: CompletenessStatus = CompletenessStatus.AMBIGUOUS
    review_required: bool = False
    manual_order: List[str] = Field(default_factory=list)
    manual_order_fingerprint: Optional[str] = None
    artifact_stale: bool = False

    @model_validator(mode="after")
    def require_review_for_ambiguous_group(self) -> "ManifestGroup":
        if self.completeness_status == CompletenessStatus.AMBIGUOUS:
            self.review_required = True
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
    groups: Dict[str, ManifestGroup] = Field(default_factory=dict)
    artifacts: Dict[str, ManifestArtifact] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize_files_or_entries(cls, data: Any) -> Any:
        """Allow 'files' as an alias for 'entries' in incoming data."""
        if isinstance(data, dict):
            if "entries" not in data and "files" in data:
                data["entries"] = data["files"]
        return data

    @property
    def files(self) -> Dict[str, ManifestEntry]:
        """Convenience property matching System Design naming."""
        return self.entries

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

    def set_group(self, group: ManifestGroup) -> None:
        """Persist a group in memory; the caller commits it through ManifestStore."""
        key = _group_storage_key(group.key)
        self.groups[key] = group
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def set_artifact(self, artifact: ManifestArtifact) -> None:
        """Persist an artifact in memory; the caller commits it atomically."""
        self.artifacts[artifact.output_relative_path.replace("\\", "/")] = artifact
        self.updated_at = datetime.now(timezone.utc).isoformat()


def _group_storage_key(key: DocumentGroupKey) -> str:
    """Return a stable storage key independent of source filename."""
    employee = key.employee_relative_dir.replace("\\", "/")
    return f"{employee}:{key.year:04d}-{key.month:02d}"


def assign_entry_context(
    entry: ManifestEntry,
    *,
    employee_relative_dir: str,
    batch_period: BatchPeriod,
    page_identity: Optional[PageIdentity] = None,
) -> ManifestEntry:
    """Assign context to a new entry while preserving existing period/group identity."""
    if entry.period is None:
        entry.period = batch_period
    if entry.group_key is None:
        entry.group_key = DocumentGroupKey(
            employee_relative_dir=employee_relative_dir,
            year=entry.period.year,
            month=entry.period.month,
        )
    if page_identity is not None:
        entry.page_identity = page_identity
    return entry


def source_order_fingerprint(entries: List[ManifestEntry]) -> str:
    """Fingerprint source identity/metadata so manual order invalidates on changes."""
    normalized_entries = []
    for entry in sorted(entries, key=lambda item: item.relative_path.replace("\\", "/")):
        relative_path = entry.relative_path.replace("\\", "/")
        normalized_entries.append(
            f"{relative_path}:{entry.size}:{entry.mtime_ns}:{entry.sha256 or ''}"
        )
    normalized = "|".join(normalized_entries)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def set_manual_group_order(
    manifest: Manifest,
    *,
    group_key: DocumentGroupKey,
    ordered_source_paths: List[str],
    source_entries: List[ManifestEntry],
) -> ManifestGroup:
    """Persist an explicit page order and bind it to current source fingerprints."""
    normalized_order = [path.replace("\\", "/") for path in ordered_source_paths]
    expected_paths = {entry.relative_path.replace("\\", "/") for entry in source_entries}
    if (
        len(normalized_order) != len(set(normalized_order))
        or set(normalized_order) != expected_paths
    ):
        raise ValueError("Manual group order must contain each current source exactly once")
    storage_key = _group_storage_key(group_key)
    group = manifest.groups.get(storage_key) or ManifestGroup(key=group_key)
    group.source_relative_paths = normalized_order
    group.manual_order = normalized_order
    group.manual_order_fingerprint = source_order_fingerprint(source_entries)
    group.review_required = False
    group.completeness_status = CompletenessStatus.COMPLETE
    manifest.set_group(group)
    return group


def _migrate_v1_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform a validated-shape v1 manifest into lossless v2 data."""
    if raw_data.get("schemaVersion", raw_data.get("schema_version")) != LEGACY_SCHEMA_VERSION:
        raise ValueError("Only manifest schema v1 can be migrated")

    migrated: Dict[str, Any] = dict(raw_data)
    raw_entries = raw_data.get("entries", raw_data.get("files", {}))
    if not isinstance(raw_entries, dict):
        raise ValueError("Manifest v1 entries must be an object")

    migrated_entries: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    for storage_key, raw_entry in raw_entries.items():
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Manifest v1 entry {storage_key!r} must be an object")
        entry = dict(raw_entry)
        relative_path = entry.get("relativePath", entry.get("relative_path", storage_key))
        output_paths = entry.get("outputRelativePaths", entry.get("output_relative_paths", []))
        if not isinstance(output_paths, list):
            raise ValueError(f"Manifest v1 output paths for {relative_path!r} must be a list")
        output_path = entry.get("outputRelativePath", entry.get("output_relative_path"))
        if output_path and output_path not in output_paths:
            output_paths = [*output_paths, output_path]

        entry["period"] = None
        entry["groupKey"] = None
        entry["pageIdentity"] = {
            "pageType": "UNKNOWN",
            "pageOrder": None,
            "confidence": None,
            "detectionMethod": "legacy_v1",
            "diagnostics": {"migratedFrom": LEGACY_SCHEMA_VERSION},
        }
        entry["artifactDependencies"] = [str(path).replace("\\", "/") for path in output_paths]
        migrated_entries[str(relative_path).replace("\\", "/")] = entry

        for output_relative_path in output_paths:
            normalized_output = str(output_relative_path).replace("\\", "/")
            artifacts.setdefault(
                normalized_output,
                {
                    "exportMode": ExportMode.PER_IMAGE.value,
                    "outputRelativePath": normalized_output,
                    "sourceRelativePaths": [str(relative_path).replace("\\", "/")],
                    "artifactVersion": entry.get("pipelineVersion", entry.get("pipeline_version")),
                    "artifactHash": None,
                },
            )

    migrated["schemaVersion"] = CURRENT_SCHEMA_VERSION
    migrated["entries"] = migrated_entries
    migrated["groups"] = {}
    migrated["artifacts"] = artifacts
    return migrated


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
        subdir_manifest = self.state_dir / root_id / "manifest.json"
        if subdir_manifest.exists():
            return subdir_manifest
        return self.state_dir / f"{root_id}.json"

    def load_manifest(
        self,
        input_root: Union[str, Path],
        output_root: Optional[Union[str, Path]] = None,
        raise_on_corrupt: bool = False,
    ) -> Manifest:
        """Load an existing manifest, or create an initial empty one.

        If the manifest file exists but is corrupted (malformed JSON, invalid schema,
        unsupported version, or root identity mismatch), it is quarantined to
        `{name}.corrupt.{timestamp}` and a fresh manifest is returned (or `StateError`
        is raised if `raise_on_corrupt=True`).

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

        try:
            path_exists = manifest_path.exists()
        except OSError:
            path_exists = False

        if not path_exists:
            now_iso = datetime.now(timezone.utc).isoformat()
            return Manifest(
                schema_version=CURRENT_SCHEMA_VERSION,
                root_id=root_id,
                input_root=canonical_input,
                output_root=canonical_output,
                created_at=now_iso,
                updated_at=now_iso,
                entries={},
                groups={},
                artifacts={},
            )

        try:
            content = manifest_path.read_text(encoding="utf-8")
            raw_data = json.loads(content)

            # Check schema version compatibility
            schema_ver = raw_data.get("schemaVersion", raw_data.get("schema_version"))
            if schema_ver == LEGACY_SCHEMA_VERSION:
                legacy_root_id = raw_data.get("rootId", raw_data.get("root_id"))
                legacy_input_root = raw_data.get("inputRoot", raw_data.get("input_root"))
                if (
                    legacy_root_id != root_id
                    or not isinstance(legacy_input_root, str)
                    or Path(legacy_input_root).resolve() != Path(canonical_input).resolve()
                ):
                    raise ValueError(
                        "Manifest v1 root identity does not match the selected input root"
                    )
                try:
                    migrated = Manifest.model_validate(_migrate_v1_data(raw_data))
                    self.save_manifest(migrated)
                    return migrated
                except StateError:
                    raise
                except Exception as migration_error:
                    raise StateError(
                        ScannerErrorCode.STATE_READ_FAILED,
                        f"Failed to migrate manifest v1 to v2: {migration_error}",
                        path=str(manifest_path),
                    ) from migration_error
            if schema_ver != CURRENT_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported schema version {schema_ver} (expected {CURRENT_SCHEMA_VERSION})"
                )

            manifest = Manifest.model_validate(raw_data)

            # Validate root identity integrity (P1)
            if manifest.root_id != root_id:
                raise ValueError(
                    f"Manifest root_id mismatch: expected '{root_id}', got '{manifest.root_id}'"
                )
            if Path(manifest.input_root).resolve() != Path(canonical_input).resolve():
                raise ValueError(
                    f"Manifest input_root mismatch: expected '{canonical_input}', "
                    f"got '{manifest.input_root}'"
                )

            return manifest

        except StateError:
            raise
        except Exception as err:
            if raise_on_corrupt:
                raise StateError(
                    ScannerErrorCode.STATE_READ_FAILED,
                    f"Failed to load manifest '{manifest_path}': {err}",
                    path=str(manifest_path),
                ) from err

            # Deterministic corrupt state quarantine
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            corrupt_path = manifest_path.parent / f"{manifest_path.name}.corrupt.{timestamp}"
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
                groups={},
                artifacts={},
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
            StateError: If writing, directory creation, or atomic replacement fails.
        """
        target_path = self.get_manifest_path(manifest.input_root)
        target_dir = target_path.parent
        temp_path = target_dir / f"{target_path.name}.tmp.{uuid.uuid4().hex}"

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
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
