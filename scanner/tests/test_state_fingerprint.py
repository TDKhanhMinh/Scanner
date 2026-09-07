"""Unit tests for file fingerprinting and manifest state store (Task AS-10)."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from attendance_scanner.contracts import (
    FileProcessingStatus,
    ScannerErrorCode,
    StateError,
)
from attendance_scanner.fingerprint import (
    FileFingerprint,
    compute_fast_fingerprint,
    compute_sha256,
)
from attendance_scanner.state import (
    CURRENT_SCHEMA_VERSION,
    Manifest,
    ManifestEntry,
    ManifestStore,
    compute_root_id,
    get_default_state_dir,
)

# ============================================================================
# 1. Root ID Stability and Formatting
# ============================================================================


def test_compute_root_id_format(tmp_path: Path) -> None:
    """Root ID must be exactly 24 hexadecimal characters."""
    root = tmp_path / "employees"
    root.mkdir()
    root_id = compute_root_id(root)

    assert isinstance(root_id, str)
    assert len(root_id) == 24
    int(root_id, 16)  # Must be valid hex


def test_compute_root_id_canonical_stability(tmp_path: Path) -> None:
    """Root ID must be identical regardless of slashes, Windows case,
    or trailing separators.
    """
    root = tmp_path / "MyCompany" / "DeptA"
    root.mkdir(parents=True)

    id_direct = compute_root_id(root)
    id_trailing_slash = compute_root_id(str(root) + "/")
    id_backslash = compute_root_id(str(root).replace("/", "\\"))
    id_case_variation = compute_root_id(str(root).lower())

    assert id_direct == id_trailing_slash
    assert id_direct == id_backslash
    assert id_direct == id_case_variation


def test_compute_root_id_distinct_for_different_paths(tmp_path: Path) -> None:
    """Different directories must produce different root IDs."""
    dir_a = tmp_path / "dept_a"
    dir_b = tmp_path / "dept_b"
    dir_a.mkdir()
    dir_b.mkdir()

    assert compute_root_id(dir_a) != compute_root_id(dir_b)


# ============================================================================
# 2. Fast Fingerprint & Streaming SHA-256
# ============================================================================


def test_compute_fast_fingerprint(tmp_path: Path) -> None:
    """Fast fingerprint must return file size and positive mtime_ns."""
    sample = tmp_path / "sample.txt"
    payload = b"Hello, Attendance Scanner Fingerprint!"
    sample.write_bytes(payload)

    size, mtime_ns = compute_fast_fingerprint(sample)
    assert size == len(payload)
    assert mtime_ns > 0


def test_compute_fast_fingerprint_missing_file(tmp_path: Path) -> None:
    """Stat on non-existent file must raise FileNotFoundError."""
    missing = tmp_path / "missing.txt"
    with pytest.raises(FileNotFoundError):
        compute_fast_fingerprint(missing)


def test_compute_sha256_streaming(tmp_path: Path) -> None:
    """Streaming SHA-256 must match standard hashlib digest."""
    sample = tmp_path / "binary.bin"
    content = b"x" * 150000  # Multi-chunk file (> 64KB)
    sample.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = compute_sha256(sample, chunk_size=32768)
    assert actual == expected


def test_file_fingerprint_model_and_lazy_hashing(tmp_path: Path) -> None:
    """FileFingerprint model stores metadata and lazily hashes on demand."""
    sample = tmp_path / "employee_card.jpg"
    payload = b"Fake JPEG Header and Content"
    sample.write_bytes(payload)

    # From path without hash
    fp = FileFingerprint.from_path(sample, compute_hash=False)
    assert fp.path == str(sample)
    assert fp.size == len(payload)
    assert fp.sha256 is None
    assert fp.matches(len(payload), fp.mtime_ns)
    assert not fp.matches(len(payload) + 1, fp.mtime_ns)
    assert not fp.matches(len(payload), fp.mtime_ns + 100)

    # Lazy hash
    expected_hash = hashlib.sha256(payload).hexdigest()
    digest = fp.ensure_sha256()
    assert digest == expected_hash
    assert fp.sha256 == expected_hash

    # From path with hash
    fp_hashed = FileFingerprint.from_path(sample, compute_hash=True)
    assert fp_hashed.sha256 == expected_hash


# ============================================================================
# 3. Manifest Entry and Manifest Model
# ============================================================================


def test_manifest_entry_output_paths_sync() -> None:
    """ManifestEntry synchronizes single output_relative_path and output_relative_paths."""
    entry1 = ManifestEntry(
        relative_path="NV001/img1.jpg",
        size=1024,
        mtime_ns=123456789,
        output_relative_path="NV001/img1.pdf",
        processed_at="2026-09-07T12:00:00Z",
    )
    assert entry1.output_relative_paths == ["NV001/img1.pdf"]

    entry2 = ManifestEntry(
        relative_path="NV002/doc.jpg",
        size=2048,
        mtime_ns=987654321,
        output_relative_paths=["NV002/doc_page1.pdf", "NV002/doc_page2.pdf"],
        processed_at="2026-09-07T12:00:00Z",
    )
    assert entry2.output_relative_path == "NV002/doc_page1.pdf"


def test_manifest_crud_operations() -> None:
    """Manifest set_entry, get_entry, and remove_entry normalize separators."""
    manifest = Manifest(
        schema_version=1,
        root_id="abcdef1234567890abcdef12",
        input_root="/data/input",
        output_root="/data/output",
        created_at="2026-09-07T00:00:00Z",
        updated_at="2026-09-07T00:00:00Z",
    )

    entry = ManifestEntry(
        relative_path="EmpA\\img1.jpg",
        size=5000,
        mtime_ns=1000,
        status=FileProcessingStatus.SUCCESS,
        processed_at="2026-09-07T01:00:00Z",
    )
    manifest.set_entry(entry)

    # Retrieval via forward or backward slash
    assert manifest.get_entry("EmpA/img1.jpg") is not None
    assert manifest.get_entry("EmpA\\img1.jpg") is not None
    assert manifest.get_entry("NonExistent/img.jpg") is None

    # Removal
    removed = manifest.remove_entry("EmpA/img1.jpg")
    assert removed is not None
    assert manifest.get_entry("EmpA/img1.jpg") is None


# ============================================================================
# 4. ManifestStore Persistence & Round-Trip
# ============================================================================


def test_manifest_store_first_time_empty(tmp_path: Path) -> None:
    """First-time loading of a new directory returns an empty manifest."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)

    input_root = tmp_path / "employees"
    input_root.mkdir()
    manifest = store.load_manifest(input_root)

    assert manifest.schema_version == CURRENT_SCHEMA_VERSION
    assert manifest.root_id == compute_root_id(input_root)
    assert manifest.input_root == str(input_root.resolve())
    assert manifest.output_root == f"{input_root.resolve()}_pdf"
    assert len(manifest.entries) == 0


def test_manifest_store_save_and_reload_lossless(tmp_path: Path) -> None:
    """Manifest store must serialize and deserialize all fields without loss."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "employees"
    output_root = tmp_path / "output_pdf"
    input_root.mkdir()
    output_root.mkdir()

    manifest = store.load_manifest(input_root, output_root=output_root)
    entry1 = ManifestEntry(
        relative_path="Nguyen Van A/01_T8.jpg",
        size=245890,
        mtime_ns=1725698400000000,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        output_relative_path="Nguyen Van A/01_T8.pdf",
        status=FileProcessingStatus.SUCCESS,
        processed_at="2026-09-07T08:30:00Z",
        pipeline_version="0.1.0",
        extra={"rotation": 90, "enhancement": "bw"},
    )
    entry2 = ManifestEntry(
        relative_path="Tran Thi B/02_T8.png",
        size=120034,
        mtime_ns=1725698500000000,
        output_relative_paths=["Tran Thi B/02_T8.pdf"],
        status=FileProcessingStatus.WARNING,
        processed_at="2026-09-07T08:31:00Z",
    )
    manifest.set_entry(entry1)
    manifest.set_entry(entry2)

    saved_path = store.save_manifest(manifest)
    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    # Reload from disk
    loaded = store.load_manifest(input_root, output_root=output_root)
    assert loaded.schema_version == CURRENT_SCHEMA_VERSION
    assert loaded.root_id == manifest.root_id
    assert len(loaded.entries) == 2

    loaded_e1 = loaded.get_entry("Nguyen Van A/01_T8.jpg")
    assert loaded_e1 is not None
    assert loaded_e1.size == 245890
    assert loaded_e1.mtime_ns == 1725698400000000
    assert loaded_e1.sha256 == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert loaded_e1.output_relative_path == "Nguyen Van A/01_T8.pdf"
    assert loaded_e1.status == FileProcessingStatus.SUCCESS
    assert loaded_e1.extra == {"rotation": 90, "enhancement": "bw"}

    loaded_e2 = loaded.get_entry("Tran Thi B/02_T8.png")
    assert loaded_e2 is not None
    assert loaded_e2.status == FileProcessingStatus.WARNING
    assert loaded_e2.output_relative_paths == ["Tran Thi B/02_T8.pdf"]


# ============================================================================
# 5. Atomic Persistence Verification
# ============================================================================


def test_atomic_persistence_no_orphan_temp_files(tmp_path: Path) -> None:
    """State saving must not leave temporary files behind upon success."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "input"
    input_root.mkdir()

    manifest = store.load_manifest(input_root)
    store.save_manifest(manifest)

    files = list(state_dir.iterdir())
    assert len(files) == 1
    assert files[0].name.endswith(".json")
    assert not any(".tmp" in f.name for f in files)


def test_atomic_persistence_failure_cleanup(tmp_path: Path) -> None:
    """If os.replace fails, temp file is cleaned up and StateError is raised."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "input"
    input_root.mkdir()
    manifest = store.load_manifest(input_root)

    with patch("os.replace", side_effect=OSError("Disk full simulation")):
        with pytest.raises(StateError) as exc_info:
            store.save_manifest(manifest)

        assert exc_info.value.code == ScannerErrorCode.STATE_WRITE_FAILED
        assert "Disk full simulation" in str(exc_info.value)

    # Ensure no leftover temp files
    remaining = list(state_dir.glob("*.tmp*"))
    assert len(remaining) == 0


# ============================================================================
# 6. Corrupt State Handling & Deterministic Quarantine
# ============================================================================


def test_corrupt_manifest_quarantined_by_default(tmp_path: Path) -> None:
    """Corrupted JSON must be quarantined to .corrupt.<timestamp> and a fresh state returned."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "input"
    input_root.mkdir()

    manifest_path = store.get_manifest_path(input_root)
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{ incomplete json: true,", encoding="utf-8")

    # Load should safely recover
    manifest = store.load_manifest(input_root)
    assert len(manifest.entries) == 0

    # Original file is gone (quarantined)
    assert not manifest_path.exists()

    # Quarantine file exists with original corrupt contents
    corrupt_files = list(state_dir.glob("*.corrupt.*"))
    assert len(corrupt_files) == 1
    assert "{ incomplete json: true," in corrupt_files[0].read_text(encoding="utf-8")


def test_corrupt_manifest_raises_when_requested(tmp_path: Path) -> None:
    """When raise_on_corrupt=True, corrupted state raises StateError."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "input"
    input_root.mkdir()

    manifest_path = store.get_manifest_path(input_root)
    state_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("NOT_JSON", encoding="utf-8")

    with pytest.raises(StateError) as exc_info:
        store.load_manifest(input_root, raise_on_corrupt=True)

    assert exc_info.value.code == ScannerErrorCode.STATE_READ_FAILED


def test_schema_version_mismatch_quarantined(tmp_path: Path) -> None:
    """Incompatible future schema version is treated as corrupt state and quarantined."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "input"
    input_root.mkdir()

    manifest_path = store.get_manifest_path(input_root)
    state_dir.mkdir(parents=True, exist_ok=True)
    future_data = {
        "schemaVersion": 99,
        "rootId": compute_root_id(input_root),
        "inputRoot": str(input_root),
        "outputRoot": str(input_root) + "_pdf",
        "createdAt": "2030-01-01T00:00:00Z",
        "updatedAt": "2030-01-01T00:00:00Z",
        "entries": {},
    }
    manifest_path.write_text(json.dumps(future_data), encoding="utf-8")

    # Load should quarantine future schema and return clean state
    manifest = store.load_manifest(input_root)
    assert manifest.schema_version == CURRENT_SCHEMA_VERSION

    corrupt_files = list(state_dir.glob("*.corrupt.*"))
    assert len(corrupt_files) == 1


# ============================================================================
# 7. Unicode & Vietnamese Paths
# ============================================================================


def test_vietnamese_accented_paths_roundtrip(tmp_path: Path) -> None:
    """Manifest store handles Vietnamese and Unicode characters across all path fields."""
    state_dir = tmp_path / "state"
    store = ManifestStore(state_dir=state_dir)
    input_root = tmp_path / "Thư mục nhân sự 2026"
    output_root = tmp_path / "Xuất PDF chấm công"
    input_root.mkdir()
    output_root.mkdir()

    manifest = store.load_manifest(input_root, output_root=output_root)
    viet_entry = ManifestEntry(
        relative_path="Nguyễn Văn Đức/Phiếu chấm công T08-2026.jpg",
        size=512000,
        mtime_ns=1725700000000000,
        output_relative_path="Nguyễn Văn Đức/Phiếu chấm công T08-2026.pdf",
        status=FileProcessingStatus.SUCCESS,
        processed_at="2026-09-07T10:00:00Z",
        extra={"phòng_ban": "Kỹ thuật", "ghi_chú": "Đã quét sắc nét"},
    )
    manifest.set_entry(viet_entry)
    store.save_manifest(manifest)

    # Reload and inspect
    reloaded = store.load_manifest(input_root, output_root=output_root)
    found = reloaded.get_entry("Nguyễn Văn Đức/Phiếu chấm công T08-2026.jpg")
    assert found is not None
    assert found.output_relative_path == "Nguyễn Văn Đức/Phiếu chấm công T08-2026.pdf"
    assert found.extra["phòng_ban"] == "Kỹ thuật"
    assert found.extra["ghi_chú"] == "Đã quét sắc nét"


# ============================================================================
# 8. Default State Directory Resolution
# ============================================================================


def test_get_default_state_dir_resolution() -> None:
    """Default state directory properly resolves without crashing."""
    d = get_default_state_dir()
    assert isinstance(d, Path)
    assert "attendance-scanner" in str(d)
    assert "state" in str(d)
