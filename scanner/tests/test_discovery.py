"""Unit tests for employee folder discovery and deterministic scan planning."""

from pathlib import Path

import pytest

from attendance_scanner.cli import main
from attendance_scanner.contracts import (
    FileClassification,
    InvalidInputRootError,
    ScannerErrorCode,
)
from attendance_scanner.discovery import (
    discover_employee_folders,
    natural_sort_key,
    resolve_target_pdf,
)
from attendance_scanner.events import ScanPlanEvent, deserialize_event


def test_natural_sort_key_behavior():
    """Verify natural sort key orders numbers numerically and handles strings."""
    items = ["img10.jpg", "img2.jpg", "img1.jpg", "img20.jpg"]
    sorted_items = sorted(items, key=natural_sort_key)
    assert sorted_items == ["img1.jpg", "img2.jpg", "img10.jpg", "img20.jpg"]

    employees = ["NV 10", "NV 2", "NV 1"]
    assert sorted(employees, key=natural_sort_key) == ["NV 1", "NV 2", "NV 10"]


def test_resolve_target_pdf_rules():
    """Verify target PDF name resolution for unique stems and collision stems."""
    # Unique stem
    assert resolve_target_pdf("2026-09.jpg", is_collision=False) == "2026-09.pdf"
    assert resolve_target_pdf("page1.PNG", is_collision=False) == "page1.pdf"

    # Collision stem: suffixes extension
    assert resolve_target_pdf("2026-09.jpg", is_collision=True) == "2026-09__jpg.pdf"
    assert resolve_target_pdf("2026-09.PNG", is_collision=True) == "2026-09__png.pdf"
    assert resolve_target_pdf("file.preview.webp", is_collision=True) == "file.preview__webp.pdf"


def test_invalid_input_root_non_existent(tmp_path: Path):
    """Verify InvalidInputRootError when root path does not exist."""
    non_existent = tmp_path / "does_not_exist"
    with pytest.raises(InvalidInputRootError) as exc_info:
        discover_employee_folders(non_existent)
    assert exc_info.value.code == ScannerErrorCode.INVALID_INPUT_ROOT
    assert "does not exist" in exc_info.value.message.lower()


def test_invalid_input_root_file_not_dir(tmp_path: Path):
    """Verify InvalidInputRootError when root path is a file instead of a directory."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(InvalidInputRootError) as exc_info:
        discover_employee_folders(file_path)
    assert exc_info.value.code == ScannerErrorCode.INVALID_INPUT_ROOT
    assert "not a directory" in exc_info.value.message.lower()


def test_empty_root_directory(tmp_path: Path):
    """Verify discovery on an empty root directory returns empty inventory."""
    result = discover_employee_folders(tmp_path)
    assert result.employee_count == 0
    assert result.image_count == 0
    assert result.unsupported_count == 0
    assert result.employees == []
    assert result.files == []
    assert result.collisions == []

    plan = result.to_scan_plan()
    assert plan.employees == 0
    assert plan.total_images == 0
    assert plan.files_to_process == 0


def test_empty_employee_folders(tmp_path: Path):
    """Verify employee folders with no images are counted but yield 0 files."""
    (tmp_path / "NguyenVanA").mkdir()
    (tmp_path / "TranVanB").mkdir()

    result = discover_employee_folders(tmp_path)
    assert result.employee_count == 2
    assert result.image_count == 0
    assert result.employees == ["NguyenVanA", "TranVanB"]
    assert result.files == []


def test_unicode_and_vietnamese_names(tmp_path: Path):
    """Verify full support for Unicode Vietnamese diacritics and spaces in paths."""
    emp_dir = tmp_path / "Nguyễn Văn An"
    emp_dir.mkdir()
    img_file = emp_dir / "Bảng chấm công 09.jpg"
    img_file.write_bytes(b"dummy_image_data")

    result = discover_employee_folders(tmp_path)
    assert result.employee_count == 1
    assert result.image_count == 1
    assert result.employees == ["Nguyễn Văn An"]

    file_item = result.files[0]
    assert file_item.employee_name == "Nguyễn Văn An"
    assert file_item.file_name == "Bảng chấm công 09.jpg"
    assert file_item.relative_path == "Nguyễn Văn An/Bảng chấm công 09.jpg"
    assert file_item.target_relative_pdf == "Nguyễn Văn An/Bảng chấm công 09.pdf"
    assert file_item.size == len(b"dummy_image_data")
    assert file_item.classification == FileClassification.NEW


def test_deterministic_natural_sort(tmp_path: Path):
    """Verify deterministic natural sorting of employees and image files."""
    # Create employee folders out of order
    for name in ["NV 10", "NV 2", "NV 1"]:
        emp = tmp_path / name
        emp.mkdir()
        for f in ["10.jpg", "2.jpg", "1.jpg"]:
            (emp / f).write_bytes(b"img")

    result = discover_employee_folders(tmp_path)
    assert result.employees == ["NV 1", "NV 2", "NV 10"]

    # For NV 1, files must be 1.jpg, 2.jpg, 10.jpg
    nv1_files = [f.file_name for f in result.files if f.employee_name == "NV 1"]
    assert nv1_files == ["1.jpg", "2.jpg", "10.jpg"]


def test_collision_policy_same_stem(tmp_path: Path):
    """Verify collision resolution when multiple files share the same stem in an employee folder."""
    emp = tmp_path / "Employee1"
    emp.mkdir()

    # Unique file
    (emp / "2026-08.jpg").write_bytes(b"img_aug")
    # Collision files with same stem '2026-09'
    (emp / "2026-09.jpg").write_bytes(b"img_sep_jpg")
    (emp / "2026-09.png").write_bytes(b"img_sep_png")

    result = discover_employee_folders(tmp_path)
    assert result.image_count == 3
    assert "Employee1/2026-09" in result.collisions

    pdf_targets = {f.file_name: f.target_relative_pdf for f in result.files}
    assert pdf_targets["2026-08.jpg"] == "Employee1/2026-08.pdf"
    assert pdf_targets["2026-09.jpg"] == "Employee1/2026-09__jpg.pdf"
    assert pdf_targets["2026-09.png"] == "Employee1/2026-09__png.pdf"


def test_case_insensitive_extensions_and_unsupported_filtering(tmp_path: Path):
    """Verify supported extensions are case-insensitive and unsupported files are counted."""
    emp = tmp_path / "NV01"
    emp.mkdir()

    # Supported with mixed cases
    (emp / "img1.JPG").write_bytes(b"1")
    (emp / "img2.jpeg").write_bytes(b"2")
    (emp / "img3.Png").write_bytes(b"3")
    (emp / "img4.WebP").write_bytes(b"4")

    # Unsupported files
    (emp / "notes.txt").write_text("notes", encoding="utf-8")
    (emp / "attendance.pdf").write_bytes(b"%PDF-1.4")
    (emp / "timesheet.docx").write_bytes(b"docx")

    # Hidden / system files that should be completely ignored
    (emp / ".DS_Store").write_bytes(b"ds")
    (emp / "Thumbs.db").write_bytes(b"thumbs")
    (emp / "desktop.ini").write_bytes(b"ini")
    (emp / "~$lock.jpg").write_bytes(b"lock")

    result = discover_employee_folders(tmp_path)
    assert result.image_count == 4
    assert result.unsupported_count == 3  # notes.txt, attendance.pdf, timesheet.docx
    assert result.to_scan_plan().unsupported_count == 3


def test_nested_directories_ignored(tmp_path: Path):
    """Verify scanner does not recurse into nested directories inside employee folder."""
    emp = tmp_path / "NV01"
    emp.mkdir()
    (emp / "root_img.jpg").write_bytes(b"root")

    nested = emp / "nested_subfolder"
    nested.mkdir()
    (nested / "nested_img.jpg").write_bytes(b"nested")

    result = discover_employee_folders(tmp_path)
    assert result.employee_count == 1
    assert result.image_count == 1
    assert result.files[0].file_name == "root_img.jpg"


def test_cli_plan_integration(tmp_path: Path, capsys):
    """Verify CLI plan command integrates with discover_employee_folders and outputs JSONL."""
    emp = tmp_path / "NV01"
    emp.mkdir()
    (emp / "img1.jpg").write_bytes(b"data1")
    (emp / "img2.jpg").write_bytes(b"data2")

    ret = main(["plan", "--input", str(tmp_path)])
    assert ret == 0

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line.strip()]
    assert len(lines) == 1

    event = deserialize_event(lines[0])
    assert isinstance(event, ScanPlanEvent)
    assert event.employees == 1
    assert event.total_images == 2
    assert event.new == 2
    assert event.files_to_process == 2
    assert event.collisions == []


def test_windows_hidden_and_system_attributes(tmp_path: Path):
    """Verify that files/folders with Windows Hidden or System attributes are ignored (P1)."""
    import os
    import subprocess

    emp = tmp_path / "NV01"
    emp.mkdir()

    # 1. Normal image
    (emp / "visible.jpg").write_bytes(b"visible")

    # 2. Image with Windows Hidden attribute
    hidden_img = emp / "hidden_by_attr.jpg"
    hidden_img.write_bytes(b"hidden")

    # 3. Hidden employee folder
    hidden_emp = tmp_path / "HiddenEmployee"
    hidden_emp.mkdir()
    (hidden_emp / "img.jpg").write_bytes(b"data")

    if os.name == "nt":
        subprocess.run(["attrib", "+h", str(hidden_img)], check=True)
        subprocess.run(["attrib", "+h", str(hidden_emp)], check=True)

    result = discover_employee_folders(tmp_path)
    # The hidden folder and hidden file must be completely excluded
    assert result.employee_count == 1
    assert result.employees == ["NV01"]
    assert result.image_count == 1
    assert result.files[0].file_name == "visible.jpg"


def test_collision_metadata_deterministic_ordering(tmp_path: Path):
    """Verify collision list is deterministically sorted regardless of creation order (P2)."""
    emp = tmp_path / "NV01"
    emp.mkdir()

    # Create collisions in reverse alphabetical / natural order: z, b, a, 10, 2
    (emp / "z.jpg").write_bytes(b"z1")
    (emp / "z.png").write_bytes(b"z2")
    (emp / "b.jpg").write_bytes(b"b1")
    (emp / "b.png").write_bytes(b"b2")
    (emp / "a.jpg").write_bytes(b"a1")
    (emp / "a.png").write_bytes(b"a2")
    (emp / "10.jpg").write_bytes(b"10_1")
    (emp / "10.png").write_bytes(b"10_2")
    (emp / "2.jpg").write_bytes(b"2_1")
    (emp / "2.png").write_bytes(b"2_2")

    result = discover_employee_folders(tmp_path)
    expected_collisions = [
        "NV01/2",
        "NV01/10",
        "NV01/a",
        "NV01/b",
        "NV01/z",
    ]
    assert result.collisions == expected_collisions
