"""Documentation contract checks for the operator and developer handoff guides."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _read_doc(name: str) -> str:
    return (REPOSITORY_ROOT / "docs" / name).read_text(encoding="utf-8")


def test_operator_guide_covers_monthly_workflow_and_failure_recovery():
    content = _read_doc("OPERATOR_GUIDE.md")

    for heading in (
        "## 1. Expected folders",
        "## 2. Monthly workflow",
        "## 3. Scan modes",
        "## 4. Plan and output semantics",
        "## 5. Result meanings",
        "## 6. Troubleshooting",
        "## 7. AppData and uninstall safety",
    ):
        assert heading in content
    for error_code in (
        "INVALID_INPUT_ROOT",
        "OUTPUT_NOT_WRITABLE",
        "PDF_WRITE_FAILED",
        "IMAGE_DECODE_FAILED",
        "DOCUMENT_NOT_DETECTED",
    ):
        assert error_code in content
    assert "Scan New Files" in content
    assert "%APPDATA%" in content


def test_developer_handoff_points_to_live_entrypoints_commands_and_boundaries():
    content = _read_doc("DEVELOPER_HANDOFF.md")

    for source_path in (
        "scanner/src/attendance_scanner/cli.py",
        "scanner/src/attendance_scanner/batch.py",
        "scanner/src/attendance_scanner/state.py",
        "apps/desktop/src-tauri/src/lib.rs",
        "apps/desktop/src/App.tsx",
        "apps/desktop/src/lib/scannerBridge.ts",
    ):
        assert (REPOSITORY_ROOT / source_path).is_file()
        assert source_path in content
    for command in (
        "pytest.exe tests -q",
        "ruff.exe check src tests",
        "mypy.exe src",
        "npm --prefix apps/desktop test -- --run",
        "build-windows.ps1",
    ):
        assert command in content
    for contract in (
        "JSONL protocol",
        "Manifest and incremental semantics",
        "Release and privacy rules",
        "Agent workflow",
        "MVP",
    ):
        assert contract in content
    assert "flowchart LR" in content
