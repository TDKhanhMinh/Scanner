"""Subprocess-level golden JSONL tests for Task AS-13."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

SCANNER_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(tmp_path: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    source_path = str(SCANNER_ROOT / "src")
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_python_path
        else source_path + os.pathsep + existing_python_path
    )
    environment["APPDATA"] = str(tmp_path / "appdata")
    return subprocess.run(
        [sys.executable, "-m", "attendance_scanner.cli", *arguments],
        cwd=SCANNER_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def test_plan_subprocess_is_golden_jsonl_and_does_not_create_output(tmp_path: Path):
    input_root = tmp_path / "Cham Công 输入" / "employee images"
    employee = input_root / "Nguyễn Văn A"
    output_root = tmp_path / "PDF output should stay absent"
    employee.mkdir(parents=True)
    Image.new("RGB", (80, 60), color=(180, 180, 180)).save(employee / "ảnh 01.png")

    result = _run_cli(
        tmp_path,
        "plan",
        "--input",
        str(input_root),
        "--output",
        str(output_root),
        "--mode",
        "color",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert not output_root.exists()
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["protocolVersion"] == 1
    assert payload["type"] == "scan_plan"
    assert payload["inputRoot"] == str(input_root.resolve())
    assert payload["outputRoot"] == str(output_root.resolve())
    assert payload["new"] == 1
    assert payload["filesToProcess"] == 1


def test_scan_batch_subprocess_supports_unicode_spaces_and_jsonl_events(tmp_path: Path):
    input_root = tmp_path / "Attendance Input Ω" / "employee images"
    employee = input_root / "Nguyễn Văn A"
    output_root = tmp_path / "Attendance Output Ω"
    employee.mkdir(parents=True)
    Image.new("RGB", (80, 60), color=(180, 180, 180)).save(employee / "ảnh 01.png")

    result = _run_cli(
        tmp_path,
        "scan-batch",
        "--input",
        str(input_root),
        "--output",
        str(output_root),
        "--mode",
        "gray",
        "--workers",
        "1",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    events = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    assert [event["type"] for event in events] == [
        "scan_plan",
        "file_started",
        "file_completed",
        "scan_completed",
    ]
    assert all(event["protocolVersion"] == 1 for event in events)
    assert events[1]["index"] == 1
    assert events[1]["total"] == 1
    assert events[2]["relativePath"] == "Nguyễn Văn A/ảnh 01.png"
    assert events[2]["outputRelativePath"] == "Nguyễn Văn A/ảnh 01.pdf"
    assert events[3]["failed"] == 0
    assert events[3]["success"] + events[3]["warning"] == 1
    assert (output_root / "Nguyễn Văn A" / "ảnh 01.pdf").is_file()


def test_scan_batch_subprocess_invalid_output_is_fatal_without_jsonl_noise(tmp_path: Path):
    input_root = tmp_path / "employees"
    input_root.mkdir()
    invalid_output = tmp_path / "output.txt"
    invalid_output.write_text("not a directory", encoding="utf-8")

    result = _run_cli(
        tmp_path,
        "scan-batch",
        "--input",
        str(input_root),
        "--output",
        str(invalid_output),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    diagnostics = json.loads(result.stderr.strip())
    assert diagnostics["errorCode"] == "OUTPUT_NOT_WRITABLE"
    assert "quyền" in diagnostics["message"]


def test_invalid_cli_arguments_use_fatal_exit_code_1(tmp_path: Path):
    input_root = tmp_path / "employees"
    input_root.mkdir()

    invalid_mode = _run_cli(
        tmp_path,
        "scan-batch",
        "--input",
        str(input_root),
        "--mode",
        "invalid",
    )
    invalid_workers = _run_cli(
        tmp_path,
        "scan-batch",
        "--input",
        str(input_root),
        "--workers",
        "not-a-number",
    )

    assert invalid_mode.returncode == 1
    assert invalid_workers.returncode == 1
    assert invalid_mode.stdout == ""
    assert invalid_workers.stdout == ""
    assert json.loads(invalid_mode.stderr)["errorCode"] == "INVALID_REQUEST"
    assert json.loads(invalid_workers.stderr)["errorCode"] == "INVALID_REQUEST"
