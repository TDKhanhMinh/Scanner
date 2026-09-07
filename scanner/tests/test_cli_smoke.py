"""Smoke tests for attendance scanner CLI."""

import json

import pytest

from attendance_scanner.cli import create_parser, main


def test_parser_creation():
    """Verify parser instantiates and has subcommands."""
    parser = create_parser()
    assert parser.prog == "attendance-scanner-sidecar"


def test_parser_help(capsys):
    """Test parser --help exits cleanly."""
    parser = create_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "plan" in captured.out
    assert "scan-batch" in captured.out


def test_main_plan_subcommand(capsys):
    """Test plan subcommand output."""
    test_args = ["plan", "--input", "test_input"]
    exit_code = main(test_args)
    assert exit_code == 0

    captured = capsys.readouterr()
    output_line = captured.out.strip().split("\n")[-1]
    data = json.loads(output_line)
    assert data["type"] == "scan_plan"
    assert data["inputRoot"] == "test_input"


def test_main_scan_batch_subcommand(capsys):
    """Test scan-batch subcommand output stream."""
    test_args = ["scan-batch", "--input", "test_input", "--mode", "gray", "--workers", "2"]
    exit_code = main(test_args)
    assert exit_code == 0

    captured = capsys.readouterr()
    lines = [line for line in captured.out.strip().split("\n") if line]
    assert len(lines) >= 2
    events = [json.loads(line) for line in lines]
    assert events[0]["type"] == "scan_plan"
    assert events[-1]["type"] == "scan_completed"


def test_main_no_args(capsys):
    """Test invoking main without arguments returns code 1."""
    exit_code = main([])
    assert exit_code == 1
