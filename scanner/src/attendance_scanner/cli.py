"""CLI entrypoint for Attendance Scanner sidecar."""

import argparse
import sys
from pathlib import Path
from typing import List, NoReturn, Optional

from attendance_scanner.batch import run_batch
from attendance_scanner.contracts import (
    DiscoveryResult,
    InvalidInputRootError,
    ScanPlan,
    StateError,
)
from attendance_scanner.discovery import (
    build_incremental_scan_plan,
    discover_employee_folders,
)
from attendance_scanner.events import (
    BaseEvent,
    ScanPlanEvent,
    serialize_event,
)
from attendance_scanner.state import Manifest, ManifestStore


class CliArgumentError(ValueError):
    """Raised for CLI argument errors that must map to the scanner fatal exit code."""


class ScannerArgumentParser(argparse.ArgumentParser):
    """Argument parser that lets ``main`` return the scanner exit code contract."""

    def error(self, message: str) -> NoReturn:
        raise CliArgumentError(message)


def emit_jsonl_event(event: BaseEvent) -> None:
    """Write one scanner event to stdout without mixing human logs into JSONL."""
    sys.stdout.write(serialize_event(event) + "\n")
    sys.stdout.flush()


def configure_stdio() -> None:
    """Use UTF-8 streams so Windows paths and names survive JSONL output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _validate_output_root(output_root: Optional[str]) -> Optional[str]:
    """Validate an optional output directory without creating or mutating it."""
    if output_root is None:
        return None
    resolved = Path(output_root).resolve()
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Output root is not a directory: {resolved}")
    return str(resolved)


def _prepare_plan(
    input_root: str,
    output_root: Optional[str],
) -> tuple[DiscoveryResult, Manifest, ManifestStore, str, ScanPlan]:
    """Discover, load state, classify files and persist only metadata-only updates."""
    validated_output_root = _validate_output_root(output_root)
    discovery = discover_employee_folders(input_root)
    store = ManifestStore()
    manifest = store.load_manifest(input_root, output_root=validated_output_root)
    effective_output_root = validated_output_root or manifest.output_root
    _validate_output_root(effective_output_root)
    previous_updated_at = manifest.updated_at
    plan = build_incremental_scan_plan(
        discovery=discovery,
        manifest=manifest,
        output_root=effective_output_root,
    )
    if manifest.updated_at != previous_updated_at:
        store.save_manifest(manifest)
    return discovery, manifest, store, effective_output_root, plan


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser with plan and scan-batch subcommands."""
    parser = ScannerArgumentParser(
        prog="attendance-scanner-sidecar",
        description="Attendance Scanner Desktop engine sidecar CLI.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="attendance-scanner 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="subcommands",
        description="valid commands",
        help="Subcommand to execute",
    )

    # plan subcommand
    plan_parser = subparsers.add_parser(
        "plan",
        help="Analyze input folder and output incremental scan plan (JSONL)",
    )
    plan_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to root employee images folder",
    )
    plan_parser.add_argument(
        "--output",
        "-o",
        required=False,
        type=str,
        default=None,
        help="Path to output PDF folder (defaults to sibling <input>_pdf)",
    )
    plan_parser.add_argument(
        "--mode",
        choices=["gray", "bw", "color"],
        default="gray",
        help="Reserved scan mode argument; planning does not process image pixels",
    )

    # scan-batch subcommand
    scan_parser = subparsers.add_parser(
        "scan-batch",
        help="Execute scan pipeline and export PDFs (streams JSONL)",
    )
    scan_parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to root employee images folder",
    )
    scan_parser.add_argument(
        "--output",
        "-o",
        required=False,
        type=str,
        default=None,
        help="Path to output PDF folder (defaults to sibling <input>_pdf)",
    )
    scan_parser.add_argument(
        "--mode",
        "-m",
        choices=["gray", "bw", "color"],
        default="gray",
        help="Enhancement mode (default: gray)",
    )
    scan_parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=3,
        help="Number of concurrent image processing workers (1-4, default: 3)",
    )

    return parser


def handle_plan(args: argparse.Namespace) -> int:
    """Execute plan subcommand by discovering employee folders and emitting ScanPlanEvent."""
    try:
        _, _, _, _, plan = _prepare_plan(args.input, args.output)
        event = ScanPlanEvent.from_plan(plan)
        emit_jsonl_event(event)
        return 0
    except (InvalidInputRootError, StateError, ValueError) as exc:
        message = exc.message if isinstance(exc, (InvalidInputRootError, StateError)) else str(exc)
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.flush()
        return 1


def handle_scan_batch(args: argparse.Namespace) -> int:
    """Execute scan-batch subcommand by planning and streaming events."""
    try:
        discovery, manifest, store, effective_output_root, plan = _prepare_plan(
            args.input,
            args.output,
        )
        plan_event = ScanPlanEvent.from_plan(plan)
        emit_jsonl_event(plan_event)

        result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=effective_output_root,
            mode=args.mode,
            workers=args.workers,
            manifest_store=store,
            emit=emit_jsonl_event,
        )
        sys.stdout.flush()
        return result.exit_code
    except (InvalidInputRootError, StateError, ValueError) as exc:
        message = exc.message if isinstance(exc, (InvalidInputRootError, StateError)) else str(exc)
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.flush()
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    configure_stdio()
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except CliArgumentError as exc:
        sys.stderr.write(f"Error: {exc}\n")
        sys.stderr.flush()
        return 1

    if not args.command:
        parser.print_help(sys.stderr)
        return 1

    if args.command == "plan":
        return handle_plan(args)
    elif args.command == "scan-batch":
        return handle_scan_batch(args)
    else:
        parser.print_help(sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
