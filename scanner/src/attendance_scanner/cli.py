"""CLI entrypoint for Attendance Scanner sidecar."""

import argparse
import sys
from typing import List, Optional

from attendance_scanner.batch import run_batch
from attendance_scanner.contracts import InvalidInputRootError, StateError
from attendance_scanner.discovery import (
    build_incremental_scan_plan,
    discover_employee_folders,
)
from attendance_scanner.events import (
    BaseEvent,
    ScanPlanEvent,
    serialize_event,
)
from attendance_scanner.state import ManifestStore


def emit_jsonl_event(event: BaseEvent) -> None:
    """Write one scanner event to stdout without mixing human logs into JSONL."""
    sys.stdout.write(serialize_event(event) + "\n")
    sys.stdout.flush()


def create_parser() -> argparse.ArgumentParser:
    """Create command line parser with plan and scan-batch subcommands."""
    parser = argparse.ArgumentParser(
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
        discovery = discover_employee_folders(args.input)
        store = ManifestStore()
        manifest = store.load_manifest(args.input, output_root=args.output)
        previous_updated_at = manifest.updated_at
        effective_output_root = args.output or manifest.output_root
        plan = build_incremental_scan_plan(
            discovery=discovery,
            manifest=manifest,
            output_root=effective_output_root,
        )
        if manifest.updated_at != previous_updated_at:
            store.save_manifest(manifest)
        event = ScanPlanEvent.from_plan(plan)
        sys.stdout.write(serialize_event(event) + "\n")
        sys.stdout.flush()
        return 0
    except (InvalidInputRootError, StateError, ValueError) as exc:
        message = exc.message if isinstance(exc, (InvalidInputRootError, StateError)) else str(exc)
        sys.stderr.write(f"Error: {message}\n")
        sys.stderr.flush()
        return 1


def handle_scan_batch(args: argparse.Namespace) -> int:
    """Execute scan-batch subcommand by planning and streaming events."""
    try:
        discovery = discover_employee_folders(args.input)
        store = ManifestStore()
        manifest = store.load_manifest(args.input, output_root=args.output)
        previous_updated_at = manifest.updated_at
        effective_output_root = args.output or manifest.output_root
        plan = build_incremental_scan_plan(
            discovery=discovery,
            manifest=manifest,
            output_root=effective_output_root,
        )
        if manifest.updated_at != previous_updated_at:
            store.save_manifest(manifest)
        plan_event = ScanPlanEvent.from_plan(plan)
        sys.stdout.write(serialize_event(plan_event) + "\n")

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
    parser = create_parser()
    args = parser.parse_args(argv)

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
