"""CLI entrypoint for Attendance Scanner sidecar."""

import argparse
import json
import sys
from typing import List, Optional


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
    """Placeholder execution for plan subcommand."""
    # MVP skeleton output in JSONL
    event = {
        "type": "scan_plan",
        "input": args.input,
        "output": args.output or f"{args.input}_pdf",
        "employees": 0,
        "totalImages": 0,
        "new": 0,
        "modified": 0,
        "unchanged": 0,
        "filesToProcess": 0,
        "collisions": [],
        "status": "skeleton_ready",
    }
    sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


def handle_scan_batch(args: argparse.Namespace) -> int:
    """Placeholder execution for scan-batch subcommand."""
    # Emit plan event
    output_dir = args.output or f"{args.input}_pdf"
    plan_event = {
        "type": "scan_plan",
        "input": args.input,
        "output": output_dir,
        "employees": 0,
        "totalImages": 0,
        "new": 0,
        "modified": 0,
        "unchanged": 0,
        "status": "skeleton_ready",
    }
    sys.stdout.write(json.dumps(plan_event, ensure_ascii=False) + "\n")

    # Emit completed event
    complete_event = {
        "type": "scan_completed",
        "success": 0,
        "failed": 0,
        "warning": 0,
        "skipped": 0,
    }
    sys.stdout.write(json.dumps(complete_event, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return 0


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
