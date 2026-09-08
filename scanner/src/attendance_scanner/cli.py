"""CLI entrypoint for Attendance Scanner sidecar."""

import argparse
import sys
from pathlib import Path
from typing import List, NoReturn, Optional

from attendance_scanner.batch import run_batch
from attendance_scanner.contracts import (
    BatchPeriod,
    DiscoveryResult,
    ExportMode,
    OutputNotWritableError,
    ScanPlan,
)
from attendance_scanner.diagnostics import (
    ScannerOperation,
    configure_logging,
    describe_scanner_error,
    log_scanner_error,
)
from attendance_scanner.discovery import (
    build_group_aware_scan_plan,
    build_incremental_scan_plan,
    discover_employee_folders,
)
from attendance_scanner.events import (
    BaseEvent,
    ScanPlanEvent,
    serialize_event,
)
from attendance_scanner.state import Manifest, ManifestStore, get_default_state_dir


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
        raise OutputNotWritableError(str(resolved), "Output root is not a directory")
    return str(resolved)


def _report_cli_error(exc: BaseException, *, operation: ScannerOperation) -> int:
    """Log a structured fatal error and keep stderr free of tracebacks."""
    info = describe_scanner_error(exc, operation=operation)
    log_scanner_error(info, exc, operation=operation)
    return 1


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


def _parse_batch_period(args: argparse.Namespace) -> Optional[BatchPeriod]:
    """Parse the optional user-selected period without consulting system time."""
    year = getattr(args, "year", None)
    month = getattr(args, "month", None)
    if year is None and month is None:
        return None
    if year is None or month is None:
        raise CliArgumentError("--year and --month must be provided together")
    try:
        return BatchPeriod(year=year, month=month)
    except ValueError as exc:
        raise CliArgumentError(str(exc)) from exc


def _parse_export_mode(value: str) -> ExportMode:
    """Normalize the CLI kebab-case export mode to the shared enum."""
    try:
        return ExportMode(value.replace("-", "_").upper())
    except ValueError as exc:
        raise CliArgumentError(f"Unsupported export mode: {value}") from exc


def _group_summary(
    input_root: str,
    output_root: str,
    period: BatchPeriod,
    export_mode: ExportMode,
) -> dict[str, int]:
    """Summarize affected document groups for a typed scan-plan event."""
    discovery = discover_employee_folders(input_root)
    manifest = ManifestStore().load_manifest(input_root, output_root=output_root)
    group_plan = build_group_aware_scan_plan(
        discovery,
        manifest,
        output_root,
        period,
        export_mode=export_mode,
    )
    counts = {
        "document_groups": group_plan.affected_group_count,
        "expected_artifacts": (
            group_plan.affected_group_count
            if export_mode == ExportMode.GROUPED
            else group_plan.source_plan.files_to_process
        ),
        "complete_groups": 0,
        "incomplete_groups": 0,
        "ambiguous_groups": 0,
        "pages_needing_review": sum(
            group.review_required for group in group_plan.affected_groups
        ),
    }
    for group in group_plan.affected_groups:
        field = f"{group.completeness_status.value.lower()}_groups"
        if field in counts:
            counts[field] += 1
    return counts


def _plan_event(
    plan: ScanPlan,
    *,
    input_root: str,
    output_root: str,
    period: Optional[BatchPeriod],
    export_mode: ExportMode,
) -> ScanPlanEvent:
    """Create a typed plan event with document-aware summary counters."""
    if period is None:
        return ScanPlanEvent.from_plan(plan, period=None, export_mode=export_mode)
    counts = _group_summary(input_root, output_root, period, export_mode)
    return ScanPlanEvent.from_plan(
        plan,
        period=period,
        export_mode=export_mode,
        document_groups=counts["document_groups"],
        expected_artifacts=counts["expected_artifacts"],
        complete_groups=counts["complete_groups"],
        incomplete_groups=counts["incomplete_groups"],
        ambiguous_groups=counts["ambiguous_groups"],
        pages_needing_review=counts["pages_needing_review"],
    )


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
    plan_parser.add_argument("--year", type=int, help="Batch year (1-9999)")
    plan_parser.add_argument("--month", type=int, help="Batch month (1-12)")
    plan_parser.add_argument(
        "--export-mode",
        choices=["per-image", "grouped"],
        default="per-image",
        help="PDF export strategy (default: per-image)",
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
    scan_parser.add_argument("--year", type=int, help="Batch year (1-9999)")
    scan_parser.add_argument("--month", type=int, help="Batch month (1-12)")
    scan_parser.add_argument(
        "--export-mode",
        choices=["per-image", "grouped"],
        default="per-image",
        help="PDF export strategy (default: per-image)",
    )

    return parser


def handle_plan(args: argparse.Namespace) -> int:
    """Execute plan subcommand by discovering employee folders and emitting ScanPlanEvent."""
    try:
        _, _, _, _, plan = _prepare_plan(args.input, args.output)
        period = _parse_batch_period(args)
        export_mode = _parse_export_mode(args.export_mode)
        event = _plan_event(
            plan,
            input_root=args.input,
            output_root=plan.output_root,
            period=period,
            export_mode=export_mode,
        )
        emit_jsonl_event(event)
        return 0
    except Exception as exc:
        return _report_cli_error(exc, operation="plan")


def handle_scan_batch(args: argparse.Namespace) -> int:
    """Execute scan-batch subcommand by planning and streaming events."""
    try:
        period = _parse_batch_period(args)
        export_mode = _parse_export_mode(args.export_mode)
        discovery, manifest, store, effective_output_root, plan = _prepare_plan(
            args.input,
            args.output,
        )
        plan_event = _plan_event(
            plan,
            input_root=args.input,
            output_root=effective_output_root,
            period=period,
            export_mode=export_mode,
        )
        emit_jsonl_event(plan_event)

        result = run_batch(
            discovery=discovery,
            manifest=manifest,
            output_root=effective_output_root,
            mode=args.mode,
            workers=args.workers,
            manifest_store=store,
            emit=emit_jsonl_event,
            batch_period=period,
        )
        sys.stdout.flush()
        return result.exit_code
    except Exception as exc:
        return _report_cli_error(exc, operation="request")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    configure_stdio()
    configure_logging(get_default_state_dir().parent / "attendance-scanner.log")
    parser = create_parser()
    try:
        args = parser.parse_args(argv)
    except CliArgumentError as exc:
        return _report_cli_error(exc, operation="request")

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
