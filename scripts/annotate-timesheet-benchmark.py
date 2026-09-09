"""Create, review, and finalize a timesheet benchmark annotation session."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
from attendance_scanner.annotation import (
    AnnotationSession,
    atomic_write_text,
    discover_annotation_session,
    finalize_annotation_session,
    render_annotation_overlay,
    resolve_annotation_image_path,
    run_annotation_gui,
    summarize_annotation_session,
    update_annotation,
)
from attendance_scanner.benchmark import BenchmarkSource


def _load_session(path: Path) -> AnnotationSession:
    if not path.is_file():
        raise ValueError(f"Annotation session does not exist: {path}")
    return AnnotationSession.load(path)


def _parse_json_array(value: str, *, label: str) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(parsed, list):
        raise TypeError(f"{label} must be a JSON array")
    return parsed


def _safe_overlay_name(sample_id: str, image_path: str) -> str:
    stem = Path(image_path).stem or sample_id
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "sample"
    identity = f"{sample_id}\0{image_path}".encode()
    suffix = hashlib.sha256(identity).hexdigest()[:10]
    return f"{safe}_{suffix}_overlay.png"


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage a resumable local timesheet benchmark annotation workflow"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser(
        "init", help="discover images and create a pending session"
    )
    init.add_argument("--image-root", type=Path, required=True)
    init.add_argument("--session", type=Path, required=True)
    init.add_argument("--dataset", required=True)
    init.add_argument(
        "--split", choices=("train", "tune", "acceptance"), default="acceptance"
    )
    init.add_argument("--source-name", required=True)
    init.add_argument("--license", required=True)
    init.add_argument("--attribution", required=True)
    init.add_argument("--source-url", default=None)
    init.add_argument(
        "--force", action="store_true", help="replace an existing session file"
    )

    gui = commands.add_parser("gui", help="open the OpenCV annotation workbench")
    gui.add_argument("--session", type=Path, required=True)
    gui.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="optional relocated image root; the session is updated to this root",
    )

    set_command = commands.add_parser("set", help="update one annotation record")
    set_command.add_argument("--session", type=Path, required=True)
    set_command.add_argument("--sample-id", required=True)
    set_command.add_argument(
        "--corners-json",
        default=None,
        help="four points as JSON, for example [[10,10],[300,12],[298,400],[12,398]]",
    )
    set_command.add_argument(
        "--split", choices=("train", "tune", "acceptance"), default=None
    )
    visibility = set_command.add_mutually_exclusive_group()
    visibility.add_argument("--visible", action="store_true")
    visibility.add_argument("--not-visible", action="store_true")
    set_command.add_argument(
        "--visibility-reason",
        choices=("fully_visible", "border_touching", "out_of_frame", "occluded"),
        default=None,
    )
    set_command.add_argument(
        "--tag", action="append", dest="scenario_tags", default=None
    )
    set_command.add_argument("--clear-tags", action="store_true")
    set_command.add_argument("--polygon-json", default=None)
    set_command.add_argument("--mask-path", default=None)
    set_command.add_argument("--sequence-id", default=None)
    set_command.add_argument("--notes", default=None)
    set_command.add_argument(
        "--status", choices=("pending", "annotated", "skipped"), default=None
    )

    preview = commands.add_parser(
        "preview", help="render a reviewer overlay for one sample"
    )
    preview.add_argument("--session", type=Path, required=True)
    preview.add_argument("--sample-id", required=True)
    preview.add_argument("--output", type=Path, default=None)
    preview.add_argument("--max-dimension", type=int, default=1600)

    summary = commands.add_parser("summary", help="print annotation coverage")
    summary.add_argument("--session", type=Path, required=True)

    finalize = commands.add_parser(
        "finalize", help="validate and write the AS-32 manifest"
    )
    finalize.add_argument("--session", type=Path, required=True)
    finalize.add_argument("--output-manifest", type=Path, required=True)
    finalize.add_argument("--min-samples", type=int, default=200)
    finalize.add_argument("--max-samples", type=int, default=300)

    return parser


def _command_init(args: argparse.Namespace) -> int:
    if args.session.exists() and not args.force:
        raise ValueError(
            f"Annotation session already exists: {args.session}; use --force to replace it"
        )
    session = discover_annotation_session(
        args.image_root,
        dataset=args.dataset,
        split=args.split,
        source=BenchmarkSource(
            name=args.source_name,
            license=args.license,
            attribution=args.attribution,
            url=args.source_url,
        ),
    )
    session.save(args.session)
    _print_json(summarize_annotation_session(session))
    return 0


def _command_gui(args: argparse.Namespace) -> int:
    session = _load_session(args.session)
    if args.image_root is not None:
        session = session.model_copy(
            update={"image_root": str(args.image_root.resolve())}
        )
        session.save(args.session)
    run_annotation_gui(session, args.session)
    return 0


def _command_set(args: argparse.Namespace) -> int:
    session = _load_session(args.session)
    corners = None
    if args.corners_json is not None:
        corners = _parse_json_array(args.corners_json, label="--corners-json")
    polygon = None
    if args.polygon_json is not None:
        polygon = _parse_json_array(args.polygon_json, label="--polygon-json")
    if args.clear_tags:
        scenario_tags: Sequence[str] | None = []
    else:
        scenario_tags = args.scenario_tags
    document_visible: bool | None = None
    if args.visible:
        document_visible = True
    elif args.not_visible:
        document_visible = False
    record = update_annotation(
        session,
        args.sample_id,
        corners=corners,
        split=args.split,
        scenario_tags=scenario_tags,
        document_visible=document_visible,
        visibility_reason=args.visibility_reason,
        document_polygon=polygon,
        mask_path=args.mask_path,
        sequence_id=args.sequence_id,
        notes=args.notes,
        status=args.status,
    )
    session.save(args.session)
    _print_json(record.model_dump(mode="json"))
    return 0


def _command_preview(args: argparse.Namespace) -> int:
    session = _load_session(args.session)
    record = next(
        (item for item in session.records if item.sample_id == args.sample_id), None
    )
    if record is None:
        raise ValueError(f"Unknown annotation sample_id: {args.sample_id}")
    if args.max_dimension <= 0:
        raise ValueError("--max-dimension must be positive")
    image_path = resolve_annotation_image_path(session, record)
    output = args.output or args.session.parent / "overlays" / _safe_overlay_name(
        record.sample_id, record.image_path
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = render_annotation_overlay(
        image_path, record, max_dimension=args.max_dimension
    )
    if not cv2.imwrite(str(output), overlay):
        raise ValueError(f"Could not write annotation overlay: {output}")
    _print_json({"sampleId": record.sample_id, "output": str(output.resolve())})
    return 0


def _command_summary(args: argparse.Namespace) -> int:
    _print_json(summarize_annotation_session(_load_session(args.session)))
    return 0


def _command_finalize(args: argparse.Namespace) -> int:
    if args.min_samples < 1 or args.max_samples < args.min_samples:
        raise ValueError("sample bounds must satisfy 1 <= min-samples <= max-samples")
    session = _load_session(args.session)
    manifest = finalize_annotation_session(
        session,
        minimum_samples=args.min_samples,
        maximum_samples=args.max_samples,
    )
    output = args.output_manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output, manifest.dump_canonical_json())
    _print_json(
        {
            "manifest": str(output.resolve()),
            "sampleCount": len(manifest.samples),
            "splits": sorted({sample.split for sample in manifest.samples}),
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        handlers = {
            "init": _command_init,
            "gui": _command_gui,
            "set": _command_set,
            "preview": _command_preview,
            "summary": _command_summary,
            "finalize": _command_finalize,
        }
        return handlers[args.command](args)
    except Exception as exc:  # noqa: BLE001 - CLI must report validation/decode failures as JSON
        _print_json({"valid": False, "error": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
