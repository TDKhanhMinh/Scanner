"""Resumable local annotation workflow for timesheet benchmark images."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from attendance_scanner.benchmark import (
    BenchmarkManifest,
    BenchmarkPoint,
    BenchmarkSource,
    BenchmarkSplit,
)
from attendance_scanner.pipeline.load import load_image

ANNOTATION_SESSION_VERSION: Literal["1.0"] = "1.0"
AnnotationStatus = Literal["pending", "annotated", "skipped"]
VisibilityReason = Literal["fully_visible", "border_touching", "out_of_frame", "occluded"]


def atomic_write_text(path: Union[str, Path], content: str) -> None:
    """Write UTF-8 text through a same-directory atomic replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


class AnnotationRecord(BaseModel):
    """Editable annotation state for one discovered source image."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    split: BenchmarkSplit = "acceptance"
    status: AnnotationStatus = "pending"
    sequence_id: Optional[str] = None
    corners: Optional[List[BenchmarkPoint]] = Field(default=None, min_length=4, max_length=4)
    document_polygon: Optional[List[BenchmarkPoint]] = Field(default=None, min_length=3)
    mask_path: Optional[str] = None
    document_visible: bool = True
    visibility_reason: Optional[VisibilityReason] = None
    scenario_tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @model_validator(mode="after")
    def validate_annotation_state(self) -> "AnnotationRecord":
        """Reject malformed completed records while allowing pending sessions."""
        if any(not tag.strip() for tag in self.scenario_tags):
            raise ValueError("scenario_tags must not contain empty values")
        if self.status == "annotated" and self.corners is None:
            raise ValueError("annotated records require four corners")
        if self.document_visible and self.visibility_reason not in (None, "fully_visible"):
            raise ValueError("visible records may only use visibility_reason=fully_visible")
        if not self.document_visible and self.visibility_reason in (None, "fully_visible"):
            raise ValueError("invisible records require an explicit visibility_reason")
        return self


class AnnotationSession(BaseModel):
    """Persisted, resumable annotation session metadata."""

    model_config = ConfigDict(extra="forbid")

    session_version: Literal["1.0"] = ANNOTATION_SESSION_VERSION
    image_root: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    source: BenchmarkSource
    records: List[AnnotationRecord] = Field(default_factory=list)
    updated_at: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def validate_record_identity(self) -> "AnnotationSession":
        ids = [record.sample_id for record in self.records]
        paths = [record.image_path.replace("\\", "/") for record in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("annotation sample_id values must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("annotation image_path values must be unique")
        return self

    def save(self, path: Union[str, Path]) -> None:
        """Persist the session deterministically for resume/audit."""
        self.updated_at = time.time()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            destination,
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnnotationSession":
        """Load and validate a previously saved session."""
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def discover_annotation_session(
    image_root: Union[str, Path],
    *,
    dataset: str,
    source: BenchmarkSource,
    split: BenchmarkSplit = "acceptance",
) -> AnnotationSession:
    """Create a stable pending session from direct child image files."""
    root = Path(image_root).resolve()
    if not root.is_dir():
        raise ValueError(f"Annotation image root does not exist: {root}")
    records: List[AnnotationRecord] = []
    for path in sorted(
        (
            candidate
            for candidate in root.iterdir()
            if candidate.is_file()
            and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ),
        key=lambda item: (item.name.casefold(), item.name),
    ):
        resolved_path = path.resolve()
        try:
            resolved_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Annotation image path escapes root: {path.name}") from exc
        relative_path = path.relative_to(root).as_posix()
        loaded = load_image(resolved_path)
        records.append(
            AnnotationRecord(
                sample_id=f"{dataset}:{relative_path}",
                image_path=relative_path,
                width=loaded.width,
                height=loaded.height,
                split=split,
            )
        )
    return AnnotationSession(
        image_root=str(root),
        dataset=dataset,
        source=source,
        records=records,
    )


def _assert_inside_image(point: BenchmarkPoint, record: AnnotationRecord) -> bool:
    return 0.0 <= point[0] < record.width and 0.0 <= point[1] < record.height


def resolve_annotation_image_path(session: AnnotationSession, record: AnnotationRecord) -> Path:
    """Resolve one record image and reject traversal or symlink escapes."""
    root = Path(session.image_root).resolve()
    image_path = (root / record.image_path).resolve()
    try:
        image_path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Annotation image path escapes root: {record.image_path}") from exc
    return image_path


def update_annotation(
    session: AnnotationSession,
    sample_id: str,
    *,
    corners: Optional[Sequence[BenchmarkPoint]] = None,
    split: Optional[BenchmarkSplit] = None,
    scenario_tags: Optional[Sequence[str]] = None,
    document_visible: Optional[bool] = None,
    visibility_reason: Optional[VisibilityReason] = None,
    document_polygon: Optional[Sequence[BenchmarkPoint]] = None,
    mask_path: Optional[str] = None,
    sequence_id: Optional[str] = None,
    notes: Optional[str] = None,
    status: Optional[AnnotationStatus] = None,
) -> AnnotationRecord:
    """Update one record through the same validation path used by finalize."""
    record = next((item for item in session.records if item.sample_id == sample_id), None)
    if record is None:
        raise ValueError(f"Unknown annotation sample_id: {sample_id}")
    values = record.model_dump()
    if corners is not None:
        values["corners"] = [tuple(point) for point in corners]
        values["status"] = "annotated"
    if split is not None:
        values["split"] = split
    if scenario_tags is not None:
        values["scenario_tags"] = list(scenario_tags)
    if document_visible is not None:
        values["document_visible"] = document_visible
        if document_visible and visibility_reason is None:
            values["visibility_reason"] = "fully_visible"
    if visibility_reason is not None:
        values["visibility_reason"] = visibility_reason
    if document_polygon is not None:
        values["document_polygon"] = [tuple(point) for point in document_polygon]
    if mask_path is not None:
        values["mask_path"] = mask_path
    if sequence_id is not None:
        values["sequence_id"] = sequence_id
    if notes is not None:
        values["notes"] = notes
    if status is not None:
        values["status"] = status
    updated = AnnotationRecord.model_validate(values)
    session.records[session.records.index(record)] = updated
    session.updated_at = time.time()
    return updated


def render_annotation_overlay(
    image_path: Union[str, Path],
    record: AnnotationRecord,
    *,
    max_dimension: int = 1600,
) -> np.ndarray:
    """Render a review overlay without changing the source image."""
    loaded = load_image(image_path)
    image = loaded.image.copy()
    scale = min(1.0, max_dimension / float(max(image.shape[:2])))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(image.shape[1] * scale), round(image.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    points = record.corners or []
    scaled_points = np.array(
        [[round(x * scale), round(y * scale)] for x, y in points], dtype=np.int32
    )
    if len(scaled_points) >= 2:
        cv2.polylines(image, [scaled_points.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    for index, (x, y) in enumerate(scaled_points):
        cv2.circle(image, (int(x), int(y)), 9, (0, 255, 0), -1)
        cv2.putText(
            image,
            ("TL", "TR", "BR", "BL")[index],
            (int(x) + 12, int(y) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return image


def run_annotation_gui(session: AnnotationSession, session_path: Union[str, Path]) -> None:
    """Run the local OpenCV click/drag annotation workbench.

    Controls: left-click adds a corner, right-click removes the last corner,
    ``r`` resets points, ``s`` saves the current four corners, ``k`` skips the
    current image, ``n``/``p`` navigate, and ``q`` saves the session and exits.
    """
    window_name = "Attendance Scanner - benchmark annotation"
    if not session.records:
        raise ValueError("Annotation session contains no images")
    current_index = 0
    points: List[BenchmarkPoint] = []
    display_scale = 1.0

    def load_current_points() -> None:
        nonlocal points
        record = session.records[current_index]
        points = list(record.corners or [])

    def save_current() -> None:
        nonlocal points
        record = session.records[current_index]
        if len(points) != 4:
            raise ValueError("Exactly four corners are required before saving")
        update_annotation(session, record.sample_id, corners=points)
        session.save(session_path)

    def handle_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x / display_scale, y / display_scale))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    load_current_points()
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, handle_mouse)
    while True:
        record = session.records[current_index]
        image_path = resolve_annotation_image_path(session, record)
        loaded = load_image(image_path)
        display_scale = min(1.0, 1400.0 / float(max(loaded.image.shape[:2])))
        display = loaded.image.copy()
        if display_scale < 1.0:
            display = cv2.resize(
                display,
                (round(display.shape[1] * display_scale), round(display.shape[0] * display_scale)),
                interpolation=cv2.INTER_AREA,
            )
        scaled = np.array(
            [[round(x * display_scale), round(y * display_scale)] for x, y in points],
            dtype=np.int32,
        )
        if len(scaled) >= 2:
            cv2.polylines(display, [scaled.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
        for index, (x, y) in enumerate(scaled):
            cv2.circle(display, (int(x), int(y)), 10, (0, 255, 0), -1)
            cv2.putText(
                display,
                ("TL", "TR", "BR", "BL")[index],
                (int(x) + 12, int(y) - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            display,
            f"{current_index + 1}/{len(session.records)} {record.image_path} points={len(points)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "click add | right-click undo | r reset | s save | k skip | n/p next/prev | q quit",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            session.save(session_path)
            break
        if key == ord("r"):
            points = []
        elif key == ord("s"):
            save_current()
        elif key == ord("k"):
            session.records[current_index] = session.records[current_index].model_copy(
                update={"status": "skipped"}
            )
            session.save(session_path)
        elif key in (ord("n"), 13):
            if current_index < len(session.records) - 1:
                current_index += 1
                load_current_points()
        elif key == ord("p"):
            if current_index > 0:
                current_index -= 1
                load_current_points()
    cv2.destroyWindow(window_name)


def finalize_annotation_session(
    session: AnnotationSession,
    *,
    minimum_samples: int = 200,
    maximum_samples: int = 300,
) -> BenchmarkManifest:
    """Validate all completed records and build the canonical AS-32 manifest."""
    if not minimum_samples <= len(session.records) <= maximum_samples:
        raise ValueError(
            f"Timesheet corpus must contain {minimum_samples}-{maximum_samples} records; "
            f"found {len(session.records)}"
        )
    pending = [record.sample_id for record in session.records if record.status != "annotated"]
    if pending:
        raise ValueError(f"Unannotated records remain: {', '.join(pending[:5])}")

    samples = []
    root = Path(session.image_root).resolve()
    for record in session.records:
        if record.corners is None:
            raise ValueError(f"Missing corners for {record.sample_id}")
        if record.document_visible and not all(
            _assert_inside_image(point, record) for point in record.corners
        ):
            raise ValueError(f"Visible record has out-of-bounds corners: {record.sample_id}")
        image_path = resolve_annotation_image_path(session, record)
        if not image_path.is_file():
            raise ValueError(f"Annotation image does not exist: {record.image_path}")
        samples.append(
            {
                "sample_id": record.sample_id,
                "image_path": record.image_path,
                "dataset": session.dataset,
                "split": record.split,
                "width": record.width,
                "height": record.height,
                "tl": record.corners[0],
                "tr": record.corners[1],
                "br": record.corners[2],
                "bl": record.corners[3],
                "document_polygon": record.document_polygon,
                "mask_path": record.mask_path,
                "document_visible": record.document_visible,
                "visibility_reason": record.visibility_reason,
                "scenario_tags": record.scenario_tags,
                "source": session.source.model_dump(),
                "sequence_id": record.sequence_id,
            }
        )
    manifest = BenchmarkManifest.model_validate({"samples": samples})
    manifest.validate_references(root)
    return manifest


def summarize_annotation_session(session: AnnotationSession) -> Dict[str, object]:
    """Return coverage counts for review and benchmark provenance."""
    status_counts = Counter(record.status for record in session.records)
    split_counts = Counter(record.split for record in session.records)
    tag_counts = Counter(tag for record in session.records for tag in record.scenario_tags)
    return {
        "sessionVersion": session.session_version,
        "dataset": session.dataset,
        "sampleCount": len(session.records),
        "status": dict(sorted(status_counts.items())),
        "bySplit": dict(sorted(split_counts.items())),
        "scenarioTags": dict(sorted(tag_counts.items())),
        "documentPolygonCount": sum(
            record.document_polygon is not None for record in session.records
        ),
        "maskCount": sum(record.mask_path is not None for record in session.records),
        "visibleCount": sum(record.document_visible for record in session.records),
    }


__all__ = [
    "ANNOTATION_SESSION_VERSION",
    "AnnotationRecord",
    "AnnotationSession",
    "atomic_write_text",
    "discover_annotation_session",
    "finalize_annotation_session",
    "render_annotation_overlay",
    "resolve_annotation_image_path",
    "run_annotation_gui",
    "summarize_annotation_session",
    "update_annotation",
]
