"""JSON Lines (JSONL) event protocol definitions for Attendance Scanner."""

from datetime import datetime, timezone
from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic.alias_generators import to_camel

from attendance_scanner.contracts import (
    PROTOCOL_VERSION,
    BatchSummary,
    ScannerErrorCode,
    ScanPlan,
)


class BaseEvent(BaseModel):
    """Base event model with protocol versioning, timestamp, and forward compatibility."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
        extra="allow",
    )

    protocol_version: int = Field(default=PROTOCOL_VERSION)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ScanPlanEvent(BaseEvent):
    """Emitted after planning inspection is completed."""

    type: Literal["scan_plan"] = "scan_plan"
    input_root: str
    output_root: str
    total_employees: int = 0
    total_images: int = 0
    new_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    files_to_process: int = 0
    collisions: List[str] = Field(default_factory=list)

    @classmethod
    def from_plan(cls, plan: ScanPlan, timestamp: Optional[str] = None) -> "ScanPlanEvent":
        """Construct ScanPlanEvent from ScanPlan contract model."""
        if timestamp is not None:
            return cls(
                input_root=plan.input_root,
                output_root=plan.output_root,
                total_employees=plan.total_employees,
                total_images=plan.total_images,
                new_count=plan.new_count,
                modified_count=plan.modified_count,
                unchanged_count=plan.unchanged_count,
                files_to_process=plan.files_to_process,
                collisions=list(plan.collisions),
                timestamp=timestamp,
            )
        return cls(
            input_root=plan.input_root,
            output_root=plan.output_root,
            total_employees=plan.total_employees,
            total_images=plan.total_images,
            new_count=plan.new_count,
            modified_count=plan.modified_count,
            unchanged_count=plan.unchanged_count,
            files_to_process=plan.files_to_process,
            collisions=list(plan.collisions),
        )


class FileStartedEvent(BaseEvent):
    """Emitted when processing starts on a specific image."""

    type: Literal["file_started"] = "file_started"
    relative_path: str
    employee_name: str
    index: int
    total: int


class FileCompletedEvent(BaseEvent):
    """Emitted when processing succeeds or finishes with a warning."""

    type: Literal["file_completed"] = "file_completed"
    relative_path: str
    employee_name: str
    output_relative_path: str
    document_detected: bool = False
    warning: Optional[str] = None
    duration_ms: int = 0


class FileFailedEvent(BaseEvent):
    """Emitted when processing fails for a specific image."""

    type: Literal["file_failed"] = "file_failed"
    relative_path: str
    employee_name: str
    error_code: ScannerErrorCode
    message: str


class ScanCompletedEvent(BaseEvent):
    """Emitted when the entire scan batch terminates."""

    type: Literal["scan_completed"] = "scan_completed"
    total_processed: int = 0
    success: int = 0
    failed: int = 0
    warning: int = 0
    skipped: int = 0
    duration_ms: int = 0

    @classmethod
    def from_summary(
        cls, summary: BatchSummary, timestamp: Optional[str] = None
    ) -> "ScanCompletedEvent":
        """Construct ScanCompletedEvent from BatchSummary contract model."""
        if timestamp is not None:
            return cls(
                total_processed=summary.total_images,
                success=summary.success,
                failed=summary.failed,
                warning=summary.warning,
                skipped=summary.skipped,
                duration_ms=summary.duration_ms or 0,
                timestamp=timestamp,
            )
        return cls(
            total_processed=summary.total_images,
            success=summary.success,
            failed=summary.failed,
            warning=summary.warning,
            skipped=summary.skipped,
            duration_ms=summary.duration_ms or 0,
        )


ScannerEvent = Annotated[
    Union[
        ScanPlanEvent,
        FileStartedEvent,
        FileCompletedEvent,
        FileFailedEvent,
        ScanCompletedEvent,
    ],
    Field(discriminator="type"),
]

_EVENT_ADAPTER: TypeAdapter[ScannerEvent] = TypeAdapter(ScannerEvent)


def serialize_event(event: BaseEvent) -> str:
    """Serialize any event model to a single-line JSON string without internal newlines."""
    # Ensure camelCase aliases and no newlines
    raw_json = event.model_dump_json(by_alias=True)
    return raw_json.replace("\r", "").replace("\n", " ")


def deserialize_event(line: str) -> ScannerEvent:
    """Deserialize a JSON string line into the strongly typed ScannerEvent."""
    clean_line = line.strip()
    if not clean_line:
        raise ValueError("Cannot deserialize empty line")
    return _EVENT_ADAPTER.validate_json(clean_line)
