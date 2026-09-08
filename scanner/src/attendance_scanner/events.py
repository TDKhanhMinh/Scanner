"""JSON Lines (JSONL) event protocol definitions for Attendance Scanner."""

from datetime import datetime, timezone
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo, model_validator
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

    @model_validator(mode="before")
    @classmethod
    def validate_wire_protocol(cls, data: Any, info: ValidationInfo) -> Any:
        """Enforce protocolVersion and timestamp presence during wire/JSON deserialization."""
        if info.mode == "json" and isinstance(data, dict):
            if "protocolVersion" not in data and "protocol_version" not in data:
                raise ValueError("Missing required wire field: protocolVersion")
            proto_val = data.get("protocolVersion", data.get("protocol_version"))
            if proto_val != PROTOCOL_VERSION:
                raise ValueError(
                    f"Unsupported protocolVersion: {proto_val} (expected {PROTOCOL_VERSION})"
                )
            if "timestamp" not in data:
                raise ValueError("Missing required wire field: timestamp")
            ts = data.get("timestamp")
            if not isinstance(ts, str) or not ts.strip():
                raise ValueError("timestamp must be a non-empty string")
        return data


class ScanPlanEvent(BaseEvent):
    """Emitted after planning inspection is completed, matching System Design fields."""

    type: Literal["scan_plan"] = "scan_plan"
    input_root: str = Field(min_length=1)
    output_root: str = Field(min_length=1)
    employees: int = Field(default=0, ge=0)
    total_images: int = Field(default=0, ge=0)
    new: int = Field(default=0, ge=0)
    modified: int = Field(default=0, ge=0)
    rebuild: int = Field(default=0, ge=0)
    unchanged: int = Field(default=0, ge=0)
    files_to_process: int = Field(default=0, ge=0)
    outdated_pipeline_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    collisions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_plan_aliases(cls, data: Any) -> Any:
        """Accept either System Design canonical fields or legacy aliases."""
        if isinstance(data, dict):
            if "employees" not in data and "total_employees" in data:
                data["employees"] = data["total_employees"]
            elif "employees" not in data and "totalEmployees" in data:
                data["employees"] = data["totalEmployees"]

            if "new" not in data and "new_count" in data:
                data["new"] = data["new_count"]
            elif "new" not in data and "newCount" in data:
                data["new"] = data["newCount"]

            if "modified" not in data and "modified_count" in data:
                data["modified"] = data["modified_count"]
            elif "modified" not in data and "modifiedCount" in data:
                data["modified"] = data["modifiedCount"]

            if "rebuild" not in data and "rebuild_count" in data:
                data["rebuild"] = data["rebuild_count"]
            elif "rebuild" not in data and "rebuildCount" in data:
                data["rebuild"] = data["rebuildCount"]

            if "files_to_process" not in data and "filesToProcess" not in data:
                counts = (data.get("new", 0), data.get("modified", 0), data.get("rebuild", 0))
                if all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in counts
                ):
                    data["files_to_process"] = sum(counts)

            if "unchanged" not in data and "unchanged_count" in data:
                data["unchanged"] = data["unchanged_count"]
            elif "unchanged" not in data and "unchangedCount" in data:
                data["unchanged"] = data["unchangedCount"]
        return data

    @model_validator(mode="after")
    def validate_process_count(self) -> "ScanPlanEvent":
        """Reject a scan-plan event whose process count disagrees with counters."""
        expected = self.new + self.modified + self.rebuild
        if self.files_to_process != expected:
            raise ValueError(
                "files_to_process must equal new + modified + rebuild "
                f"({expected}), got {self.files_to_process}"
            )
        return self

    @property
    def total_employees(self) -> int:
        return self.employees

    @property
    def new_count(self) -> int:
        return self.new

    @property
    def modified_count(self) -> int:
        return self.modified

    @property
    def rebuild_count(self) -> int:
        return self.rebuild

    @property
    def unchanged_count(self) -> int:
        return self.unchanged

    @classmethod
    def from_plan(cls, plan: ScanPlan, timestamp: Optional[str] = None) -> "ScanPlanEvent":
        """Construct ScanPlanEvent from ScanPlan contract model."""
        if timestamp is not None:
            return cls(
                input_root=plan.input_root,
                output_root=plan.output_root,
                employees=plan.employees,
                total_images=plan.total_images,
                new=plan.new,
                modified=plan.modified,
                rebuild=plan.rebuild,
                unchanged=plan.unchanged,
                files_to_process=plan.files_to_process,
                outdated_pipeline_count=plan.outdated_pipeline_count,
                unsupported_count=plan.unsupported_count,
                collisions=list(plan.collisions),
                timestamp=timestamp,
            )
        return cls(
            input_root=plan.input_root,
            output_root=plan.output_root,
            employees=plan.employees,
            total_images=plan.total_images,
            new=plan.new,
            modified=plan.modified,
            rebuild=plan.rebuild,
            unchanged=plan.unchanged,
            files_to_process=plan.files_to_process,
            outdated_pipeline_count=plan.outdated_pipeline_count,
            unsupported_count=plan.unsupported_count,
            collisions=list(plan.collisions),
        )


class FileStartedEvent(BaseEvent):
    """Emitted when processing starts on a specific image."""

    type: Literal["file_started"] = "file_started"
    relative_path: str = Field(min_length=1)
    employee_name: str = Field(min_length=1)
    index: int = Field(ge=0)
    total: int = Field(ge=0)


class FileCompletedEvent(BaseEvent):
    """Emitted when processing succeeds or finishes with a warning."""

    type: Literal["file_completed"] = "file_completed"
    relative_path: str = Field(min_length=1)
    employee_name: str = Field(min_length=1)
    output_relative_path: str = Field(min_length=1)
    document_detected: bool = False
    warning: Optional[str] = None
    duration_ms: int = Field(default=0, ge=0)


class FileFailedEvent(BaseEvent):
    """Emitted when processing fails for a specific image."""

    type: Literal["file_failed"] = "file_failed"
    relative_path: str = Field(min_length=1)
    employee_name: str = Field(min_length=1)
    error_code: ScannerErrorCode
    message: str = Field(min_length=1)


class ScanCompletedEvent(BaseEvent):
    """Emitted when the entire scan batch terminates."""

    type: Literal["scan_completed"] = "scan_completed"
    total_processed: int = Field(default=0, ge=0)
    success: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    warning: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    duration_ms: int = Field(default=0, ge=0)

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
    raw_json = event.model_dump_json(by_alias=True)
    return raw_json.replace("\r", "").replace("\n", " ")


def deserialize_event(line: str) -> ScannerEvent:
    """Deserialize a JSON string line into ScannerEvent with strict validation."""
    clean_line = line.strip()
    if not clean_line:
        raise ValueError("Cannot deserialize empty line")
    return _EVENT_ADAPTER.validate_json(clean_line)
