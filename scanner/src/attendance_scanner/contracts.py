"""Shared contracts and type definitions for Attendance Scanner."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

# Canonical protocol version for all JSONL events
PROTOCOL_VERSION = 1


class ScanMode(str, Enum):
    """Scan enhancement filter mode."""

    GRAY = "gray"
    BW = "bw"
    COLOR = "color"


class FileClassification(str, Enum):
    """Classification state for discovered files during incremental planning."""

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    REBUILD = "rebuild"
    COLLISION = "collision"
    UNSUPPORTED = "unsupported"


class FileProcessingStatus(str, Enum):
    """Processing lifecycle status of an image file."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScannerErrorCode(str, Enum):
    """Canonical error codes for scanner operations."""

    INVALID_INPUT_ROOT = "INVALID_INPUT_ROOT"
    OUTPUT_NOT_WRITABLE = "OUTPUT_NOT_WRITABLE"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    PDF_WRITE_FAILED = "PDF_WRITE_FAILED"
    STATE_READ_FAILED = "STATE_READ_FAILED"
    STATE_WRITE_FAILED = "STATE_WRITE_FAILED"
    OUTPUT_COLLISION = "OUTPUT_COLLISION"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class BaseContract(BaseModel):
    """Base contract model supporting camelCase aliases for interoperability."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class DiscoveredFile(BaseContract):
    """Metadata of a discovered image file."""

    employee_name: str
    file_name: str
    relative_path: str
    absolute_path: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None
    classification: FileClassification = FileClassification.NEW
    target_relative_pdf: str


class ScanPlan(BaseContract):
    """Overall summary plan before executing scan batch, matching System Design fields."""

    input_root: str
    output_root: str
    employees: int = 0
    total_images: int = 0
    new: int = 0
    modified: int = 0
    unchanged: int = 0
    files_to_process: int = 0
    collisions: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_aliases(cls, data: Any) -> Any:
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

            if "unchanged" not in data and "unchanged_count" in data:
                data["unchanged"] = data["unchanged_count"]
            elif "unchanged" not in data and "unchangedCount" in data:
                data["unchanged"] = data["unchangedCount"]
        return data

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
    def unchanged_count(self) -> int:
        return self.unchanged


class ScanBatchRequest(BaseContract):
    """Parameters passed to initiate a scan batch."""

    input_root: str
    output_root: Optional[str] = None
    mode: ScanMode = ScanMode.GRAY
    workers: int = Field(default=3, ge=1, le=4)


class FileResult(BaseContract):
    """Result of processing a single image file."""

    relative_path: str
    employee_name: str
    target_relative_pdf: str
    status: FileProcessingStatus
    document_detected: bool = False
    warning: Optional[str] = None
    error_code: Optional[ScannerErrorCode] = None
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class BatchSummary(BaseContract):
    """Cumulative summary after completing or terminating a scan batch."""

    total_images: int
    success: int = 0
    failed: int = 0
    warning: int = 0
    skipped: int = 0
    duration_ms: Optional[int] = None
