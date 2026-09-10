"""Shared contracts and type definitions for Attendance Scanner."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

# Canonical protocol version for all JSONL events
PROTOCOL_VERSION = 1


class ScanMode(str, Enum):
    """Scan enhancement filter mode."""

    GRAY = "gray"
    BW = "bw"
    COLOR = "color"
    SMART_DOCUMENT = "smart_document"


class ExportMode(str, Enum):
    """Output strategy for a scan batch."""

    PER_IMAGE = "PER_IMAGE"
    GROUPED = "GROUPED"


class PageType(str, Enum):
    """Known page roles for the current attendance template."""

    FIRST_HALF = "FIRST_HALF"
    SECOND_HALF = "SECOND_HALF"
    UNKNOWN = "UNKNOWN"


class CompletenessStatus(str, Enum):
    """Document-group completeness/review state."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    AMBIGUOUS = "AMBIGUOUS"


class BaseContract(BaseModel):
    """Base contract model supporting camelCase aliases for interoperability."""

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=to_camel,
    )


class BatchPeriod(BaseContract):
    """User-selected year/month context for one scan batch."""

    year: int = Field(ge=1, le=9999)
    month: int = Field(ge=1, le=12)


class PageIdentity(BaseContract):
    """Typed identity and ordering metadata for one document page."""

    page_type: PageType = PageType.UNKNOWN
    page_order: Optional[int] = Field(default=None, ge=1)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detection_method: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)


class DocumentGroupKey(BaseContract):
    """Stable group identity independent of source filename."""

    employee_relative_dir: str = Field(min_length=1)
    year: int = Field(ge=1, le=9999)
    month: int = Field(ge=1, le=12)


class SourcePage(BaseContract):
    """Technical source reference paired with its page identity."""

    source_relative_path: str = Field(min_length=1)
    identity: PageIdentity = Field(default_factory=PageIdentity)


class ReviewGroup(BaseContract):
    """Review payload shown before an ambiguous grouped export."""

    key: DocumentGroupKey
    source_pages: List[SourcePage] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    completeness_status: CompletenessStatus = CompletenessStatus.AMBIGUOUS
    review_required: bool = True
    manual_order: List[str] = Field(default_factory=list)


class DocumentGroup(BaseContract):
    """Document pages belonging to one employee and batch period."""

    key: DocumentGroupKey
    source_pages: List[SourcePage] = Field(default_factory=list)
    completeness_status: CompletenessStatus = CompletenessStatus.AMBIGUOUS
    review_required: bool = False

    @model_validator(mode="after")
    def require_review_for_ambiguous_group(self) -> "DocumentGroup":
        """Never allow an ambiguous group to bypass the review workflow."""
        if self.completeness_status == CompletenessStatus.AMBIGUOUS:
            self.review_required = True
        return self


class FileClassification(str, Enum):
    """Classification state for discovered files during incremental planning."""

    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    REBUILD = "rebuild"
    COLLISION = "collision"
    UNSUPPORTED = "unsupported"
    NEEDS_REPROCESS = "needs_reprocess"


class FileProcessingStatus(str, Enum):
    """Processing lifecycle status of an image file."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class DetectionMetadata(BaseContract):
    """Minimal persisted detector provenance and quality summary."""

    detector_name: Optional[str] = None
    detector_mode: Optional[str] = None
    detector_model_version: Optional[str] = None
    detector_model_checksum: Optional[str] = None
    detection_status: Optional[Literal["detected", "fallback", "failed", "not_run"]] = None
    detection_fallback_used: bool = False
    detection_quality_summary: Dict[str, Union[str, int, float, bool, None]] = Field(
        default_factory=dict
    )


class ScannerErrorCode(str, Enum):
    """Canonical error codes for scanner operations."""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_INPUT_ROOT = "INVALID_INPUT_ROOT"
    OUTPUT_NOT_WRITABLE = "OUTPUT_NOT_WRITABLE"
    IMAGE_DECODE_FAILED = "IMAGE_DECODE_FAILED"
    PDF_WRITE_FAILED = "PDF_WRITE_FAILED"
    STATE_READ_FAILED = "STATE_READ_FAILED"
    STATE_WRITE_FAILED = "STATE_WRITE_FAILED"
    OUTPUT_COLLISION = "OUTPUT_COLLISION"
    GROUP_EXPORT_BLOCKED = "GROUP_EXPORT_BLOCKED"
    UNEXPECTED_ERROR = "UNEXPECTED_ERROR"


class ScannerWarningCode(str, Enum):
    """Canonical warning codes for scanner operations."""

    DOCUMENT_NOT_DETECTED = "DOCUMENT_NOT_DETECTED"
    IMAGE_DOWNSCALED = "IMAGE_DOWNSCALED"
    WARP_FALLBACK = "WARP_FALLBACK"
    DOCUMENT_CLIPPED = "DOCUMENT_CLIPPED"


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
    reprocess_reason: Optional[str] = None


class ScanPlan(BaseContract):
    """Overall summary plan before executing scan batch, matching System Design fields."""

    input_root: str
    output_root: str
    employees: int = 0
    total_images: int = 0
    new: int = 0
    modified: int = 0
    rebuild: int = 0
    unchanged: int = 0
    files_to_process: int = 0
    outdated_pipeline_count: int = 0
    unsupported_count: int = 0
    needs_reprocess: int = 0
    reprocess_reasons: Dict[str, int] = Field(default_factory=dict)
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
    def validate_process_count(self) -> "ScanPlan":
        """Keep the aggregate process count consistent with classifications."""
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


class ScanBatchRequest(BaseContract):
    """Parameters passed to initiate a scan batch."""

    input_root: str
    output_root: Optional[str] = None
    mode: ScanMode = ScanMode.GRAY
    workers: int = Field(default=3, ge=1, le=4)
    period: Optional[BatchPeriod] = None
    export_mode: ExportMode = ExportMode.PER_IMAGE
    manual_order: Dict[str, List[str]] = Field(default_factory=dict)
    skip_groups: List[str] = Field(default_factory=list)


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


class ScannerError(Exception):
    """Base exception for Attendance Scanner operations."""

    def __init__(self, code: ScannerErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class InvalidInputRootError(ScannerError):
    """Raised when the input root directory is non-existent or invalid."""

    def __init__(
        self, path: str, reason: str = "Path does not exist or is not a directory"
    ) -> None:
        super().__init__(
            ScannerErrorCode.INVALID_INPUT_ROOT,
            f"Invalid input root: '{path}' ({reason})",
        )
        self.path = path


class OutputNotWritableError(ScannerError):
    """Raised when the configured output directory cannot accept scanner output."""

    def __init__(self, path: str, reason: str = "Path is not a writable directory") -> None:
        super().__init__(
            ScannerErrorCode.OUTPUT_NOT_WRITABLE,
            f"Output root is not writable: '{path}' ({reason})",
        )
        self.path = path


class ImageDecodeError(ScannerError):
    """Raised when an image file fails to read or decode."""

    def __init__(self, path: str, reason: str = "Unable to decode image") -> None:
        super().__init__(
            ScannerErrorCode.IMAGE_DECODE_FAILED,
            f"Failed to decode image '{path}': {reason}",
        )
        self.path = path


class ImageProcessError(ScannerError):
    """Raised when an image processing pipeline stage fails."""

    def __init__(
        self,
        reason: str,
        path: Optional[str] = None,
        code: ScannerErrorCode = ScannerErrorCode.UNEXPECTED_ERROR,
    ) -> None:
        target = f" on '{path}'" if path else ""
        super().__init__(
            code,
            f"Image processing failed{target}: {reason}",
        )
        self.reason = reason
        self.path = path


class PdfWriteError(ScannerError):
    """Raised when PDF generation or filesystem commitment fails."""

    def __init__(
        self,
        path: str,
        reason: str = "Failed to write PDF",
        code: ScannerErrorCode = ScannerErrorCode.PDF_WRITE_FAILED,
    ) -> None:
        super().__init__(
            code,
            f"Failed to write PDF to '{path}': {reason}",
        )
        self.path = path
        self.reason = reason


class StateError(ScannerError):
    """Raised when manifest state reading, parsing, or writing fails."""

    def __init__(
        self,
        code: ScannerErrorCode,
        message: str,
        path: Optional[str] = None,
    ) -> None:
        super().__init__(code, message)
        self.path = path


class DiscoveryResult(BaseContract):
    """Inventory result of employee folder discovery."""

    input_root: str
    employees: List[str] = Field(default_factory=list)
    files: List[DiscoveredFile] = Field(default_factory=list)
    employee_count: int = 0
    image_count: int = 0
    unsupported_count: int = 0
    collisions: List[str] = Field(default_factory=list)
    outdated_pipeline_count: int = 0
    needs_reprocess_count: int = 0
    reprocess_reasons: Dict[str, int] = Field(default_factory=dict)

    @property
    def files_to_process(self) -> List[DiscoveredFile]:
        """Return the exact discovered files selected for processing."""
        processable = {
            FileClassification.NEW,
            FileClassification.MODIFIED,
            FileClassification.REBUILD,
        }
        return [file for file in self.files if file.classification in processable]

    def to_scan_plan(self, output_root: Optional[str] = None) -> ScanPlan:
        """Convert discovery result, including classifications, into a ScanPlan."""
        out_root = output_root or f"{self.input_root}_pdf"
        counts = {
            classification: sum(1 for file in self.files if file.classification == classification)
            for classification in FileClassification
        }
        files_to_process = (
            counts[FileClassification.NEW]
            + counts[FileClassification.MODIFIED]
            + counts[FileClassification.REBUILD]
        )
        return ScanPlan(
            input_root=self.input_root,
            output_root=out_root,
            employees=self.employee_count,
            total_images=self.image_count,
            new=counts[FileClassification.NEW],
            modified=counts[FileClassification.MODIFIED],
            rebuild=counts[FileClassification.REBUILD],
            unchanged=counts[FileClassification.UNCHANGED],
            files_to_process=files_to_process,
            outdated_pipeline_count=self.outdated_pipeline_count,
            unsupported_count=self.unsupported_count,
            needs_reprocess=self.needs_reprocess_count,
            reprocess_reasons=dict(self.reprocess_reasons),
            collisions=list(self.collisions),
        )
