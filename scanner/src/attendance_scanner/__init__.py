"""Attendance Scanner Engine - Local document scanner for employee attendance sheets."""

from attendance_scanner.contracts import (
    PROTOCOL_VERSION,
    BatchSummary,
    DiscoveredFile,
    DiscoveryResult,
    FileClassification,
    FileProcessingStatus,
    FileResult,
    InvalidInputRootError,
    ScanBatchRequest,
    ScanMode,
    ScannerError,
    ScannerErrorCode,
    ScanPlan,
)
from attendance_scanner.discovery import (
    discover_employee_folders,
    natural_sort_key,
    resolve_target_pdf,
)
from attendance_scanner.events import (
    BaseEvent,
    FileCompletedEvent,
    FileFailedEvent,
    FileStartedEvent,
    ScanCompletedEvent,
    ScannerEvent,
    ScanPlanEvent,
    deserialize_event,
    serialize_event,
)

__version__ = "0.1.0"

__all__ = [
    "PROTOCOL_VERSION",
    "ScanMode",
    "FileClassification",
    "FileProcessingStatus",
    "ScannerErrorCode",
    "ScannerError",
    "InvalidInputRootError",
    "DiscoveredFile",
    "DiscoveryResult",
    "ScanPlan",
    "ScanBatchRequest",
    "FileResult",
    "BatchSummary",
    "discover_employee_folders",
    "natural_sort_key",
    "resolve_target_pdf",
    "BaseEvent",
    "ScanPlanEvent",
    "FileStartedEvent",
    "FileCompletedEvent",
    "FileFailedEvent",
    "ScanCompletedEvent",
    "ScannerEvent",
    "serialize_event",
    "deserialize_event",
]
