"""Attendance Scanner Engine - Local document scanner for employee attendance sheets."""

from attendance_scanner.contracts import (
    PROTOCOL_VERSION,
    BatchSummary,
    DiscoveredFile,
    FileClassification,
    FileProcessingStatus,
    FileResult,
    ScanBatchRequest,
    ScanMode,
    ScannerErrorCode,
    ScanPlan,
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
    "DiscoveredFile",
    "ScanPlan",
    "ScanBatchRequest",
    "FileResult",
    "BatchSummary",
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
