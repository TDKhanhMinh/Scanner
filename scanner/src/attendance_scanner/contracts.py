"""Shared contracts and type definitions for Attendance Scanner."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class ScanMode(str, Enum):
    """Scan enhancement filter mode."""
    GRAY = "gray"
    BW = "bw"
    COLOR = "color"


class FileState(str, Enum):
    """Classification state for discovered files."""
    NEW = "new"
    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    REBUILD = "rebuild"


class DiscoveredFile(BaseModel):
    """Metadata of a discovered image file."""
    employee_name: str
    file_name: str
    relative_path: str
    absolute_path: str
    size: int
    mtime_ns: int
    sha256: Optional[str] = None
    state: FileState = FileState.NEW
    target_relative_pdf: str


class ScanPlan(BaseModel):
    """Overall summary plan before executing scan batch."""
    input_root: str
    output_root: str
    total_employees: int = 0
    total_images: int = 0
    new_count: int = 0
    modified_count: int = 0
    unchanged_count: int = 0
    files_to_process: int = 0
    collisions: list[str] = Field(default_factory=list)
