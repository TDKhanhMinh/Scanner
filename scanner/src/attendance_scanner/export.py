"""Pluggable per-image and grouped PDF export strategies."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from attendance_scanner.contracts import (
    BatchPeriod,
    ExportMode,
    PageIdentity,
    PageType,
    SourcePage,
)
from attendance_scanner.page_classification import order_source_pages
from attendance_scanner.pdf_export import (
    ImageInput,
    PdfExportConfig,
    PdfExportResult,
    export_pdf_pages,
    export_single_page_pdf,
)


@dataclass(frozen=True)
class ExportPage:
    """One processed page with technical source and business identity metadata."""

    image: ImageInput
    source_relative_path: str
    employee_name: str
    identity: SourcePage = field(init=False)
    page_type: PageType = PageType.UNKNOWN
    page_order: Optional[int] = None
    confidence: Optional[float] = None
    detection_method: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity",
            SourcePage(
                source_relative_path=self.source_relative_path,
                identity=PageIdentity(
                    page_type=self.page_type,
                    page_order=self.page_order,
                    confidence=self.confidence,
                    detection_method=self.detection_method,
                ),
            ),
        )


@dataclass(frozen=True)
class ExportArtifact:
    """Metadata for one atomically committed output artifact."""

    output_path: Path
    page_count: int
    mode: ExportMode
    source_relative_paths: List[str]
    pdf: PdfExportResult


class ExportReviewRequiredError(ValueError):
    """Raised when GROUPED export cannot safely determine page order."""

    review_required = True


def sanitize_filename_component(value: str) -> str:
    """Make employee names safe for Windows filenames while preserving Unicode."""
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    if not sanitized:
        return "employee"
    if sanitized.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }:
        sanitized = f"_{sanitized}"
    return sanitized


def _short_source_fingerprint(source_relative_path: str) -> str:
    normalized = source_relative_path.replace("\\", "/").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()[:8]


def _base_per_image_name(page: ExportPage, period: BatchPeriod) -> str:
    employee = sanitize_filename_component(page.employee_name)
    if page.page_order is not None and page.page_type != PageType.UNKNOWN:
        return f"{period.year:04d}-{period.month:02d}_{employee}_p{page.page_order:02d}.pdf"
    fingerprint = _short_source_fingerprint(page.source_relative_path)
    return f"{period.year:04d}-{period.month:02d}_{employee}_{fingerprint}.pdf"


def _resolve_collision_free_names(
    pages: Sequence[ExportPage],
    period: BatchPeriod,
) -> Dict[str, str]:
    """Assign deterministic names without making input enumeration order observable."""
    candidates: Dict[str, List[ExportPage]] = {}
    for page in pages:
        candidates.setdefault(_base_per_image_name(page, period), []).append(page)

    names: Dict[str, str] = {}
    for base_name, collision_pages in candidates.items():
        if len(collision_pages) == 1:
            names[collision_pages[0].source_relative_path] = base_name
            continue
        for page in sorted(collision_pages, key=lambda item: item.source_relative_path):
            stem = Path(base_name).stem
            names[page.source_relative_path] = (
                f"{stem}_{_short_source_fingerprint(page.source_relative_path)}.pdf"
            )
    return names


def _output_directory(output_root: Path, employee_name: str) -> Path:
    return output_root / sanitize_filename_component(employee_name)


def export_per_image(
    pages: Sequence[ExportPage],
    output_root: Path,
    period: BatchPeriod,
    *,
    config: Optional[PdfExportConfig] = None,
) -> List[ExportArtifact]:
    """Export every processed page as one PDF with deterministic naming."""
    names = _resolve_collision_free_names(pages, period)
    artifacts: List[ExportArtifact] = []
    for page in pages:
        target = (
            _output_directory(output_root, page.employee_name) / names[page.source_relative_path]
        )
        pdf = export_single_page_pdf(page.image, target, config=config)
        artifacts.append(
            ExportArtifact(
                output_path=target,
                page_count=1,
                mode=ExportMode.PER_IMAGE,
                source_relative_paths=[page.source_relative_path],
                pdf=pdf,
            )
        )
    return artifacts


def export_grouped(
    pages: Sequence[ExportPage],
    output_root: Path,
    period: BatchPeriod,
    *,
    config: Optional[PdfExportConfig] = None,
    manual_order: bool = False,
    manual_orders: Optional[Dict[str, List[str]]] = None,
) -> List[ExportArtifact]:
    """Export one multi-page PDF per employee/period group."""
    groups: Dict[str, List[ExportPage]] = {}
    for page in pages:
        groups.setdefault(page.employee_name, []).append(page)

    artifacts: List[ExportArtifact] = []
    for employee_name, group_pages in groups.items():
        by_source = {page.source_relative_path: page for page in group_pages}
        group_key = f"{employee_name}:{period.year:04d}-{period.month:02d}"
        explicit_order = manual_orders.get(group_key) if manual_orders else None
        if explicit_order is not None:
            normalized_order = [path.replace("\\", "/") for path in explicit_order]
            if len(normalized_order) != len(set(normalized_order)) or set(normalized_order) != set(
                by_source
            ):
                raise ExportReviewRequiredError(
                    f"Manual page order does not match current sources for group {group_key!r}"
                )
            ordered_pages = [by_source[source] for source in normalized_order]
        else:
            decision = order_source_pages([page.identity for page in group_pages])
            if decision.review_required:
                raise ExportReviewRequiredError(
                    f"Grouped export requires page-order review for employee {employee_name!r}: "
                    f"{decision.reason}"
                )
            ordered_pages = [by_source[source.source_relative_path] for source in decision.ordered]
        target_name = (
            f"{period.year:04d}-{period.month:02d}_{sanitize_filename_component(employee_name)}.pdf"
        )
        target = _output_directory(output_root, employee_name) / target_name
        pdf = export_pdf_pages([page.image for page in ordered_pages], target, config=config)
        artifacts.append(
            ExportArtifact(
                output_path=target,
                page_count=len(ordered_pages),
                mode=ExportMode.GROUPED,
                source_relative_paths=[page.source_relative_path for page in ordered_pages],
                pdf=pdf,
            )
        )
    return artifacts


def export_pages(
    pages: Sequence[ExportPage],
    output_root: Path,
    period: BatchPeriod,
    mode: ExportMode = ExportMode.PER_IMAGE,
    *,
    config: Optional[PdfExportConfig] = None,
    manual_order: bool = False,
    manual_orders: Optional[Dict[str, List[str]]] = None,
) -> List[ExportArtifact]:
    """Dispatch to a pluggable export strategy without re-running scan pipeline work."""
    if mode == ExportMode.PER_IMAGE:
        return export_per_image(pages, output_root, period, config=config)
    if mode == ExportMode.GROUPED:
        return export_grouped(
            pages,
            output_root,
            period,
            config=config,
            manual_order=manual_order,
            manual_orders=manual_orders,
        )
    raise ValueError(f"Unsupported export mode: {mode!r}")
