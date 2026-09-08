"""AS-25 per-image and grouped export strategy tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, PdfParser

from attendance_scanner.contracts import BatchPeriod, PageType
from attendance_scanner.export import (
    ExportPage,
    ExportReviewRequiredError,
    export_grouped,
    export_per_image,
    sanitize_filename_component,
)
from attendance_scanner.pdf_export import export_pdf_pages


def _page(
    source: str,
    employee: str = "Nguyễn Văn A",
    page_type: PageType = PageType.UNKNOWN,
    page_order: int | None = None,
    color: int = 180,
) -> ExportPage:
    return ExportPage(
        image=Image.new("L", (120, 80), color=color),
        source_relative_path=source,
        employee_name=employee,
        page_type=page_type,
        page_order=page_order,
        confidence=0.95 if page_order is not None else None,
        detection_method="fixture",
    )


def test_per_image_names_known_pages_and_sanitizes_unicode_windows_names(tmp_path: Path):
    pages = [
        _page("NV01/random-a.png", page_type=PageType.FIRST_HALF, page_order=1),
        _page("NV01/random-b.png", employee="A/B:*?", page_type=PageType.SECOND_HALF, page_order=2),
    ]

    artifacts = export_per_image(pages, tmp_path, BatchPeriod(year=2026, month=9))

    assert artifacts[0].output_path.name == "2026-09_Nguyễn Văn A_p01.pdf"
    assert artifacts[1].output_path.name == "2026-09_A_B____p02.pdf"
    assert all(artifact.pdf.page_count == 1 for artifact in artifacts)
    assert all(artifact.output_path.stat().st_size > 100 for artifact in artifacts)
    assert sanitize_filename_component("CON") == "_CON"


def test_per_image_unknown_page_collision_names_are_order_independent(tmp_path: Path):
    pages = [_page("NV01/hash-z.png"), _page("NV01/hash-a.png")]
    period = BatchPeriod(year=2026, month=9)

    forward = export_per_image(pages, tmp_path / "forward", period)
    reverse = export_per_image(list(reversed(pages)), tmp_path / "reverse", period)
    forward_names = {
        artifact.source_relative_paths[0]: artifact.output_path.name for artifact in forward
    }
    reverse_names = {
        artifact.source_relative_paths[0]: artifact.output_path.name for artifact in reverse
    }

    assert forward_names == reverse_names
    assert all("p01" not in name for name in forward_names.values())


def test_grouped_export_orders_page_two_after_page_one_and_writes_one_two_page_pdf(
    tmp_path: Path,
):
    pages = [
        _page("NV01/page-two-random.png", page_type=PageType.SECOND_HALF, page_order=2),
        _page("NV01/page-one-random.png", page_type=PageType.FIRST_HALF, page_order=1),
    ]

    artifacts = export_grouped(pages, tmp_path, BatchPeriod(year=2026, month=9))

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.output_path.name == "2026-09_Nguyễn Văn A.pdf"
    assert artifact.source_relative_paths == [
        "NV01/page-one-random.png",
        "NV01/page-two-random.png",
    ]
    parser = PdfParser.PdfParser(filename=str(artifact.output_path))
    assert len(parser.pages) == 2


def test_grouped_export_refuses_ambiguous_order_without_manual_override(tmp_path: Path):
    pages = [_page("NV01/a.png"), _page("NV01/b.png")]

    with pytest.raises(ExportReviewRequiredError, match="requires page-order review"):
        export_grouped(pages, tmp_path, BatchPeriod(year=2026, month=9))
    assert not list(tmp_path.rglob("*.pdf"))


def test_grouped_export_manual_order_can_export_observed_order(tmp_path: Path):
    pages = [_page("NV01/a.png"), _page("NV01/b.png")]

    artifacts = export_grouped(
        pages,
        tmp_path,
        BatchPeriod(year=2026, month=9),
        manual_order=True,
    )

    assert artifacts[0].source_relative_paths == ["NV01/a.png", "NV01/b.png"]
    assert artifacts[0].pdf.page_count == 2


def test_grouped_export_uses_explicit_review_order_for_unknown_pages(tmp_path: Path):
    pages = [
        _page("NV01/page-one.png", employee="NV01"),
        _page("NV01/page-two.png", employee="NV01"),
    ]

    artifacts = export_grouped(
        pages,
        tmp_path,
        BatchPeriod(year=2026, month=9),
        manual_orders={
            "NV01:2026-09": ["NV01/page-two.png", "NV01/page-one.png"],
        },
    )

    assert artifacts[0].source_relative_paths == [
        "NV01/page-two.png",
        "NV01/page-one.png",
    ]
    assert artifacts[0].pdf.page_count == 2


def test_grouped_export_rejects_explicit_order_with_missing_source(tmp_path: Path):
    pages = [
        _page("NV01/page-one.png", employee="NV01"),
        _page("NV01/page-two.png", employee="NV01"),
    ]

    with pytest.raises(ExportReviewRequiredError, match="does not match current sources"):
        export_grouped(
            pages,
            tmp_path,
            BatchPeriod(year=2026, month=9),
            manual_orders={"NV01:2026-09": ["NV01/page-one.png"]},
        )
    assert not list(tmp_path.rglob("*.pdf"))


def test_multi_page_export_replaces_existing_output_atomically(tmp_path: Path):
    target = tmp_path / "existing.pdf"
    target.write_bytes(b"old artifact")

    result = export_pdf_pages(
        [np.full((80, 120), 100, dtype=np.uint8), np.full((60, 120), 200, dtype=np.uint8)],
        target,
    )

    assert result.page_count == 2
    assert target.read_bytes().startswith(b"%PDF-")
    assert not list(tmp_path.glob(".*.tmp.*"))
