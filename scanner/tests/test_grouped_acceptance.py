"""AS-30 grouped-export and random-filename acceptance coverage."""

from pathlib import Path

import pytest
from PIL import Image, PdfParser

from attendance_scanner.contracts import (
    BatchPeriod,
    CompletenessStatus,
    ExportMode,
    PageType,
)
from attendance_scanner.discovery import build_group_aware_scan_plan, discover_employee_folders
from attendance_scanner.export import (
    ExportPage,
    ExportReviewRequiredError,
    export_grouped,
    export_per_image,
)
from attendance_scanner.state import ManifestStore


def _page(
    source: str,
    employee: str,
    page_type: PageType = PageType.UNKNOWN,
    page_order: int | None = None,
) -> ExportPage:
    return ExportPage(
        image=Image.new("L", (120, 80), color=160),
        source_relative_path=source,
        employee_name=employee,
        page_type=page_type,
        page_order=page_order,
        confidence=0.96 if page_order is not None else None,
        detection_method="acceptance-fixture",
    )


def test_random_names_and_reverse_discovery_produce_expected_per_image_and_grouped_artifacts(
    tmp_path: Path,
):
    pages = [
        _page(
            "NV01/hash-z9f2.png",
            "NV01",
            PageType.SECOND_HALF,
            2,
        ),
        _page(
            "NV01/hash-a1b7.png",
            "NV01",
            PageType.FIRST_HALF,
            1,
        ),
    ]
    period = BatchPeriod(year=2026, month=9)

    per_image = export_per_image(pages, tmp_path / "per-image", period)
    grouped = export_grouped(pages, tmp_path / "grouped", period)

    assert len(per_image) == 2
    assert all(artifact.pdf.page_count == 1 for artifact in per_image)
    assert len(grouped) == 1
    assert grouped[0].source_relative_paths == [
        "NV01/hash-a1b7.png",
        "NV01/hash-z9f2.png",
    ]
    assert grouped[0].output_path.name == "2026-09_NV01.pdf"
    parser = PdfParser.PdfParser(filename=str(grouped[0].output_path))
    assert len(parser.pages) == 2


def test_grouped_export_keeps_employees_and_periods_separate(tmp_path: Path):
    period = BatchPeriod(year=2026, month=9)
    pages = [
        _page("A/random-2.png", "A", PageType.SECOND_HALF, 2),
        _page("A/random-1.png", "A", PageType.FIRST_HALF, 1),
        _page("B/random-2.png", "B", PageType.SECOND_HALF, 2),
        _page("B/random-1.png", "B", PageType.FIRST_HALF, 1),
    ]

    artifacts = export_grouped(pages, tmp_path / "same-period", period)

    assert {artifact.output_path.name for artifact in artifacts} == {
        "2026-09_A.pdf",
        "2026-09_B.pdf",
    }
    assert all(
        {path.split("/", 1)[0] for path in artifact.source_relative_paths}
        == {artifact.output_path.parent.name}
        for artifact in artifacts
    )

    a_period_pages = [
        _page("A/aug-2-random.png", "A", PageType.SECOND_HALF, 2),
        _page("A/aug-1-random.png", "A", PageType.FIRST_HALF, 1),
    ]
    august = export_grouped(
        a_period_pages,
        tmp_path / "august",
        BatchPeriod(year=2026, month=8),
    )
    assert august[0].output_path.name == "2026-08_A.pdf"
    assert all("sep" not in path for path in august[0].source_relative_paths)


def test_unknown_and_duplicate_page_one_never_silently_export(tmp_path: Path):
    unknown = [_page("NV01/random-a.png", "NV01"), _page("NV01/random-b.png", "NV01")]
    with pytest.raises(ExportReviewRequiredError):
        export_grouped(unknown, tmp_path / "unknown", BatchPeriod(year=2026, month=9))

    duplicate_page_one = [
        _page("NV01/random-a.png", "NV01", PageType.FIRST_HALF, 1),
        _page("NV01/random-b.png", "NV01", PageType.FIRST_HALF, 1),
    ]
    with pytest.raises(ExportReviewRequiredError):
        export_grouped(
            duplicate_page_one,
            tmp_path / "duplicate",
            BatchPeriod(year=2026, month=9),
        )
    assert not list(tmp_path.rglob("*.pdf"))


def test_missing_page_two_is_incomplete_and_requires_review(tmp_path: Path):
    input_root = tmp_path / "employees"
    (input_root / "NV01").mkdir(parents=True)
    Image.new("RGB", (40, 30), color=(160, 160, 160)).save(input_root / "NV01/random-page.png")
    output_root = tmp_path / "output"
    manifest = ManifestStore(state_dir=tmp_path / "state").load_manifest(
        input_root, output_root=output_root
    )

    plan = build_group_aware_scan_plan(
        discover_employee_folders(input_root),
        manifest,
        output_root,
        BatchPeriod(year=2026, month=9),
        export_mode=ExportMode.GROUPED,
    )

    assert plan.affected_group_count == 1
    assert plan.affected_groups[0].completeness_status == CompletenessStatus.INCOMPLETE
    assert plan.affected_groups[0].review_required is True
    assert "missing_expected_page" in plan.affected_groups[0].reasons
    assert plan.review_groups[0].review_required is True
