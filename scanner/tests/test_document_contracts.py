"""AS-23 shared period/export/document-group contract tests."""

import json

import pytest
from pydantic import ValidationError

from attendance_scanner.contracts import (
    BatchPeriod,
    CompletenessStatus,
    DocumentGroup,
    DocumentGroupKey,
    ExportMode,
    PageIdentity,
    PageType,
    ScanBatchRequest,
    ScanMode,
    SourcePage,
)


def test_period_and_export_mode_round_trip_with_camel_case_json():
    request = ScanBatchRequest(
        input_root="D:/Attendance Input",
        output_root="D:/Attendance Output",
        mode=ScanMode.GRAY,
        workers=2,
        period=BatchPeriod(year=2026, month=9),
        export_mode=ExportMode.GROUPED,
    )

    payload = json.loads(request.model_dump_json(by_alias=True))
    restored = ScanBatchRequest.model_validate_json(json.dumps(payload))

    assert payload["exportMode"] == "GROUPED"
    assert payload["period"] == {"year": 2026, "month": 9}
    assert restored.period == BatchPeriod(year=2026, month=9)
    assert restored.export_mode == ExportMode.GROUPED


@pytest.mark.parametrize(
    ("year", "month"),
    [(0, 1), (10000, 1), (2026, 0), (2026, 13)],
)
def test_invalid_period_values_are_rejected(year: int, month: int):
    with pytest.raises(ValidationError):
        BatchPeriod(year=year, month=month)


def test_page_identity_supports_known_roles_and_explicit_unknown_role():
    unknown = PageIdentity(page_type=PageType.UNKNOWN, detection_method="manual_review")
    first_half = PageIdentity(page_type=PageType.FIRST_HALF, page_order=1, confidence=0.98)

    assert unknown.model_dump(by_alias=True)["pageType"] == "UNKNOWN"
    assert first_half.page_order == 1
    with pytest.raises(ValidationError):
        PageIdentity.model_validate({"pageType": "THIRD_PAGE"})


def test_document_group_round_trip_and_ambiguous_groups_require_review():
    key = DocumentGroupKey(employee_relative_dir="NV01", year=2026, month=9)
    pages = [
        SourcePage(
            source_relative_path="NV01/random-a.png",
            identity=PageIdentity(page_type=PageType.FIRST_HALF, page_order=1),
        ),
        SourcePage(
            source_relative_path="NV01/random-b.png",
            identity=PageIdentity(page_type=PageType.SECOND_HALF, page_order=2),
        ),
    ]

    complete = DocumentGroup(
        key=key,
        source_pages=pages,
        completeness_status=CompletenessStatus.COMPLETE,
        review_required=False,
    )
    incomplete = DocumentGroup(
        key=key,
        source_pages=pages[:1],
        completeness_status=CompletenessStatus.INCOMPLETE,
        review_required=False,
    )
    ambiguous = DocumentGroup(
        key=key,
        source_pages=pages,
        completeness_status=CompletenessStatus.AMBIGUOUS,
        review_required=False,
    )
    restored = DocumentGroup.model_validate_json(complete.model_dump_json(by_alias=True))

    assert restored == complete
    assert incomplete.completeness_status == CompletenessStatus.INCOMPLETE
    assert ambiguous.review_required is True
    assert ambiguous.completeness_status == CompletenessStatus.AMBIGUOUS
