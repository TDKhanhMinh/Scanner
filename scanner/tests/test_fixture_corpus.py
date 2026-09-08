"""Representative synthetic fixture regression tests (AS-20)."""

from pathlib import Path

import pytest
from PIL import PdfParser

from attendance_scanner.contracts import ImageDecodeError, ScannerErrorCode
from attendance_scanner.pdf_export import export_single_page_pdf
from attendance_scanner.pipeline.orchestrator import scan_one
from tests.fixture_corpus import FixtureCase, create_fixture_corpus, fixture_cases


@pytest.fixture(scope="module")
def fixture_corpus(
    tmp_path_factory: pytest.TempPathFactory,
) -> list[tuple[FixtureCase, Path]]:
    return create_fixture_corpus(tmp_path_factory.mktemp("scanner-fixtures"))


def test_fixture_manifest_is_complete_and_privacy_safe(fixture_corpus):
    keys = [case.key for case, _ in fixture_corpus]
    assert keys == [case.key for case in fixture_cases()]
    assert "real" not in " ".join(keys).lower()
    assert any("Nguyễn" in case.relative_path for case, _ in fixture_corpus)


@pytest.mark.parametrize("case_index", range(13))
def test_representative_fixture_has_expected_scan_outcome(
    fixture_corpus, case_index: int, tmp_path: Path
):
    case, source_path = fixture_corpus[case_index]

    if case.expected_error_code is not None:
        with pytest.raises(ImageDecodeError) as exc_info:
            scan_one(source_path)
        assert exc_info.value.code == ScannerErrorCode.IMAGE_DECODE_FAILED
        return

    result = scan_one(source_path)
    assert result.document_detected is case.expected_document_detected
    if not case.expected_document_detected:
        assert result.warning is not None
        assert "DOCUMENT_NOT_DETECTED" in result.warning

    output_path = tmp_path / f"{case.key}.pdf"
    export_single_page_pdf(result, output_path)
    assert output_path.stat().st_size > 100
    parser = PdfParser.PdfParser(filename=str(output_path))
    assert len(parser.pages) == 1
