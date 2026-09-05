import pytest
from pydantic import ValidationError

from src.validators import (
    MAX_PDF_SIZE_BYTES,
    InputValidationError,
    JobInput,
    has_valid_resume_text,
    select_resume_text,
    validate_pdf_upload,
)


VALID_RESUME = "A" * 50
VALID_JD = "岗位职责与任职要求" * 8


def test_accepts_valid_job_input_and_normalises_optional_fields() -> None:
    job = JobInput(
        company="  示例公司  ",
        job_title="  数据分析实习生 ",
        location=" ",
        job_type="",
        job_url=" https://jobs.example.com/role/123 ",
        jd_text=VALID_JD,
    )

    assert job.company == "示例公司"
    assert job.job_title == "数据分析实习生"
    assert job.location is None
    assert job.job_type is None
    assert job.job_url == "https://jobs.example.com/role/123"


def test_rejects_invalid_job_url() -> None:
    with pytest.raises(ValidationError):
        JobInput(
            company="示例公司",
            job_title="分析师",
            job_url="jobs.example.com/role/123",
            jd_text=VALID_JD,
        )


@pytest.mark.parametrize("field", ["company", "job_title"])
def test_rejects_missing_required_job_metadata(field: str) -> None:
    values = {"company": "示例公司", "job_title": "产品实习生", "jd_text": VALID_JD}
    values[field] = "   "

    with pytest.raises(ValidationError):
        JobInput(**values)


def test_rejects_short_jd_after_whitespace_is_removed() -> None:
    with pytest.raises(ValidationError):
        JobInput(company="示例公司", job_title="分析师", jd_text="要求：SQL  Python")


def test_validates_pdf_presence_extension_and_size() -> None:
    with pytest.raises(InputValidationError):
        validate_pdf_upload(None, None)
    with pytest.raises(InputValidationError):
        validate_pdf_upload("resume.txt", 100)
    with pytest.raises(InputValidationError):
        validate_pdf_upload("resume.pdf", MAX_PDF_SIZE_BYTES + 1)

    validate_pdf_upload("RESUME.PDF", MAX_PDF_SIZE_BYTES)


def test_resume_text_threshold_ignores_whitespace() -> None:
    assert has_valid_resume_text("A " * 50)
    assert not has_valid_resume_text("A " * 49)


def test_valid_pdf_always_wins_over_pasted_text() -> None:
    selected, source = select_resume_text("P" * 50, "T" * 80)

    assert selected == "P" * 50
    assert source == "pdf"


def test_valid_paste_is_used_only_when_pdf_is_invalid() -> None:
    selected, source = select_resume_text("too short", "T" * 50)

    assert selected == "T" * 50
    assert source == "pasted"


def test_rejects_when_both_resume_sources_are_invalid() -> None:
    with pytest.raises(InputValidationError):
        select_resume_text("short", "also short")


def test_switching_to_valid_pdf_ignores_previous_paste() -> None:
    first_text, first_source = select_resume_text("short", "T" * 50)
    second_text, second_source = select_resume_text("P" * 50, first_text)

    assert first_source == "pasted"
    assert second_source == "pdf"
    assert second_text == "P" * 50
