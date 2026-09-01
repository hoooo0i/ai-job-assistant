from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator


MAX_PDF_SIZE_BYTES = 10 * 1024 * 1024
MIN_RESUME_CHARACTERS = 50
MIN_JD_CHARACTERS = 50


class InputValidationError(ValueError):
    """A validation error safe to display directly to the user."""


def meaningful_character_count(text: Optional[str]) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def has_valid_resume_text(text: Optional[str]) -> bool:
    return meaningful_character_count(text) >= MIN_RESUME_CHARACTERS


def validate_resume_text(text: Optional[str]) -> str:
    cleaned = (text or "").strip()
    if not has_valid_resume_text(cleaned):
        raise InputValidationError(
            f"简历内容至少需要 {MIN_RESUME_CHARACTERS} 个非空白字符。"
        )
    return cleaned


def validate_pdf_upload(filename: Optional[str], size: Optional[int]) -> None:
    if not filename:
        raise InputValidationError("请先上传一份 PDF 简历。")
    if Path(filename).suffix.lower() != ".pdf":
        raise InputValidationError("仅支持 PDF 格式的简历文件。")
    if size is None or size <= 0:
        raise InputValidationError("PDF 文件为空，请重新选择文件。")
    if size > MAX_PDF_SIZE_BYTES:
        raise InputValidationError("PDF 文件不能超过 10 MB。")


def select_resume_text(
    pdf_text: Optional[str], pasted_text: Optional[str]
) -> tuple[str, Literal["pdf", "pasted"]]:
    """Select exactly one source, always preferring valid PDF text."""
    if has_valid_resume_text(pdf_text):
        return (pdf_text or "").strip(), "pdf"
    if has_valid_resume_text(pasted_text):
        return (pasted_text or "").strip(), "pasted"
    raise InputValidationError(
        "PDF 无法提取有效文字，请粘贴至少 50 个非空白字符的简历内容。"
    )


class JobInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str
    job_title: str
    location: Optional[str] = None
    job_type: Optional[str] = None
    jd_text: str

    @field_validator("company", "job_title")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("此项为必填项。")
        return value

    @field_validator("location", "job_type", mode="before")
    @classmethod
    def normalise_optional_text(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("jd_text")
    @classmethod
    def validate_jd_length(cls, value: str) -> str:
        if meaningful_character_count(value) < MIN_JD_CHARACTERS:
            raise ValueError(f"至少需要 {MIN_JD_CHARACTERS} 个非空白字符。")
        return value
