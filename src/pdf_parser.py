from __future__ import annotations

import re

import pymupdf
from pydantic import BaseModel, ConfigDict


class PdfParserError(Exception):
    """Base class for safe, user-facing PDF parsing errors."""


class InvalidPdfError(PdfParserError):
    """Raised when uploaded bytes are not a valid PDF."""


class EncryptedPdfError(PdfParserError):
    """Raised when a PDF requires a password."""


class PdfReadError(PdfParserError):
    """Raised when text cannot be read from an otherwise valid PDF."""


class PdfExtractionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    filename: str
    page_count: int
    text: str


def _normalise_extracted_text(page_texts: list[str]) -> str:
    pages: list[str] = []
    for text in page_texts:
        lines = [line.rstrip() for line in text.replace("\x00", "").splitlines()]
        page = "\n".join(lines).strip()
        if page:
            pages.append(page)
    combined = "\n\n".join(pages)
    return re.sub(r"\n{3,}", "\n\n", combined).strip()


def extract_pdf_text(data: bytes, filename: str) -> PdfExtractionResult:
    """Extract text from a PDF held in memory without writing the file to disk."""
    if not data or not data.lstrip().startswith(b"%PDF-"):
        raise InvalidPdfError("文件不是有效的 PDF，请重新选择 PDF 文件。")

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError("PDF 文件已损坏或格式无法识别，请重新导出后上传。") from exc

    try:
        if document.needs_pass:
            raise EncryptedPdfError("PDF 已加密或受密码保护，请上传未加密版本。")

        page_count = document.page_count
        page_texts = [document.load_page(index).get_text("text") for index in range(page_count)]
    except EncryptedPdfError:
        raise
    except Exception as exc:
        raise PdfReadError("读取 PDF 时发生错误，请重新导出或粘贴简历内容。") from exc
    finally:
        document.close()

    return PdfExtractionResult(
        filename=filename,
        page_count=page_count,
        text=_normalise_extracted_text(page_texts),
    )
