import pymupdf
import pytest

from src.pdf_parser import EncryptedPdfError, InvalidPdfError, extract_pdf_text


def make_pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_extracts_text_and_page_count_from_memory() -> None:
    result = extract_pdf_text(make_pdf("Education and projects", "Skills and experience"), "cv.pdf")

    assert result.filename == "cv.pdf"
    assert result.page_count == 2
    assert "Education and projects" in result.text
    assert "Skills and experience" in result.text


def test_blank_pdf_returns_empty_text() -> None:
    result = extract_pdf_text(make_pdf(""), "blank.pdf")

    assert result.page_count == 1
    assert result.text == ""


def test_short_pdf_text_is_returned_for_separate_validation() -> None:
    result = extract_pdf_text(make_pdf("Short text"), "short.pdf")

    assert result.text == "Short text"


@pytest.mark.parametrize("data", [b"", b"not a pdf", b"PK\x03\x04fake archive"])
def test_rejects_non_pdf_bytes(data: bytes) -> None:
    with pytest.raises(InvalidPdfError):
        extract_pdf_text(data, "invalid.pdf")


def test_rejects_damaged_pdf() -> None:
    with pytest.raises(InvalidPdfError):
        extract_pdf_text(b"%PDF-1.7\nthis is incomplete", "damaged.pdf")


def test_rejects_encrypted_pdf() -> None:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Protected document")
    data = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-password",
        user_pw="user-password",
    )
    document.close()

    with pytest.raises(EncryptedPdfError):
        extract_pdf_text(data, "protected.pdf")
