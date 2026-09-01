from src.privacy import redact_sensitive_info


def test_redacts_email_addresses() -> None:
    result = redact_sensitive_info("Email: sample.person@example.test. Please contact me.")

    assert "sample.person@example.test" not in result
    assert "[已隐藏邮箱]" in result


def test_redacts_chinese_and_international_phone_numbers() -> None:
    text = "手机：138 0013 8000\nMobile: +61 412 345 678"
    result = redact_sensitive_info(text)

    assert "138 0013 8000" not in result
    assert "+61 412 345 678" not in result
    assert result.count("[已隐藏电话]") == 2


def test_redacts_labelled_and_street_addresses() -> None:
    text = "地址：示例市示例路88号2室\nUnit 3, 25 Example Road"
    result = redact_sensitive_info(text)

    assert "示例路" not in result
    assert "Example Road" not in result
    assert result.count("[已隐藏详细地址]") == 2


def test_preserves_non_sensitive_resume_content_and_short_numbers() -> None:
    text = "使用 Python 完成 2025 年项目，准确率提升 12%。"

    assert redact_sensitive_info(text) == text
