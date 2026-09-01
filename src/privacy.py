from __future__ import annotations

import re


EMAIL_PATTERN = re.compile(
    r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w-])",
    re.IGNORECASE,
)

PHONE_CANDIDATE_PATTERN = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[\s().-]*)?(?:\d[\s().-]*){7,14}\d(?!\w)"
)

ADDRESS_LABEL_PATTERN = re.compile(
    r"^\s*(?:通讯地址|家庭住址|居住地址|联系地址|地址|住址|"
    r"home\s+address|residential\s+address|mailing\s+address|address)\s*[:：]",
    re.IGNORECASE,
)

STREET_ADDRESS_PATTERN = re.compile(
    r"(?:\d+\s*(?:号|栋|幢|单元|室)|"
    r"(?:路|街|道|巷|弄)\s*\d+\s*号|"
    r"\b\d+[A-Z]?(?:[-/]\d+)?\s+[\w.'-]+(?:\s+[\w.'-]+){0,4}\s+"
    r"(?:Road|Rd|Street|St|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Unit)\b)",
    re.IGNORECASE,
)


def _redact_phone_candidate(match: re.Match[str]) -> str:
    value = match.group(0)
    digit_count = sum(character.isdigit() for character in value)
    return "[已隐藏电话]" if 8 <= digit_count <= 15 else value


def redact_sensitive_info(text: str) -> str:
    """Redact common contact details from text intended for UI previews."""
    if not text:
        return ""

    redacted_lines: list[str] = []
    for line in text.splitlines():
        if ADDRESS_LABEL_PATTERN.search(line) or STREET_ADDRESS_PATTERN.search(line):
            redacted_lines.append("[已隐藏详细地址]")
            continue

        line = EMAIL_PATTERN.sub("[已隐藏邮箱]", line)
        line = PHONE_CANDIDATE_PATTERN.sub(_redact_phone_candidate, line)
        redacted_lines.append(line)

    return "\n".join(redacted_lines)
