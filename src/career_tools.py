from __future__ import annotations

import difflib
import html
import re
from dataclasses import dataclass

from src.privacy import redact_sensitive_info
from src.schemas import JobProfile, MatchAnalysis, MatchStatus, ResumeProfile


@dataclass(frozen=True)
class EvidenceRecord:
    id: str
    source: str
    text: str


@dataclass(frozen=True)
class KeywordGap:
    requirement_id: str
    keyword: str
    status: MatchStatus
    explanation: str


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def collect_application_evidence(
    resume_profile: ResumeProfile,
    analysis: MatchAnalysis,
    limit: int = 16,
) -> list[EvidenceRecord]:
    """Collect redacted, deduplicated evidence for application documents."""
    candidates: list[tuple[str, str]] = []
    for match in analysis.matches:
        if match.status in {MatchStatus.matched, MatchStatus.partial}:
            if match.evidence:
                candidates.extend(
                    (
                        "用户确认"
                        if item.source.value == "user_confirmed"
                        else f"简历·岗位要求 {match.requirement_id}",
                        item.text,
                    )
                    for item in match.evidence
                )
            else:
                candidates.extend(
                    (f"简历·岗位要求 {match.requirement_id}", text)
                    for text in match.resume_evidence
                )
    candidates.extend(
        (item.source_section or "简历", item.text)
        for item in resume_profile.evidence_chunks
    )

    seen: set[str] = set()
    records: list[EvidenceRecord] = []
    for source, raw_text in candidates:
        text = redact_sensitive_info(raw_text).strip()
        key = _normalise(text)
        if not key or key in seen:
            continue
        seen.add(key)
        records.append(
            EvidenceRecord(
                id=f"ev_{len(records) + 1:03d}",
                source=source,
                text=text,
            )
        )
        if len(records) >= limit:
            break
    return records


def analyse_keyword_gaps(
    job_profile: JobProfile,
    analysis: MatchAnalysis,
) -> list[KeywordGap]:
    """Map every JD requirement to its evidence status without another model call."""
    matches = {item.requirement_id: item for item in analysis.matches}
    gaps: list[KeywordGap] = []
    for requirement in job_profile.requirements:
        match = matches[requirement.id]
        gaps.append(
            KeywordGap(
                requirement_id=requirement.id,
                keyword=requirement.normalized_name,
                status=match.status,
                explanation=match.explanation,
            )
        )
    return gaps


def accepted_resume_suggestions(
    analysis: MatchAnalysis,
    decisions: dict[str, dict[str, str]],
) -> list[tuple[str, str]]:
    """Return only explicitly accepted, non-empty suggestion rewrites."""
    accepted: list[tuple[str, str]] = []
    for index, suggestion in enumerate(analysis.resume_suggestions):
        record = decisions.get(str(index), {})
        if record.get("decision") != "accepted":
            continue
        text = redact_sensitive_info(record.get("text", "")).strip()
        if text and not contains_unresolved_placeholder(text):
            accepted.append((suggestion.original_text, text))
    return accepted


def contains_unresolved_placeholder(text: str) -> bool:
    """Detect editor-only placeholders that must never enter an exported resume."""
    patterns = [
        r"\[[^\]]+\]",
        r"【[^】]+】",
        r"<[^>]+>",
        r"（请补充[^）]*）",
        r"请补充(?:具体)?(?:项目|情境|行动|结果|数据|经历|内容)",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def render_resume_diff_html(original: str, suggested: str) -> str:
    """Return an escaped, token-level HTML diff suitable for Streamlit."""
    token_pattern = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|\s+|[^\w\s]")
    before = token_pattern.findall(original)
    after = token_pattern.findall(suggested)
    parts: list[str] = []
    for operation, before_start, before_end, after_start, after_end in difflib.SequenceMatcher(
        None,
        before,
        after,
    ).get_opcodes():
        old_text = html.escape("".join(before[before_start:before_end]))
        new_text = html.escape("".join(after[after_start:after_end]))
        if operation == "equal":
            parts.append(old_text)
        elif operation == "delete":
            parts.append(f"<del>{old_text}</del>")
        elif operation == "insert":
            parts.append(f"<mark>{new_text}</mark>")
        else:
            parts.append(f"<del>{old_text}</del><mark>{new_text}</mark>")
    return "".join(parts)
