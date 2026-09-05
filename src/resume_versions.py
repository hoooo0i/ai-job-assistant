from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from src.career_tools import contains_unresolved_placeholder
from src.privacy import redact_sensitive_info
from src.schemas import ResumeEditDecision, ResumeVersion


class ResumeVersionError(ValueError):
    """Raised when a resume version cannot be saved or restored."""


MAX_RESUME_VERSIONS = 10


def create_resume_version(
    job_id: str,
    label: str,
    decisions: dict[str, dict[str, str]],
    accepted_suggestions: list[tuple[str, str]],
    *,
    created_at: str | None = None,
) -> ResumeVersion:
    clean_label = redact_sensitive_info(label).strip()
    if not clean_label:
        raise ResumeVersionError("请填写版本名称。")
    if len(clean_label) > 60:
        raise ResumeVersionError("版本名称不能超过 60 个字符。")
    if not accepted_suggestions:
        raise ResumeVersionError("至少采纳一条有效建议后才能保存版本。")
    if any(contains_unresolved_placeholder(text) for _, text in accepted_suggestions):
        raise ResumeVersionError("版本中仍有未填写的占位符，请先补充或取消采纳。")

    validated_decisions = {
        key: ResumeEditDecision.model_validate(value)
        for key, value in decisions.items()
    }
    timestamp = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    identity_payload = {
        "job_id": job_id,
        "created_at": timestamp,
        "label": clean_label,
        "accepted_suggestions": accepted_suggestions,
    }
    version_id = hashlib.sha256(
        json.dumps(identity_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return ResumeVersion(
        id=version_id,
        label=clean_label,
        created_at=timestamp,
        decisions=validated_decisions,
        accepted_suggestions=accepted_suggestions,
    )


def add_resume_version(
    versions: list[dict],
    version: ResumeVersion,
    *,
    maximum: int = MAX_RESUME_VERSIONS,
) -> list[dict]:
    validated = [ResumeVersion.model_validate(item) for item in versions]
    if len(validated) >= maximum:
        raise ResumeVersionError(f"每个岗位最多保留 {maximum} 个版本，请先删除旧版本。")
    if any(item.id == version.id for item in validated):
        raise ResumeVersionError("这个版本已经保存，请稍后再试。")
    return [version.model_dump(mode="json"), *[item.model_dump(mode="json") for item in validated]]


def restore_resume_decisions(version: dict) -> dict[str, dict[str, str]]:
    validated = ResumeVersion.model_validate(version)
    return {
        key: decision.model_dump(mode="json")
        for key, decision in validated.decisions.items()
    }
