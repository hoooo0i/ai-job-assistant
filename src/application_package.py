from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from src.reporting import (
    build_cover_letter_docx,
    build_cover_letter_pdf,
    build_docx_report,
    build_pdf_report,
    build_tailored_resume_docx,
)
from src.schemas import (
    CandidateFact,
    CoverLetterDraft,
    JobProfile,
    MatchAnalysis,
    ResumeProfile,
    ResumeVersion,
    ScoreResult,
)
from src.submission import build_submission_checklist


@dataclass(frozen=True)
class ApplicationPackageResult:
    data: bytes
    files: tuple[str, ...]
    warnings: tuple[str, ...]


def _checklist_text(bundle: dict, analysis: MatchAnalysis, job: JobProfile, facts: list[CandidateFact]) -> str:
    checklist = build_submission_checklist(
        analysis,
        bundle.get("resume_edits", {}),
        job,
        facts,
    )
    lines = [
        "投递前检查",
        "",
        f"目标岗位：{job.company} · {job.title}",
        f"总体状态：{'可以投递' if checklist.ready else '仍有项目需要核对'}",
        "",
    ]
    for item in checklist.items:
        lines.append(f"[{'通过' if item.passed else '需处理'}] {item.label}：{item.detail}")
    lines.extend(
        [
            "",
            "请在正式投递前再次核对所有经历、日期和数字。",
            "本检查不代表录取概率或 ATS 通过率。",
        ]
    )
    return "\n".join(lines)


def build_application_package(
    resume_profile: ResumeProfile,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
    score: ScoreResult,
    bundle: dict,
    candidate_facts: list[CandidateFact],
) -> ApplicationPackageResult:
    """Build a private in-memory ZIP from already generated, evidence-bound materials."""
    files: dict[str, bytes] = {
        "job-analysis-report.docx": build_docx_report(
            resume_profile,
            job_profile,
            analysis,
            score,
            bundle.get("interview_feedback", {}),
        ),
        "job-analysis-report.pdf": build_pdf_report(
            resume_profile,
            job_profile,
            analysis,
            score,
            bundle.get("interview_feedback", {}),
        ),
        "submission-checklist.txt": _checklist_text(
            bundle, analysis, job_profile, candidate_facts
        ).encode("utf-8"),
    }
    warnings: list[str] = []
    versions = [
        ResumeVersion.model_validate(item)
        for item in bundle.get("resume_versions", [])
    ]
    tracking = bundle.get("application_tracking", {})
    selected_id = tracking.get("resume_version_id")
    selected_version = next((item for item in versions if item.id == selected_id), None)
    if selected_version is None and versions:
        selected_version = versions[0]
        if selected_id:
            warnings.append("绑定的简历版本已不存在，材料包改用最新保存版本。")
    if selected_version:
        files["tailored-resume.docx"] = build_tailored_resume_docx(
            resume_profile,
            job_profile,
            selected_version.accepted_suggestions,
        )
    else:
        warnings.append("尚未保存简历版本，材料包未包含定制简历。")

    cover_letters = bundle.get("cover_letters", {})
    language = "zh" if cover_letters.get("zh") else "en" if cover_letters.get("en") else None
    if language:
        draft = CoverLetterDraft.model_validate(cover_letters[language]["draft"])
        files[f"cover-letter-{language}.docx"] = build_cover_letter_docx(job_profile, draft)
        files[f"cover-letter-{language}.pdf"] = build_cover_letter_pdf(job_profile, draft)
    else:
        warnings.append("尚未生成求职信，材料包未包含求职信。")

    manifest = [
        "AI 求职助手投递材料包",
        "",
        f"目标岗位：{job_profile.company} · {job_profile.title}",
        "包含文件：",
        *[f"- {name}" for name in sorted(files)],
        "",
        "隐私说明：文件在内存中生成，由浏览器下载；请妥善保管并在投递前核对。",
    ]
    if warnings:
        manifest.extend(["", "注意事项：", *[f"- {item}" for item in warnings]])
    files["README.txt"] = "\n".join(manifest).encode("utf-8")

    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for filename, data in files.items():
            archive.writestr(filename, data)
    return ApplicationPackageResult(
        data=output.getvalue(),
        files=tuple(sorted(files)),
        warnings=tuple(warnings),
    )
