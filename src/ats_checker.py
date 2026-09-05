from __future__ import annotations

import re

import fitz

from src.schemas import (
    AtsCheck,
    AtsReport,
    JobProfile,
    MatchAnalysis,
    PdfLayoutSignals,
)


def _has_overlapping_columns(page: fitz.Page) -> bool:
    midpoint = page.rect.width / 2
    blocks = [
        block
        for block in page.get_text("blocks")
        if len(block) >= 7 and block[6] == 0 and str(block[4]).strip()
    ]
    left = [block for block in blocks if block[2] <= midpoint * 1.16]
    right = [block for block in blocks if block[0] >= midpoint * 0.84]
    return any(
        max(left_block[1], right_block[1])
        < min(left_block[3], right_block[3]) - 6
        for left_block in left
        for right_block in right
    )


def inspect_pdf_layout(data: bytes) -> PdfLayoutSignals:
    """Extract non-sensitive PDF layout signals without retaining the source bytes."""
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception:
        return PdfLayoutSignals(readable=False)

    text_blocks = 0
    column_pages = 0
    tables = 0
    images = 0
    drawings = 0
    font_sizes: list[float] = []
    extracted_text: list[str] = []
    try:
        for page in document:
            extracted_text.append(page.get_text("text"))
            blocks = page.get_text("blocks")
            text_blocks += sum(
                len(block) >= 7 and block[6] == 0 and bool(str(block[4]).strip())
                for block in blocks
            )
            column_pages += int(_has_overlapping_columns(page))
            images += len(page.get_images(full=True))
            drawings += len(page.get_drawings())
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = span.get("size")
                        if size:
                            font_sizes.append(float(size))
            find_tables = getattr(page, "find_tables", None)
            if find_tables is not None:
                try:
                    tables += len(find_tables().tables)
                except Exception:
                    pass
        return PdfLayoutSignals(
            page_count=document.page_count,
            text_block_count=text_blocks,
            column_page_count=column_pages,
            table_count=tables,
            image_count=images,
            drawing_count=drawings,
            minimum_font_size=round(min(font_sizes), 1) if font_sizes else None,
            has_contact_details=contains_contact_details("\n".join(extracted_text)),
            readable=True,
        )
    except Exception:
        return PdfLayoutSignals(page_count=document.page_count, readable=False)
    finally:
        document.close()


def contains_contact_details(text: str) -> bool:
    email = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.IGNORECASE)
    phone = re.search(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)", text)
    return bool(email or phone)


def _section_count(text: str) -> int:
    groups = [
        ("工作经历", "实习经历", "experience", "employment"),
        ("教育经历", "教育背景", "education"),
        ("项目经历", "项目经验", "projects", "project experience"),
        ("技能", "专业技能", "skills", "technical skills"),
    ]
    lowered = text.casefold()
    return sum(any(keyword.casefold() in lowered for keyword in group) for group in groups)


def _date_style_count(text: str) -> int:
    patterns = [
        r"\b(?:19|20)\d{2}[/-](?:0?[1-9]|1[0-2])\b",
        r"(?:19|20)\d{2}\s*年\s*(?:0?[1-9]|1[0-2])\s*月",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+(?:19|20)\d{2}\b",
        r"\b(?:0?[1-9]|1[0-2])/(?:19|20)\d{2}\b",
    ]
    return sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns)


def build_ats_report(
    resume_text: str,
    job_profile: JobProfile,
    analysis: MatchAnalysis,
    layout: PdfLayoutSignals,
    resume_source: str,
) -> AtsReport:
    """Build a deterministic ATS-oriented heuristic report without a model call."""
    checks: list[AtsCheck] = []

    def add(
        code: str,
        severity: str,
        title: str,
        detail: str,
        recommendation: str | None = None,
    ) -> None:
        checks.append(
            AtsCheck(
                code=code,
                severity=severity,
                title=title,
                detail=detail,
                recommendation=recommendation,
            )
        )

    if resume_source != "pdf" or not layout.readable:
        add(
            "text_pdf",
            "critical",
            "PDF 可解析性",
            "当前文件无法作为稳定的文本型 PDF 被检查。",
            "导出文本型 PDF，并确认复制文字后不会出现乱码。",
        )
    else:
        add("text_pdf", "passed", "PDF 可解析性", "PDF 可以提取文本。")

    if layout.page_count > 3:
        add("page_count", "critical", "简历页数", f"当前共 {layout.page_count} 页。", "优先压缩到 1-2 页。")
    elif layout.page_count > 2:
        add("page_count", "warning", "简历页数", f"当前共 {layout.page_count} 页。", "若经验允许，建议控制在 2 页内。")
    else:
        add("page_count", "passed", "简历页数", f"当前共 {layout.page_count} 页。")

    if layout.column_page_count:
        add(
            "columns",
            "warning",
            "多栏版式",
            f"检测到 {layout.column_page_count} 页可能采用多栏布局。",
            "重要内容尽量使用单栏，避免 ATS 打乱阅读顺序。",
        )
    else:
        add("columns", "passed", "多栏版式", "未检测到明显的并排文本栏。")

    if layout.table_count:
        add("tables", "warning", "表格", f"检测到约 {layout.table_count} 个表格。", "避免用表格承载核心经历。")
    else:
        add("tables", "passed", "表格", "未检测到明显表格。")

    if layout.image_count > 2:
        add("images", "warning", "图片内容", f"PDF 包含 {layout.image_count} 个图片对象。", "不要把技能和经历只放在图片中。")
    else:
        add("images", "passed", "图片内容", f"检测到 {layout.image_count} 个图片对象。")

    if layout.minimum_font_size is not None and layout.minimum_font_size < 8.5:
        add("font_size", "warning", "字号", f"检测到最小字号约 {layout.minimum_font_size:g} pt。", "正文建议不小于 9-10 pt。")
    else:
        size = "未知" if layout.minimum_font_size is None else f"约 {layout.minimum_font_size:g} pt"
        add("font_size", "passed", "字号", f"最小字号为{size}。")

    contact_present = bool(layout.has_contact_details) or contains_contact_details(resume_text)
    if contact_present:
        add("contact", "passed", "联系方式", "检测到联系方式字段；具体内容不会在体检中显示。")
    else:
        add("contact", "warning", "联系方式", "未检测到常见电话或邮箱格式。", "投递前确认简历中包含可用联系方式。")

    sections = _section_count(resume_text)
    if sections < 2:
        add("sections", "warning", "标准板块", f"仅识别到 {sections} 类常见简历板块。", "使用清晰的教育、经历、项目和技能标题。")
    else:
        add("sections", "passed", "标准板块", f"识别到 {sections} 类常见简历板块。")

    date_styles = _date_style_count(resume_text)
    if date_styles > 1:
        add("dates", "warning", "日期格式", "检测到多种日期写法。", "统一使用 YYYY-MM 或 Month YYYY。")
    else:
        add("dates", "passed", "日期格式", "未检测到明显混用的日期格式。")

    long_lines = sum(len(line.strip()) > 180 for line in resume_text.splitlines())
    if long_lines:
        add("long_lines", "warning", "段落长度", f"有 {long_lines} 行超过 180 个字符。", "拆成简短、结果导向的要点。")
    else:
        add("long_lines", "passed", "段落长度", "未检测到过长文本行。")

    normalized_resume = re.sub(r"\s+", "", resume_text).casefold()
    requirements = job_profile.requirements
    covered = sum(
        re.sub(r"\s+", "", item.normalized_name).casefold() in normalized_resume
        for item in requirements
        if item.normalized_name.strip()
    )
    coverage = round(covered / len(requirements) * 100, 1) if requirements else 100.0
    if coverage < 40:
        add("keywords", "critical", "JD 关键词覆盖", f"原简历文字直接覆盖约 {coverage:.1f}% 的岗位要求关键词。", "只补充真实具备的关键词，并放入相关经历或技能板块。")
    elif coverage < 70:
        add("keywords", "warning", "JD 关键词覆盖", f"原简历文字直接覆盖约 {coverage:.1f}% 的岗位要求关键词。", "优先强化有证据但表述不明显的要求。")
    else:
        add("keywords", "passed", "JD 关键词覆盖", f"原简历文字直接覆盖约 {coverage:.1f}% 的岗位要求关键词。")

    # Keep the analysis argument part of the interface: it guarantees this report is
    # produced for the same validated requirement set as the visible match result.
    if {item.requirement_id for item in analysis.matches} != {item.id for item in requirements}:
        add("analysis_alignment", "critical", "分析一致性", "匹配结果与当前 JD 要求不一致。", "重新运行岗位分析。")
    else:
        add("analysis_alignment", "passed", "分析一致性", "匹配结果与当前 JD 要求一致。")

    deductions = sum(
        20 if item.severity == "critical" else 8 if item.severity == "warning" else 0
        for item in checks
    )
    return AtsReport(
        score=max(0, 100 - deductions),
        checks=checks,
        keyword_coverage=coverage,
        layout=layout,
    )
