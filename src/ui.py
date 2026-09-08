from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import altair as alt
import streamlit as st


ThemeMode = Literal["light", "dark"]

NAV_GROUPS = ("岗位", "简历", "投递", "面试", "更多")

DETAIL_SECTIONS: dict[str, tuple[str, ...]] = {
    "岗位": ("岗位匹配", "关键词缺口"),
    "简历": ("简历优化", "ATS 体检", "简历结构"),
    "投递": ("投递管理", "材料包", "求职信", "报告下载"),
    "面试": ("面试辅助",),
    "更多": ("JD 结构",),
}

DEFAULT_DETAIL_SECTION = {
    "岗位": "岗位匹配",
    "简历": "简历优化",
    "投递": "投递管理",
    "面试": "面试辅助",
    "更多": "JD 结构",
}

_THEME_TOKENS: dict[ThemeMode, dict[str, str]] = {
    "light": {
        "page": "#EAEBED",
        "canvas": "#FBFAF7",
        "surface": "#FFFFFF",
        "surface_muted": "#F4F3EF",
        "text": "#17181B",
        "muted": "#72767F",
        "line": "#E5E2DB",
        "accent": "#F2C84B",
        "accent_soft": "#FFF4C8",
        "success": "#5A9F76",
        "danger": "#C55B52",
        "primary": "#15171C",
        "primary_text": "#FFFFFF",
        "shadow": "rgba(28, 30, 34, 0.08)",
    },
    "dark": {
        "page": "#101216",
        "canvas": "#17191E",
        "surface": "#20232A",
        "surface_muted": "#252931",
        "text": "#F5F3EE",
        "muted": "#A4A8B1",
        "line": "#30343C",
        "accent": "#E0B84E",
        "accent_soft": "#3A321C",
        "success": "#6FC494",
        "danger": "#EE8178",
        "primary": "#F5F3EE",
        "primary_text": "#15171C",
        "shadow": "rgba(0, 0, 0, 0.24)",
    },
}


def initialise_ui_state() -> None:
    """Initialise presentation-only state without touching analysis data."""
    query_theme = st.query_params.get("theme")
    default_theme: ThemeMode = "dark" if query_theme == "dark" else "light"
    st.session_state.setdefault("ui_theme", default_theme)
    st.session_state.setdefault(
        "ui_theme_dark", st.session_state.get("ui_theme") == "dark"
    )
    st.session_state.setdefault("workspace_section", "岗位")
    st.session_state.setdefault("comparison_selected_job_id", None)


def clear_session_preserving_ui_preferences() -> None:
    """Clear analysis data while retaining the user's presentation choices."""
    theme: ThemeMode = (
        "dark"
        if st.session_state.get("ui_theme_dark")
        or st.session_state.get("ui_theme") == "dark"
        else "light"
    )
    st.session_state.clear()
    st.session_state["ui_theme"] = theme
    st.session_state["ui_theme_dark"] = theme == "dark"
    st.query_params["theme"] = theme


def theme_tokens(theme: ThemeMode) -> dict[str, str]:
    """Return a copy of the selected palette for rendering and tests."""
    return dict(_THEME_TOKENS[theme])


def apply_design_system() -> None:
    """Inject the static design system with session-selected CSS variables."""
    theme: ThemeMode = (
        "dark" if st.session_state.get("ui_theme") == "dark" else "light"
    )
    tokens = theme_tokens(theme)
    variables = "\n".join(
        f"  --ui-{name.replace('_', '-')}: {value};" for name, value in tokens.items()
    )
    css_path = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
    css = css_path.read_text(encoding="utf-8")
    st.html(f"<style>:root {{\n{variables}\n}}\n{css}</style>")


def _set_navigation_value(state_key: str, value: str) -> None:
    st.session_state[state_key] = value


def render_pill_navigation(
    options: Sequence[str],
    *,
    state_key: str,
    default: str,
    container_key: str,
    disabled: bool = False,
) -> str:
    """Render stable, themeable navigation without fragile component internals."""
    current = st.session_state.get(state_key)
    if current not in options:
        current = default if default in options else options[0]
        st.session_state[state_key] = current

    with st.container(
        key=container_key,
        horizontal=True,
        wrap=False,
        horizontal_alignment="center",
        vertical_alignment="center",
        gap="small",
    ):
        for index, option in enumerate(options):
            state = "active" if option == current else "idle"
            st.button(
                option,
                key=f"{container_key}_{state}_{index}",
                disabled=disabled,
                width="stretch",
                on_click=_set_navigation_value,
                args=(state_key, option),
            )
    return str(current)


def render_app_header(
    *,
    job_label: Optional[str] = None,
    model_call_count: int = 0,
    navigation_enabled: bool = True,
    force_group: Optional[str] = None,
) -> str:
    """Render the shared product header and return the selected navigation group."""
    if force_group in NAV_GROUPS:
        st.session_state["workspace_section"] = force_group

    with st.container(key="app_header"):
        brand_column, navigation_column, job_column, count_column, theme_column = (
            st.columns(
                [1.2, 3.25, 1.35, 0.65, 0.35],
                gap="small",
                vertical_alignment="center",
            )
        )
        with brand_column:
            st.markdown("### AI 求职助手")
        with navigation_column:
            selected = render_pill_navigation(
                NAV_GROUPS,
                state_key="workspace_section",
                default="岗位",
                container_key="primary_navigation",
                disabled=not navigation_enabled,
            )
        with job_column:
            with st.container(key="header_job_context"):
                st.caption("当前岗位")
                st.write(job_label or "尚未选择")
        with count_column:
            with st.container(key="header_model_calls"):
                st.caption("模型调用")
                st.write(str(model_call_count))
        with theme_column:
            with st.container(key="header_theme"):
                dark_mode = st.toggle(
                    "深色主题",
                    key="ui_theme_dark",
                    label_visibility="collapsed",
                    help="切换明亮或深色主题",
                )

    selected_theme: ThemeMode = "dark" if dark_mode else "light"
    if selected_theme != st.session_state.get("ui_theme"):
        st.session_state["ui_theme"] = selected_theme
        st.query_params["theme"] = selected_theme
        st.rerun()

    return selected if selected in NAV_GROUPS else "岗位"


def detail_sections(group: str) -> tuple[str, ...]:
    return DETAIL_SECTIONS.get(group, DETAIL_SECTIONS["岗位"])


def default_detail_section(group: str) -> str:
    return DEFAULT_DETAIL_SECTION.get(group, "岗位匹配")


def normalise_selected_job_id(
    job_ids: Sequence[str], requested_job_id: Optional[str]
) -> Optional[str]:
    if requested_job_id in job_ids:
        return requested_job_id
    return job_ids[0] if job_ids else None


def count_pending_confirmations(job_analyses: Iterable[dict]) -> int:
    """Count unanswered or unsure clarification questions across jobs."""
    pending = 0
    for bundle in job_analyses:
        answers = {
            item.get("question_id"): item.get("status")
            for item in bundle.get("clarification_answers", [])
        }
        for question in bundle.get("clarification_questions", []):
            status = answers.get(question.get("id"), "unanswered")
            if status in {"unanswered", "unsure", None}:
                pending += 1
    return pending


def render_score_donut(
    score: Optional[float],
    *,
    label: str = "综合匹配度",
    height: int = 210,
) -> None:
    """Render a compact, theme-aware match-score chart."""
    theme: ThemeMode = (
        "dark" if st.session_state.get("ui_theme") == "dark" else "light"
    )
    tokens = theme_tokens(theme)
    value = max(0.0, min(float(score or 0.0), 100.0))
    data = alt.Data(
        values=[
            {"part": "score", "value": value, "order": 1},
            {"part": "remaining", "value": 100.0 - value, "order": 2},
        ]
    )
    ring = (
        alt.Chart(data)
        .mark_arc(innerRadius=58, outerRadius=72, cornerRadius=7)
        .encode(
            theta=alt.Theta("value:Q", stack=True),
            color=alt.Color(
                "part:N",
                scale=alt.Scale(
                    domain=["score", "remaining"],
                    range=[tokens["accent"], tokens["line"]],
                ),
                legend=None,
            ),
            order=alt.Order("order:Q"),
        )
    )
    score_text = "--" if score is None else f"{value:.0f}%"
    center = (
        alt.Chart(alt.Data(values=[{"score": score_text, "label": label}]))
        .mark_text(
            align="center",
            baseline="middle",
            dy=-7,
            fontSize=30,
            fontWeight=500,
            color=tokens["text"],
        )
        .encode(text="score:N")
    )
    subtitle = (
        alt.Chart(alt.Data(values=[{"label": label}]))
        .mark_text(
            align="center",
            baseline="middle",
            dy=24,
            fontSize=13,
            color=tokens["muted"],
        )
        .encode(text="label:N")
    )
    chart = (
        (ring + center + subtitle)
        .properties(height=height, width=height, background="transparent")
        .configure_view(stroke=None)
    )
    st.altair_chart(chart, width="stretch", theme=None)
