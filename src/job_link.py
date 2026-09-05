from __future__ import annotations

import ipaddress
import json
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from src.schemas import JobLinkResult
from src.validators import meaningful_character_count


MAX_HTML_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 4
Resolver = Callable[..., list[tuple]]


class JobLinkError(ValueError):
    """Raised when a job page cannot be fetched or parsed safely."""


def validate_public_job_url(
    url: str,
    *,
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    cleaned = url.strip()
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise JobLinkError("岗位链接必须是完整的 http:// 或 https:// 公网地址。")
    if parsed.username or parsed.password:
        raise JobLinkError("岗位链接不能包含用户名或密码。")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise JobLinkError("岗位链接端口无效。") from exc
    try:
        literal_ip = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None and not literal_ip.is_global:
        raise JobLinkError("出于安全原因，不能访问本机或内网地址。")
    try:
        addresses = resolver(parsed.hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise JobLinkError("无法解析岗位链接的域名。") from exc
    if not addresses:
        raise JobLinkError("无法解析岗位链接的域名。")
    for address in addresses:
        ip_value = ipaddress.ip_address(address[4][0])
        if not ip_value.is_global:
            raise JobLinkError("出于安全原因，不能访问本机或内网地址。")
    return cleaned


def _walk_json(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_json(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_json(item)


def _find_job_posting(soup: BeautifulSoup) -> dict | None:
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            payload = json.loads(script.string or script.get_text() or "")
        except (TypeError, json.JSONDecodeError):
            continue
        for item in _walk_json(payload):
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if "JobPosting" in types:
                return item
    return None


def _clean_html_text(value: str) -> str:
    soup = BeautifulSoup(value or "", "html.parser")
    for node in soup(["script", "style", "noscript", "svg"]):
        node.decompose()
    lines = [" ".join(line.split()) for line in soup.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _location_text(value: Any) -> str:
    locations = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        address = location.get("address", location)
        if isinstance(address, str):
            parts.append(address)
            continue
        if isinstance(address, dict):
            country = address.get("addressCountry")
            if isinstance(country, dict):
                country = country.get("name")
            text = ", ".join(
                str(item)
                for item in [
                    address.get("addressLocality"),
                    address.get("addressRegion"),
                    country,
                ]
                if item
            )
            if text:
                parts.append(text)
    return " / ".join(dict.fromkeys(parts))


def _job_type_text(value: Any) -> str:
    values = value if isinstance(value, list) else [value]
    mapping = {
        "FULL_TIME": "全职",
        "PART_TIME": "兼职",
        "INTERN": "实习",
        "INTERNSHIP": "实习",
        "CONTRACTOR": "合同",
        "CONTRACT": "合同",
        "TEMPORARY": "合同",
    }
    for item in values:
        key = str(item or "").upper().replace("-", "_").replace(" ", "_")
        if key in mapping:
            return mapping[key]
    return ""


def parse_job_posting_html(html: str, source_url: str) -> JobLinkResult:
    soup = BeautifulSoup(html, "html.parser")
    posting = _find_job_posting(soup)
    company = ""
    title = ""
    location = ""
    job_type = ""
    description = ""
    if posting:
        title = str(posting.get("title") or "").strip()
        organisation = posting.get("hiringOrganization")
        if isinstance(organisation, dict):
            company = str(organisation.get("name") or "").strip()
        elif isinstance(organisation, str):
            company = organisation.strip()
        location = _location_text(posting.get("jobLocation"))
        job_type = _job_type_text(posting.get("employmentType"))
        description = _clean_html_text(str(posting.get("description") or ""))

    if not title:
        heading = soup.find("h1")
        title = " ".join(heading.get_text(" ").split()) if heading else ""
    if not title:
        meta_title = soup.find("meta", attrs={"property": "og:title"})
        title = str(meta_title.get("content") or "").strip() if meta_title else ""
    if not company:
        site_name = soup.find("meta", attrs={"property": "og:site_name"})
        company = str(site_name.get("content") or "").strip() if site_name else ""
    if meaningful_character_count(description) < 50:
        root = soup.find("main") or soup.find("article") or soup.body
        if root:
            for node in root.find_all(["nav", "header", "footer", "script", "style", "form"]):
                node.decompose()
            description = _clean_html_text(str(root))
    description = description[:20_000].strip()
    if meaningful_character_count(description) < 50:
        raise JobLinkError("网页中没有提取到足够的岗位描述，请手动粘贴 JD。")
    return JobLinkResult(
        source_url=source_url,
        company=company,
        title=title,
        location=location,
        job_type=job_type,
        description=description,
    )


def fetch_job_posting(
    url: str,
    *,
    client: httpx.Client | None = None,
    resolver: Resolver = socket.getaddrinfo,
) -> JobLinkResult:
    """Fetch a small public HTML page and extract a JobPosting payload when present."""
    current_url = validate_public_job_url(url, resolver=resolver)
    owns_client = client is None
    http_client = client or httpx.Client(
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=False,
        headers={"User-Agent": "AIJobAssistant/1.0 (+job-description-import)"},
    )
    try:
        for _ in range(MAX_REDIRECTS + 1):
            validate_public_job_url(current_url, resolver=resolver)
            try:
                with http_client.stream("GET", current_url) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location:
                            raise JobLinkError("岗位页面返回了无效跳转。")
                        current_url = urljoin(current_url, location)
                        continue
                    if response.status_code >= 400:
                        raise JobLinkError(f"岗位页面访问失败（HTTP {response.status_code}）。")
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and "html" not in content_type and "xhtml" not in content_type:
                        raise JobLinkError("岗位链接返回的不是网页内容。")
                    chunks: list[bytes] = []
                    size = 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > MAX_HTML_BYTES:
                            raise JobLinkError("岗位页面超过 2 MB，请手动粘贴 JD。")
                        chunks.append(chunk)
                    encoding = response.encoding or "utf-8"
                    html = b"".join(chunks).decode(encoding, errors="replace")
                    return parse_job_posting_html(html, current_url)
            except httpx.HTTPError as exc:
                raise JobLinkError("无法访问岗位页面，请检查链接或手动粘贴 JD。") from exc
        raise JobLinkError("岗位页面跳转次数过多。")
    finally:
        if owns_client:
            http_client.close()
