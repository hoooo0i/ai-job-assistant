from __future__ import annotations

import base64
import ipaddress
import json
import re
import socket
from collections.abc import Callable
from html import unescape
from typing import Any
from urllib.parse import parse_qs, quote, urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

from src.schemas import JobLinkResult
from src.validators import meaningful_character_count


MAX_PAGE_BYTES = 4 * 1024 * 1024
MAX_REDIRECTS = 6
MAX_DESCRIPTION_CHARACTERS = 24_000
Resolver = Callable[..., list[tuple]]

DESCRIPTION_KEYS = {
    "description",
    "descriptionhtml",
    "jobcontent",
    "jobdesc",
    "jobdescription",
    "jobdescriptionhtml",
    "jobdetails",
    "positiondescription",
    "positiondetail",
    "postingdescription",
    "content",
}
TITLE_KEYS = {
    "title",
    "jobname",
    "jobtitle",
    "positionname",
    "positiontitle",
    "postingtitle",
}
COMPANY_KEYS = {
    "company",
    "companyfullname",
    "companyname",
    "employer",
    "employername",
    "organization",
    "organizationname",
}
LOCATION_KEYS = {
    "citydistrict",
    "cityname",
    "joblocation",
    "location",
    "locationname",
    "positionworkcity",
    "primarylocation",
    "workcity",
}
JOB_TYPE_KEYS = {
    "commitment",
    "employmenttype",
    "empltype",
    "jobtype",
    "worknature",
    "worktype",
}

DESCRIPTION_SELECTORS = [
    '[itemprop="description"]',
    '[data-automation="jobAdDetails"]',
    '[data-testid*="job-description" i]',
    '[data-test*="job-description" i]',
    '[id*="job-description" i]',
    '[id*="jobdescription" i]',
    '[class*="job-description" i]',
    '[class*="jobdescription" i]',
    '[class*="posting-description" i]',
    '#jobDescriptionText',
    # 中国招聘站点常见容器；页面改版后仍会回退到内嵌 JSON/正文识别。
    '.describtion-card__content',
    '.job-sec-text',
    '.job-detail-section',
    '.job-intro-container',
    '.job-detail',
    '.job_bt',
    '.job_msg',
    '.tBorderTop_box',
]

CHINESE_JOB_PLATFORMS = {
    "51job.com": "前程无忧",
    "lagou.com": "拉勾",
    "liepin.com": "猎聘",
    "zhaopin.com": "智联招聘",
    "zhipin.com": "BOSS直聘",
}

EXTRACTION_LABELS = {
    "bytedance_api": "字节跳动招聘公开岗位接口",
    "greenhouse_api": "招聘平台公开接口",
    "moka_api": "Moka 公开岗位数据",
    "schema_org": "网页结构化岗位数据",
    "embedded_json": "页面内嵌岗位数据",
    "page_content": "页面正文",
    "tencent_api": "腾讯招聘公开岗位接口",
    "xiaomi_api": "小米招聘公开岗位接口",
}


class JobLinkError(ValueError):
    """Raised when a job page cannot be fetched or parsed safely."""


def validate_public_job_url(
    url: str,
    *,
    resolver: Resolver = socket.getaddrinfo,
) -> str:
    cleaned = url.strip()
    if not cleaned.startswith(("http://", "https://")) or any(
        character.isspace() for character in cleaned
    ):
        shared_url = re.search(r"https?://[^\s<>\"'，。]+", cleaned)
        if shared_url:
            cleaned = shared_url.group(0).rstrip(")]】};")
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


def _normalise_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _load_script_json(raw: str) -> Any | None:
    candidates = [raw.strip(), unescape(raw.strip())]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        positions = [
            position
            for position in (candidate.find("{"), candidate.find("["))
            if position >= 0
        ]
        for position in sorted(positions):
            try:
                return decoder.raw_decode(candidate[position:])[0]
            except json.JSONDecodeError:
                continue
    return None


def _script_payloads(soup: BeautifulSoup) -> list[Any]:
    payloads: list[Any] = []
    for script in soup.find_all("script"):
        script_type = str(script.get("type") or "").casefold()
        script_id = str(script.get("id") or "")
        raw = script.string or script.get_text() or ""
        likely_json = (
            "json" in script_type
            or script_id in {"__NEXT_DATA__", "__NUXT_DATA__"}
            or any(
                marker in raw
                for marker in (
                    "__INITIAL_STATE__",
                    "JobPosting",
                    "jobDetail",
                    "jobDesc",
                    "jobDescription",
                    "job_description",
                    "positionName",
                    "postingDescription",
                )
            )
        )
        if not likely_json:
            continue
        payload = _load_script_json(raw)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _find_job_posting(payloads: list[Any]) -> dict | None:
    for payload in payloads:
        for item in _walk_json(payload):
            item_type = item.get("@type")
            types = item_type if isinstance(item_type, list) else [item_type]
            if any(str(value).casefold().endswith("jobposting") for value in types):
                return item
    return None


def _clean_html_text(value: str) -> str:
    current = value or ""
    for _ in range(3):
        decoded = unescape(current)
        soup = BeautifulSoup(decoded, "html.parser")
        for node in soup(
            ["script", "style", "noscript", "svg", "nav", "footer", "form", "button"]
        ):
            node.decompose()
        lines = [" ".join(line.split()) for line in soup.get_text("\n").splitlines()]
        deduplicated = list(dict.fromkeys(line for line in lines if line))
        cleaned = "\n".join(deduplicated).strip()
        if not re.search(r"</?(?:p|div|li|ul|ol|h[1-6]|br)\b", cleaned, re.I):
            return cleaned
        if cleaned == current:
            return cleaned
        current = cleaned
    return current.strip()


def _location_text(value: Any) -> str:
    locations = value if isinstance(value, list) else [value]
    parts: list[str] = []
    for location in locations:
        if isinstance(location, str) and location.strip():
            parts.append(location.strip())
            continue
        if not isinstance(location, dict):
            continue
        for key in ("name", "location"):
            if isinstance(location.get(key), str) and location[key].strip():
                parts.append(location[key].strip())
                break
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
        "全职": "全职",
        "PART_TIME": "兼职",
        "兼职": "兼职",
        "INTERN": "实习",
        "INTERNSHIP": "实习",
        "实习": "实习",
        "CONTRACTOR": "合同",
        "CONTRACT": "合同",
        "TEMPORARY": "合同",
        "合同": "合同",
    }
    for item in values:
        key = str(item or "").upper().replace("-", "_").replace(" ", "_")
        if key in mapping:
            return mapping[key]
    return ""


def _dict_value(item: dict, keys: set[str]) -> Any:
    for key, value in item.items():
        if _normalise_key(key) in keys and value not in (None, "", [], {}):
            return value
    return None


def _simple_text(value: Any) -> str:
    if isinstance(value, str):
        return _clean_html_text(value)
    if isinstance(value, dict):
        for key in ("name", "text", "label", "value"):
            if value.get(key):
                return _simple_text(value[key])
    if isinstance(value, list):
        return " / ".join(
            dict.fromkeys(text for item in value if (text := _simple_text(item)))
        )
    return ""


def _job_result_from_payloads(
    payloads: list[Any],
    source_url: str,
) -> JobLinkResult | None:
    candidates: list[tuple[int, dict, str]] = []
    for payload in payloads:
        for item in _walk_json(payload):
            normalised_keys = {_normalise_key(key) for key in item}
            for key, value in item.items():
                normalised_key = _normalise_key(key)
                if normalised_key not in DESCRIPTION_KEYS or not isinstance(value, str):
                    continue
                description = _clean_html_text(value)
                length = meaningful_character_count(description)
                if length < 50:
                    continue
                context_score = 0
                if normalised_key != "content":
                    context_score += 2_000
                if normalised_keys & TITLE_KEYS:
                    context_score += 1_000
                if normalised_keys & (COMPANY_KEYS | LOCATION_KEYS | JOB_TYPE_KEYS):
                    context_score += 500
                candidates.append((context_score + min(length, 30_000), item, description))
    if not candidates:
        return None
    _, selected, description = max(candidates, key=lambda candidate: candidate[0])

    def find_global(keys: set[str]) -> Any:
        local = _dict_value(selected, keys)
        if local not in (None, "", [], {}):
            return local
        for payload in payloads:
            for item in _walk_json(payload):
                value = _dict_value(item, keys)
                if value not in (None, "", [], {}):
                    return value
        return None

    return JobLinkResult(
        source_url=source_url,
        company=_simple_text(find_global(COMPANY_KEYS)),
        title=_simple_text(find_global(TITLE_KEYS)),
        location=_location_text(find_global(LOCATION_KEYS)),
        job_type=_job_type_text(find_global(JOB_TYPE_KEYS)),
        description=description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="embedded_json",
    )


def _zhaopin_result(payloads: list[Any], source_url: str) -> JobLinkResult | None:
    """Read the server-rendered ``__INITIAL_STATE__`` used by Zhaopin pages."""
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        job_detail = payload.get("jobDetail")
        if not isinstance(job_detail, dict):
            continue
        position = job_detail.get("detailedPosition")
        company_info = job_detail.get("detailedCompany")
        if not isinstance(position, dict):
            continue
        company_info = company_info if isinstance(company_info, dict) else {}
        description = _clean_html_text(
            str(position.get("description") or position.get("jobDesc") or "")
        )
        if meaningful_character_count(description) < 50:
            continue
        city = _simple_text(
            position.get("positionWorkCity") or position.get("workCity")
        )
        district = _simple_text(position.get("positionCityDistrict"))
        location = "-".join(
            dict.fromkeys(value for value in (city, district) if value)
        )
        return JobLinkResult(
            source_url=source_url,
            company=_simple_text(
                company_info.get("companyName") or position.get("companyName")
            ),
            title=_simple_text(
                position.get("positionName") or position.get("name")
            ),
            location=location,
            job_type=_job_type_text(
                position.get("workType") or position.get("emplType")
            ),
            description=description[:MAX_DESCRIPTION_CHARACTERS],
            extraction_method="embedded_json",
        )
    return None


def _best_page_description(soup: BeautifulSoup) -> str:
    candidates: list[str] = []
    for selector in DESCRIPTION_SELECTORS:
        for node in soup.select(selector):
            text = _clean_html_text(str(node))
            if meaningful_character_count(text) >= 50:
                candidates.append(text)
    for node in [soup.find("main"), soup.find("article"), soup.body]:
        if node:
            text = _clean_html_text(str(node))
            if meaningful_character_count(text) >= 50:
                candidates.append(text)
    if not candidates:
        return ""

    def score(text: str) -> tuple[int, int]:
        lowered = text.casefold()
        job_markers = sum(
            marker in lowered
            for marker in (
                "responsibilit",
                "requirement",
                "qualification",
                "about the role",
                "what you'll",
                "岗位职责",
                "任职要求",
                "职位描述",
            )
        )
        return job_markers, min(meaningful_character_count(text), 30_000)

    return max(candidates, key=score)[:MAX_DESCRIPTION_CHARACTERS]


def _meta_content(soup: BeautifulSoup, *identifiers: tuple[str, str]) -> str:
    for attribute, value in identifiers:
        node = soup.find("meta", attrs={attribute: value})
        if node and node.get("content"):
            return _clean_html_text(str(node.get("content")))
    return ""


def _greenhouse_api_url(url: str) -> str | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host not in {
        "boards.greenhouse.io",
        "job-boards.greenhouse.io",
        "boards.eu.greenhouse.io",
        "job-boards.eu.greenhouse.io",
    }:
        return None
    parts = [part for part in parsed.path.split("/") if part]
    board = ""
    job_id = ""
    if len(parts) >= 3 and parts[1] == "jobs":
        board, job_id = parts[0], parts[2]
    elif parts[:2] == ["embed", "job_app"]:
        query = parse_qs(parsed.query)
        board = (query.get("for") or [""])[0]
        job_id = (query.get("token") or [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", board) or not re.fullmatch(r"\d+", job_id):
        return None
    return (
        "https://boards-api.greenhouse.io/v1/boards/"
        f"{quote(board, safe='')}/jobs/{quote(job_id, safe='')}"
    )


def _greenhouse_result(payload: Any, source_url: str) -> JobLinkResult | None:
    if not isinstance(payload, dict):
        return None
    description = _clean_html_text(str(payload.get("content") or ""))
    if meaningful_character_count(description) < 50:
        return None
    location = payload.get("location")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), list) else []
    job_type_values = [
        item.get("value")
        for item in metadata
        if isinstance(item, dict)
        and _normalise_key(item.get("name")) in {"employmenttype", "jobtype", "worktype"}
    ]
    return JobLinkResult(
        source_url=source_url,
        company=_simple_text(payload.get("company_name")),
        title=_simple_text(payload.get("title")),
        location=_location_text(location),
        job_type=_job_type_text(job_type_values),
        description=description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="greenhouse_api",
    )


def _bytedance_api_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if (parsed.hostname or "").casefold() != "jobs.bytedance.com":
        return None
    match = re.search(r"/position/(\d+)/detail(?:/|$)", parsed.path)
    if not match:
        return None
    job_id = match.group(1)
    return (
        f"https://jobs.bytedance.com/api/v1/job/posts/{quote(job_id, safe='')}"
        "?portal_type=2&lang=zh-CN"
    )


def _bytedance_result(payload: Any, source_url: str) -> JobLinkResult | None:
    if not isinstance(payload, dict) or payload.get("code") != 0:
        return None
    data = payload.get("data")
    detail = data.get("job_post_detail") if isinstance(data, dict) else None
    if not isinstance(detail, dict):
        return None
    if detail.get("channel_online_status") in {0, False}:
        raise JobLinkError("该字节跳动岗位已下线或停止招聘。")

    description = _clean_html_text(str(detail.get("description") or ""))
    requirement = _clean_html_text(str(detail.get("requirement") or ""))
    sections = []
    if description:
        sections.append(f"职位描述\n{description}")
    if requirement:
        sections.append(f"职位要求\n{requirement}")
    full_description = "\n\n".join(sections)
    if meaningful_character_count(full_description) < 50:
        return None

    city_values: list[str] = []
    city_list = detail.get("city_list")
    if isinstance(city_list, list):
        city_values.extend(
            _simple_text(city)
            for city in city_list
            if isinstance(city, dict) and _simple_text(city)
        )
    if not city_values:
        city = _simple_text(detail.get("city_info"))
        if city:
            city_values.append(city)

    recruit_type = _simple_text(detail.get("recruit_type"))
    if "实习" in recruit_type:
        job_type = "实习"
    elif any(marker in recruit_type for marker in ("正式", "社招", "校招")):
        job_type = "全职"
    else:
        job_type = _job_type_text(recruit_type)

    return JobLinkResult(
        source_url=source_url,
        company="字节跳动",
        title=_simple_text(detail.get("title")),
        location=" / ".join(dict.fromkeys(city_values)),
        job_type=job_type,
        description=full_description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="bytedance_api",
    )


def _tencent_api_url(url: str) -> str | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host not in {"careers.tencent.com", "hr.tencent.com"}:
        return None
    post_id = (parse_qs(parsed.query).get("postId") or [""])[0]
    if not re.fullmatch(r"\d+", post_id):
        return None
    return (
        "https://careers.tencent.com/tencentcareer/api/post/ByPostId"
        f"?postId={quote(post_id, safe='')}&language=zh-cn"
    )


def _tencent_result(payload: Any, source_url: str) -> JobLinkResult | None:
    if not isinstance(payload, dict) or payload.get("Code") != 200:
        return None
    detail = payload.get("Data")
    if not isinstance(detail, dict):
        return None
    responsibility = _clean_html_text(str(detail.get("Responsibility") or ""))
    requirement = _clean_html_text(str(detail.get("Requirement") or ""))
    sections = []
    if responsibility:
        sections.append(f"岗位职责\n{responsibility}")
    if requirement:
        sections.append(f"岗位要求\n{requirement}")
    description = "\n\n".join(sections)
    if meaningful_character_count(description) < 50:
        return None
    return JobLinkResult(
        source_url=source_url,
        company="腾讯",
        title=_simple_text(detail.get("RecruitPostName")),
        location=_simple_text(detail.get("LocationName")),
        description=description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="tencent_api",
    )


def _xiaomi_api_url(url: str) -> str | None:
    parsed = urlsplit(url)
    if (parsed.hostname or "").casefold() != "xiaomi.jobs.f.mioffice.cn":
        return None
    match = re.search(r"/position/(\d+)/detail(?:/|$)", parsed.path)
    if not match:
        match = re.search(r"/position/detail/(\d+)(?:/|$)", parsed.path)
    if not match:
        return None
    job_post_id = match.group(1)
    return (
        "https://xiaomi.jobs.f.mioffice.cn/api/v1/job/posts/"
        f"{quote(job_post_id, safe='')}"
    )


def _is_xiaomi_job_list_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host == "hr.xiaomi.com":
        return parsed.path.rstrip("/") in {
            "",
            "/job",
            "/website/opportunities.html",
        }
    if host != "xiaomi.jobs.f.mioffice.cn":
        return False
    return not bool(
        re.search(
            r"/position/(?:\d+/detail|detail/\d+)(?:/|$)",
            parsed.path,
        )
    )


def _xiaomi_result(payload: Any, source_url: str) -> JobLinkResult | None:
    if not isinstance(payload, dict) or payload.get("code") != 0:
        return None
    data = payload.get("data")
    detail = data.get("job_post_detail") if isinstance(data, dict) else None
    if not isinstance(detail, dict):
        return None
    if detail.get("channel_online_status") in {0, False}:
        raise JobLinkError("该小米岗位已下线或停止招聘。")

    responsibility = _clean_html_text(str(detail.get("description") or ""))
    requirement = _clean_html_text(str(detail.get("requirement") or ""))
    description = "\n\n".join(
        section for section in (responsibility, requirement) if section
    )
    if meaningful_character_count(description) < 50:
        return None

    locations: list[str] = []
    city_list = detail.get("city_list")
    if isinstance(city_list, list):
        locations.extend(
            _simple_text(city)
            for city in city_list
            if isinstance(city, dict) and _simple_text(city)
        )

    return JobLinkResult(
        source_url=source_url,
        company="小米",
        title=_simple_text(detail.get("title")),
        location=" / ".join(dict.fromkeys(locations)),
        job_type=_job_type_text(_simple_text(detail.get("recruit_type"))),
        description=description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="xiaomi_api",
    )


def _moka_coordinates(url: str) -> tuple[str, str, str, str] | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if host != "mokahr.com" and not host.endswith(".mokahr.com"):
        return None
    portal_match = re.match(
        r"/(?:social-recruitment|campus-recruitment|recommendation-recruitment|apply)"
        r"/([^/]+)/([0-9]+)",
        parsed.path,
    )
    fragment_match = re.search(
        r"(?:^|/)job/([A-Za-z0-9-]{8,})(?:[/?]|$)",
        parsed.fragment,
    )
    if not portal_match or not fragment_match:
        return None
    org_id, site_id = portal_match.groups()
    job_id = fragment_match.group(1)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    return origin, org_id, site_id, job_id


def _is_moka_portal_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    return (host == "mokahr.com" or host.endswith(".mokahr.com")) and bool(
        re.match(
            r"/(?:social-recruitment|campus-recruitment|"
            r"recommendation-recruitment|apply)/[^/]+/[0-9]+",
            parsed.path,
        )
    )


def _moka_page_context(html: str) -> tuple[str, str] | None:
    node = BeautifulSoup(html, "html.parser").select_one("#init-data")
    if not node or not node.get("value"):
        return None
    try:
        payload = json.loads(str(node.get("value")))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    organisation = payload.get("org")
    company = _simple_text(organisation.get("name")) if isinstance(organisation, dict) else ""
    aes_iv = str(payload.get("aesIv") or "")
    if len(aes_iv.encode("utf-8")) != AES.block_size:
        return None
    return company, aes_iv


def _decrypt_moka_payload(payload: Any, aes_iv: str) -> Any | None:
    if not isinstance(payload, dict):
        return None
    if not payload.get("necromancer"):
        return payload
    key = str(payload.get("necromancer") or "").encode("utf-8")
    iv = aes_iv.encode("utf-8")
    if len(key) not in {16, 24, 32} or len(iv) != AES.block_size:
        return None
    try:
        encrypted = base64.b64decode(str(payload.get("data") or ""), validate=True)
        decrypted = AES.new(key, AES.MODE_CBC, iv).decrypt(encrypted)
        return json.loads(unpad(decrypted, AES.block_size).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _moka_result(payload: Any, source_url: str, company: str) -> JobLinkResult | None:
    if not isinstance(payload, dict) or payload.get("code") not in {0, 200}:
        return None
    detail = payload.get("data")
    if not isinstance(detail, dict):
        return None
    if str(detail.get("status") or "").casefold() not in {"", "open"}:
        raise JobLinkError("该 Moka 岗位已下线或停止招聘。")
    description = _clean_html_text(str(detail.get("jobDescription") or ""))
    if meaningful_character_count(description) < 50:
        return None
    locations = detail.get("locations")
    location_values: list[str] = []
    if isinstance(locations, list):
        for location in locations:
            if not isinstance(location, dict):
                continue
            value = _simple_text(
                location.get("cityName")
                or location.get("provinceName")
                or location.get("address")
            )
            if value:
                location_values.append(value)
    return JobLinkResult(
        source_url=source_url,
        company=company,
        title=_simple_text(detail.get("title")),
        location=" / ".join(dict.fromkeys(location_values)),
        job_type=_job_type_text(detail.get("commitment")),
        description=description[:MAX_DESCRIPTION_CHARACTERS],
        extraction_method="moka_api",
    )


def _platform_name(url: str) -> str:
    host = (urlsplit(url).hostname or "").casefold()
    for domain, name in CHINESE_JOB_PLATFORMS.items():
        if host == domain or host.endswith(f".{domain}"):
            return name
    return "该招聘网站"


def _looks_like_access_challenge(html: str, source_url: str = "") -> bool:
    lowered = html.casefold()
    parsed = urlsplit(source_url)
    challenge_path = any(
        marker in parsed.path.casefold()
        for marker in ("/security", "/passport", "/captcha", "/verify", "/transit")
    )
    return challenge_path or any(
        marker in lowered
        for marker in (
            "additional verification required",
            "cf-chl-",
            "captcha-container",
            "g-recaptcha",
            "<title>authenticating...</title>",
            "verify you are human",
            "<title>请稍候 - boss直聘</title>",
            "安全验证",
            "滑动验证",
            "访问过于频繁",
            "异常访问",
            "您的环境存在异常",
        )
    )


def _looks_like_inactive_job(html: str) -> bool:
    return any(
        marker in html
        for marker in (
            "该职位已暂停招聘",
            "该职位已停止招聘",
            "职位已下线",
            "岗位已下线",
            "职位已过期",
            "职位已关闭",
            "招聘已结束",
        )
    )


def _looks_like_search_snippet(description: str) -> bool:
    compact = "".join(description.split())
    has_job_sections = any(
        marker in compact
        for marker in (
            "岗位职责",
            "任职要求",
            "职位描述",
            "工作内容",
            "职位要求",
            "responsibilities",
            "requirements",
        )
    )
    seo_markers = ("招聘负责人", "随时沟通岗位", "聊一聊", "立即投递")
    return not has_job_sections and any(marker in compact for marker in seo_markers)


def parse_job_posting_html(html: str, source_url: str) -> JobLinkResult:
    if _looks_like_access_challenge(html, source_url):
        raise JobLinkError(
            f"{_platform_name(source_url)}要求登录或人机验证，无法自动读取；"
            "请复制岗位正文并粘贴到 JD 输入框。"
        )
    if _looks_like_inactive_job(html):
        raise JobLinkError(
            "该职位已暂停、过期或下线，没有导入页面中的相似岗位。"
        )
    soup = BeautifulSoup(html, "html.parser")
    payloads = _script_payloads(soup)
    if "zhaopin.com" in (urlsplit(source_url).hostname or "").casefold():
        zhaopin_result = _zhaopin_result(payloads, source_url)
        if zhaopin_result:
            return zhaopin_result
    posting = _find_job_posting(payloads)
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
        title = _meta_content(
            soup,
            ("property", "og:title"),
            ("name", "twitter:title"),
        )
    if not company:
        company = _meta_content(soup, ("property", "og:site_name"))
    extraction_method = "schema_org" if posting else "page_content"
    if meaningful_character_count(description) < 50:
        embedded = _job_result_from_payloads(payloads, source_url)
        if embedded:
            company = company or embedded.company
            title = title or embedded.title
            location = location or embedded.location
            job_type = job_type or embedded.job_type
            description = embedded.description
            extraction_method = embedded.extraction_method
    if meaningful_character_count(description) < 50:
        description = _best_page_description(soup)
        extraction_method = "page_content"
    description = description[:MAX_DESCRIPTION_CHARACTERS].strip()
    if meaningful_character_count(description) < 50:
        visible_text = _clean_html_text(str(soup.body or soup))
        if _looks_like_search_snippet(visible_text):
            raise JobLinkError(
                "网页只提供了岗位搜索摘要，没有返回完整职责和要求；"
                "为避免误分析，请手动粘贴完整 JD。"
            )
        raise JobLinkError(
            "网页只返回了动态页面框架，没有可读取的岗位正文；请复制岗位正文并粘贴到 JD 输入框。"
        )
    if _looks_like_search_snippet(description):
        raise JobLinkError(
            "网页只提供了岗位搜索摘要，没有返回完整职责和要求；"
            "为避免误分析，请手动粘贴完整 JD。"
        )
    return JobLinkResult(
        source_url=source_url,
        company=company,
        title=title,
        location=location,
        job_type=job_type,
        description=description,
        extraction_method=extraction_method,
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
        timeout=httpx.Timeout(15.0, connect=5.0),
        follow_redirects=False,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/128.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/json;q=0.9,"
                "application/xml;q=0.8,*/*;q=0.7"
            ),
            "Accept-Language": "en-AU,en;q=0.9,zh-CN;q=0.7,zh;q=0.6",
        },
    )

    def read_limited(response: httpx.Response) -> bytes:
        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_PAGE_BYTES:
                raise JobLinkError("岗位页面超过 4 MB，请复制岗位正文并手动粘贴。")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        if _is_xiaomi_job_list_url(current_url):
            raise JobLinkError(
                "这是小米招聘首页或岗位列表页，不是具体 JD。"
                "请先打开某个岗位，再复制包含 /position/…/detail 的完整链接。"
            )
        moka_coordinates = _moka_coordinates(current_url)
        if _is_moka_portal_url(current_url) and not moka_coordinates:
            raise JobLinkError(
                "这是 Moka 公司招聘首页或岗位列表页，不是具体 JD。"
                "请先打开具体岗位，再复制地址栏中包含 #/job/ 的完整链接。"
            )
        if moka_coordinates:
            origin, org_id, site_id, job_id = moka_coordinates
            base_url = current_url.split("#", 1)[0]
            api_url = f"{origin}/api/outer/ats-apply/website/job"
            validate_public_job_url(api_url, resolver=resolver)
            try:
                page_context = None
                page_url = base_url
                for _ in range(3):
                    validate_public_job_url(page_url, resolver=resolver)
                    with http_client.stream("GET", page_url) as page_response:
                        if page_response.status_code in {301, 302, 303, 307, 308}:
                            location = page_response.headers.get("location")
                            if not location:
                                break
                            page_url = urljoin(page_url, location).split("#", 1)[0]
                            continue
                        if page_response.status_code == 200:
                            page_raw = read_limited(page_response)
                            page_html = page_raw.decode(
                                page_response.encoding or "utf-8", errors="replace"
                            )
                            page_context = _moka_page_context(page_html)
                        break
                if page_context:
                    company, aes_iv = page_context
                    with http_client.stream(
                        "POST",
                        api_url,
                        headers={"Accept": "application/json", "Referer": current_url},
                        json={
                            "orgId": org_id,
                            "siteId": site_id,
                            "jobId": job_id,
                            "locale": "zh-CN",
                        },
                    ) as response:
                        if response.status_code == 200:
                            raw = read_limited(response)
                            try:
                                envelope = json.loads(
                                    raw.decode(response.encoding or "utf-8")
                                )
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                envelope = None
                            payload = _decrypt_moka_payload(envelope, aes_iv)
                            result = _moka_result(payload, current_url, company)
                            if result:
                                return result
            except httpx.HTTPError:
                pass

        xiaomi_url = _xiaomi_api_url(current_url)
        if xiaomi_url:
            validate_public_job_url(xiaomi_url, resolver=resolver)
            try:
                with http_client.stream(
                    "GET",
                    xiaomi_url,
                    headers={"Accept": "application/json", "Referer": current_url},
                ) as response:
                    if response.status_code == 200:
                        raw = read_limited(response)
                        try:
                            payload = json.loads(raw.decode(response.encoding or "utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            payload = None
                        result = _xiaomi_result(payload, current_url)
                        if result:
                            return result
            except httpx.HTTPError:
                pass

        tencent_url = _tencent_api_url(current_url)
        if tencent_url:
            validate_public_job_url(tencent_url, resolver=resolver)
            try:
                with http_client.stream(
                    "GET",
                    tencent_url,
                    headers={"Accept": "application/json", "Referer": current_url},
                ) as response:
                    if response.status_code == 200:
                        raw = read_limited(response)
                        try:
                            payload = json.loads(raw.decode(response.encoding or "utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            payload = None
                        result = _tencent_result(payload, current_url)
                        if result:
                            return result
            except httpx.HTTPError:
                pass

        bytedance_url = _bytedance_api_url(current_url)
        if bytedance_url:
            validate_public_job_url(bytedance_url, resolver=resolver)
            try:
                with http_client.stream(
                    "GET",
                    bytedance_url,
                    headers={"Accept": "application/json", "Referer": current_url},
                ) as response:
                    if response.status_code == 200:
                        raw = read_limited(response)
                        try:
                            payload = json.loads(raw.decode(response.encoding or "utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            payload = None
                        result = _bytedance_result(payload, current_url)
                        if result:
                            return result
            except httpx.HTTPError:
                pass

        greenhouse_url = _greenhouse_api_url(current_url)
        if greenhouse_url:
            validate_public_job_url(greenhouse_url, resolver=resolver)
            try:
                with http_client.stream(
                    "GET",
                    greenhouse_url,
                    headers={"Accept": "application/json"},
                ) as response:
                    if response.status_code == 200:
                        raw = read_limited(response)
                        try:
                            payload = json.loads(raw.decode(response.encoding or "utf-8"))
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            payload = None
                        result = _greenhouse_result(payload, current_url)
                        if result:
                            return result
            except httpx.HTTPError:
                pass

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
                    if response.status_code in {401, 403}:
                        raise JobLinkError(
                            f"招聘网站拒绝自动读取（HTTP {response.status_code}），"
                            "通常是登录或人机验证限制；请复制岗位正文并粘贴到 JD 输入框。"
                        )
                    if response.status_code == 429:
                        raise JobLinkError(
                            "招聘网站请求过于频繁（HTTP 429），请稍后重试或手动粘贴 JD。"
                        )
                    if response.status_code == 404:
                        raise JobLinkError(
                            "岗位页面不存在或职位已经下线（HTTP 404），请检查链接。"
                        )
                    if response.status_code >= 400:
                        raise JobLinkError(f"岗位页面访问失败（HTTP {response.status_code}）。")
                    content_type = response.headers.get("content-type", "").lower()
                    if content_type and not any(
                        value in content_type for value in ("html", "xhtml", "json")
                    ):
                        raise JobLinkError("岗位链接返回的不是网页内容。")
                    raw = read_limited(response)
                    encoding = response.encoding or "utf-8"
                    html = raw.decode(encoding, errors="replace")
                    if _looks_like_access_challenge(html, current_url):
                        raise JobLinkError(
                            f"{_platform_name(current_url)}要求登录或人机验证，无法自动读取；"
                            "请复制岗位正文并粘贴到 JD 输入框。"
                        )
                    if "json" in content_type:
                        try:
                            result = _job_result_from_payloads(
                                [json.loads(html)],
                                current_url,
                            )
                        except json.JSONDecodeError:
                            result = None
                        if result:
                            return result
                        raise JobLinkError(
                            "接口返回了数据，但没有找到完整岗位描述；请手动粘贴 JD。"
                        )
                    return parse_job_posting_html(html, current_url)
            except httpx.HTTPError as exc:
                raise JobLinkError("无法访问岗位页面，请检查链接或手动粘贴 JD。") from exc
        raise JobLinkError("岗位页面跳转次数过多。")
    finally:
        if owns_client:
            http_client.close()
