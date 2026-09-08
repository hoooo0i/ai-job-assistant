import base64
import json
import socket

import httpx
import pytest
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

from src.job_link import (
    JobLinkError,
    fetch_job_posting,
    parse_job_posting_html,
    validate_public_job_url,
)


def _public_resolver(host: str, port: int, **kwargs):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]


def test_extracts_schema_org_job_posting() -> None:
    html = """
    <html><head><script type="application/ld+json">
    {
      "@context": "https://schema.org",
      "@type": "JobPosting",
      "title": "Data Analyst Intern",
      "hiringOrganization": {"name": "Example Analytics"},
      "employmentType": "INTERN",
      "jobLocation": {"address": {
        "addressLocality": "Melbourne", "addressRegion": "VIC",
        "addressCountry": "AU"
      }},
      "description": "<p>Work with product data, SQL and Python. Build dashboards, explain findings, collaborate with stakeholders, and validate experiments.</p>"
    }
    </script></head><body></body></html>
    """

    result = parse_job_posting_html(html, "https://jobs.example.com/123")

    assert result.company == "Example Analytics"
    assert result.title == "Data Analyst Intern"
    assert result.location == "Melbourne, VIC, AU"
    assert result.job_type == "实习"
    assert "SQL and Python" in result.description


def test_falls_back_to_visible_page_content() -> None:
    result = parse_job_posting_html(
        """
        <html><head><meta property="og:site_name" content="Example Co"></head>
        <body><main><h1>Product Analyst</h1><p>
        Analyse customer journeys, define product metrics, build reports, work with
        engineering partners, present recommendations, and improve experiments.
        </p></main></body></html>
        """,
        "https://jobs.example.com/product-analyst",
    )

    assert result.company == "Example Co"
    assert result.title == "Product Analyst"
    assert "customer journeys" in result.description


def test_rejects_private_or_short_job_pages() -> None:
    with pytest.raises(JobLinkError, match="本机或内网"):
        validate_public_job_url("http://127.0.0.1:8501/private")
    with pytest.raises(JobLinkError, match="没有可读取"):
        parse_job_posting_html("<main><h1>Role</h1><p>Short</p></main>", "https://example.com")


def test_accepts_url_from_chinese_share_text() -> None:
    result = validate_public_job_url(
        "【分享岗位】数据分析师 https://jobs.example.com/role/123 复制后打开",
        resolver=_public_resolver,
    )

    assert result == "https://jobs.example.com/role/123"


def test_decodes_double_escaped_schema_description() -> None:
    result = parse_job_posting_html(
        """
        <script type="application/ld+json">
        {
          "@type": "JobPosting",
          "title": "Data Analyst",
          "hiringOrganization": {"name": "Example Co"},
          "description": "&amp;lt;p&amp;gt;Analyse product data with SQL and Python, build dashboards, explain findings, work with stakeholders, and validate experiments.&amp;lt;/p&amp;gt;"
        }
        </script>
        """,
        "https://example.com/job",
    )

    assert result.extraction_method == "schema_org"
    assert "<p>" not in result.description
    assert result.description.startswith("Analyse product data")


def test_extracts_job_from_nextjs_embedded_json() -> None:
    result = parse_job_posting_html(
        """
        <html><body><div id="app"></div>
        <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"job":{
          "jobTitle":"Product Analyst",
          "companyName":"Example Product Co",
          "location":"Sydney, NSW",
          "employmentType":"FULL_TIME",
          "jobDescription":"Use SQL to analyse journeys, define metrics, build dashboards, partner with product teams, present recommendations, and improve experiments."
        }}}}
        </script></body></html>
        """,
        "https://careers.example.com/jobs/123",
    )

    assert result.extraction_method == "embedded_json"
    assert result.company == "Example Product Co"
    assert result.title == "Product Analyst"
    assert result.location == "Sydney, NSW"
    assert result.job_type == "全职"


def test_extracts_zhaopin_initial_state() -> None:
    result = parse_job_posting_html(
        """
        <html><head><script>
        __INITIAL_STATE__={
          "jobDetail": {
            "detailedCompany": {"companyName": "示例数据科技公司"},
            "detailedPosition": {
              "positionName": "数据分析师",
              "positionWorkCity": "上海",
              "positionCityDistrict": "浦东新区",
              "workType": "全职",
              "description": "岗位职责：使用 SQL 和 Python 处理业务数据，搭建指标体系和可视化看板，与产品和运营团队协作。任职要求：本科及以上学历，具备良好的沟通能力与分析思维。"
            }
          }
        };
        </script></head><body></body></html>
        """,
        "https://jobs.zhaopin.com/CC123J123.htm",
    )

    assert result.extraction_method == "embedded_json"
    assert result.company == "示例数据科技公司"
    assert result.title == "数据分析师"
    assert result.location == "上海-浦东新区"
    assert result.job_type == "全职"
    assert "SQL 和 Python" in result.description


def test_rejects_inactive_chinese_job_instead_of_recommendations() -> None:
    html = """
    <html><body><main>
      <h1>该职位已暂停招聘</h1>
      <section>相似岗位推荐：数据分析师，负责数据处理、看板建设、业务分析、跨团队沟通、实验设计和汇报。</section>
    </main></body></html>
    """

    with pytest.raises(JobLinkError, match="过期或下线"):
        parse_job_posting_html(html, "https://www.liepin.com/job/123.shtml")


def test_rejects_chinese_search_snippet_as_incomplete_jd() -> None:
    with pytest.raises(JobLinkError, match="搜索摘要"):
        parse_job_posting_html(
            """
            <html><body><main><h1>数据分析师</h1><p>
            示例公司招聘上海数据分析师，薪资1-2万，要求本科，
            招聘负责人刚在线，随时沟通岗位。
            </p></main></body></html>
            """,
            "https://jobs.zhaopin.com/CC123.htm",
        )


def test_greenhouse_url_uses_public_job_board_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "boards-api.greenhouse.io"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "company_name": "Example Greenhouse Co",
                "title": "Software Engineer",
                "location": {"name": "Melbourne, Victoria"},
                "content": (
                    "&lt;p&gt;Build reliable Python services, write tests, review logs, "
                    "document decisions, partner with product teams, and improve deployment safety.&lt;/p&gt;"
                ),
                "metadata": [{"name": "Employment Type", "value": "Full Time"}],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://job-boards.greenhouse.io/example/jobs/12345",
            client=client,
            resolver=_public_resolver,
        )

    assert result.extraction_method == "greenhouse_api"
    assert result.company == "Example Greenhouse Co"
    assert result.location == "Melbourne, Victoria"
    assert result.job_type == "全职"


def test_bytedance_detail_url_uses_public_job_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/job/posts/7675348782494189877"
        assert request.url.params["portal_type"] == "2"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "code": 0,
                "data": {
                    "job_post_detail": {
                        "id": "7675348782494189877",
                        "title": "AI Agent开发工程师",
                        "description": (
                            "负责 AI Agent 平台的后端架构设计、工具编排、"
                            "数据接入和可观测性建设，保障平台的稳定性。"
                        ),
                        "requirement": (
                            "熟悉 Python 或 Go，理解分布式系统和 RAG 机制，"
                            "具备良好的工程实践和团队协作能力。"
                        ),
                        "city_list": [{"name": "北京"}, {"name": "上海"}],
                        "recruit_type": {"name": "正式"},
                        "channel_online_status": 1,
                    }
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://jobs.bytedance.com/campus/position/7675348782494189877/detail",
            client=client,
            resolver=_public_resolver,
        )

    assert result.extraction_method == "bytedance_api"
    assert result.company == "字节跳动"
    assert result.title == "AI Agent开发工程师"
    assert result.location == "北京 / 上海"
    assert result.job_type == "全职"
    assert "职位描述" in result.description
    assert "职位要求" in result.description


def test_bytedance_api_reports_inactive_job() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "code": 0,
                "data": {
                    "job_post_detail": {
                        "title": "已下线岗位",
                        "channel_online_status": 0,
                    }
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JobLinkError, match="字节跳动岗位已下线"):
            fetch_job_posting(
                "https://jobs.bytedance.com/campus/position/123456789/detail",
                client=client,
                resolver=_public_resolver,
            )


def test_tencent_detail_url_uses_public_job_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tencentcareer/api/post/ByPostId"
        assert request.url.params["postId"] == "2064295398285164544"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "Code": 200,
                "Data": {
                    "RecruitPostName": "腾讯云数据分析师",
                    "LocationName": "深圳",
                    "Responsibility": (
                        "负责业务指标体系、数据分析、可视化看板和专项洞察，"
                        "协助产品与运营团队发现问题并推动落地。"
                    ),
                    "Requirement": (
                        "熟练使用 SQL 和 Python，具备良好的逻辑思维、"
                        "沟通能力和跨团队协作能力。"
                    ),
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://careers.tencent.com/jobdesc.html?postId=2064295398285164544",
            client=client,
            resolver=_public_resolver,
        )

    assert result.extraction_method == "tencent_api"
    assert result.company == "腾讯"
    assert result.title == "腾讯云数据分析师"
    assert result.location == "深圳"
    assert "岗位职责" in result.description
    assert "岗位要求" in result.description


def test_xiaomi_detail_url_uses_public_job_api() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/job/posts/7342833451802542188"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "code": 0,
                "data": {
                    "job_post_detail": {
                        "title": "服务产品及市场",
                        "description": (
                            "工作内容\n负责服务产品运营管理，制定业务策略，"
                            "分析经营数据并推动产品与运营项目落地。"
                        ),
                        "requirement": (
                            "职位要求\n本科及以上学历，具备互联网业务经验、"
                            "数据分析能力、沟通协调能力和项目执行力。"
                        ),
                        "recruit_type": {"name": "全职"},
                        "channel_online_status": 1,
                        "city_list": [{"name": "北京"}],
                    }
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://xiaomi.jobs.f.mioffice.cn/index/position/"
            "7342833451802542188/detail",
            client=client,
            resolver=_public_resolver,
        )

    assert result.extraction_method == "xiaomi_api"
    assert result.company == "小米"
    assert result.title == "服务产品及市场"
    assert result.location == "北京"
    assert result.job_type == "全职"
    assert "职位要求" in result.description


@pytest.mark.parametrize(
    "url",
    [
        "https://hr.xiaomi.com/job",
        "https://hr.xiaomi.com/website/opportunities.html",
        "https://xiaomi.jobs.f.mioffice.cn/index/position/list",
    ],
)
def test_xiaomi_list_page_requests_a_specific_job_link(url: str) -> None:
    with pytest.raises(JobLinkError, match="小米.*岗位列表页"):
        fetch_job_posting(url, resolver=_public_resolver)


def test_moka_detail_url_uses_encrypted_public_job_data() -> None:
    aes_iv = "de7c21ed8d6f50fe"
    key = "b0bc54d6181ab82c"
    detail_payload = {
        "code": 0,
        "data": {
            "title": "AI编译器开发工程师",
            "status": "open",
            "commitment": "全职",
            "jobDescription": (
                "<p>岗位职责：负责 AI 编译器的设计和实现，进行性能分析与优化。</p>"
                "<p>任职要求：熟悉 C++ 、MLIR 或 TVM，具备良好的算法基础。</p>"
            ),
            "locations": [{"cityName": "上海市"}, {"cityName": "北京市"}],
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            init_data = json.dumps(
                {"org": {"name": "壁仞科技"}, "aesIv": aes_iv},
                ensure_ascii=False,
            )
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text=f"<input id='init-data' value='{init_data}'>",
            )
        assert request.url.path == "/api/outer/ats-apply/website/job"
        request_data = json.loads(request.content)
        assert request_data["jobId"] == "0847b2bb-f169-4eb6-aa9f-a3b64a61287e"
        plaintext = json.dumps(detail_payload, ensure_ascii=False).encode("utf-8")
        encrypted = AES.new(key.encode(), AES.MODE_CBC, aes_iv.encode()).encrypt(
            pad(plaintext, AES.block_size)
        )
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={
                "data": base64.b64encode(encrypted).decode(),
                "necromancer": key,
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://app.mokahr.com/social-recruitment/biren/44726"
            "#/job/0847b2bb-f169-4eb6-aa9f-a3b64a61287e",
            client=client,
            resolver=_public_resolver,
        )

    assert result.extraction_method == "moka_api"
    assert result.company == "壁仞科技"
    assert result.title == "AI编译器开发工程师"
    assert result.location == "上海市 / 北京市"
    assert result.job_type == "全职"


def test_moka_list_page_requests_a_specific_job_link() -> None:
    with pytest.raises(JobLinkError, match="Moka.*岗位列表页"):
        fetch_job_posting(
            "https://app.mokahr.com/social-recruitment/biren/44726#/jobs",
            resolver=_public_resolver,
        )


def test_reports_access_challenge_instead_of_treating_it_as_jd() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            headers={"content-type": "text/html"},
            text="<title>Authenticating...</title>",
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JobLinkError, match="登录或人机验证"):
            fetch_job_posting(
                "https://jobs.example.com/protected",
                client=client,
                resolver=_public_resolver,
            )


def test_reports_boss_json_challenge() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            json={"code": 37, "message": "您的环境存在异常."},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JobLinkError, match="BOSS直聘.*人机验证"):
            fetch_job_posting(
                "https://www.zhipin.com/job_detail/example.html",
                client=client,
                resolver=_public_resolver,
            )


def test_fetches_html_and_rejects_redirect_to_private_network() -> None:
    html = """
    <main><h1>Engineering Intern</h1><p>
    Support Python services, write tests, review logs, document decisions, work with
    a small product team, and learn reliable deployment practices.
    </p></main>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/redirect":
            return httpx.Response(302, headers={"location": "http://127.0.0.1/admin"})
        return httpx.Response(200, headers={"content-type": "text/html"}, text=html)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_job_posting(
            "https://jobs.example.com/role",
            client=client,
            resolver=_public_resolver,
        )
        assert result.title == "Engineering Intern"
        with pytest.raises(JobLinkError, match="本机或内网"):
            fetch_job_posting(
                "https://jobs.example.com/redirect",
                client=client,
                resolver=_public_resolver,
            )
