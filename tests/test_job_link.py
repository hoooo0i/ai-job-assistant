import socket

import httpx
import pytest

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
    with pytest.raises(JobLinkError, match="足够"):
        parse_job_posting_html("<main><h1>Role</h1><p>Short</p></main>", "https://example.com")


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
