from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import time

import requests

from core.config import Settings
from core.utils import read_json, write_json

CROSSREF_API_URL = "https://api.crossref.org/works"
CROSSREF_SELECT_FIELDS = (
    "DOI,title,abstract,author,published,published-online,published-print,"
    "issued,URL,subject,type,indexed,resource,container-title"
)
RETRY_STATUS_CODES = {429, 503}
MAX_RETRIES = 3


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    primary_category: str
    published: Any
    updated: Any
    abs_url: str
    pdf_url: str
    comment: str


def _first_text(value: Any) -> str:
    if isinstance(value, list):
        return str(value[0]).strip() if value else ""
    if value is None:
        return ""
    return str(value).strip()


def _author_name(author: dict[str, Any]) -> str:
    if author.get("name"):
        return str(author["name"]).strip()

    parts = [author.get("given", ""), author.get("family", "")]
    return " ".join(str(part).strip() for part in parts if str(part).strip())


def _published_date(item: dict[str, Any]) -> Any:
    return (
        item.get("published")
        or item.get("published-online")
        or item.get("published-print")
        or item.get("issued")
        or ""
    )


def _resource_url(item: dict[str, Any]) -> str:
    resource = item.get("resource")
    if not isinstance(resource, dict):
        return ""

    primary = resource.get("primary")
    if not isinstance(primary, dict):
        return ""

    return _first_text(primary.get("URL"))


def parse_crossref_payload(payload: dict) -> list[PaperRecord]:
    """Parse Crossref payload thanh list PaperRecord."""
    records: list[PaperRecord] = []
    items = payload.get("message", {}).get("items", [])

    for item in items:
        paper_id = _first_text(item.get("DOI"))
        title = _first_text(item.get("title"))
        summary = _first_text(item.get("abstract"))

        if not paper_id or not title:
            continue

        authors = [
            name
            for author in item.get("author", [])
            if isinstance(author, dict) and (name := _author_name(author))
        ]
        categories = [
            str(category).strip()
            for category in item.get("subject", [])
            if str(category).strip()
        ]

        records.append(
            PaperRecord(
                paper_id=paper_id,
                title=title,
                summary=summary,
                authors=authors,
                categories=categories,
                primary_category=categories[0] if categories else "",
                published=_published_date(item),
                updated=item.get("indexed", ""),
                abs_url=_first_text(item.get("URL")),
                pdf_url=_resource_url(item),
                comment="",
            )
        )

    return records


def fetch_source_records(settings: Settings) -> list[PaperRecord]:
    """Goi Crossref API, luu raw response, parse thanh records."""
    params = {
        "query": settings.source_query,
        "filter": settings.source_filter,
        "rows": settings.max_results,
        "select": CROSSREF_SELECT_FIELDS,
        "mailto": "student-lab@example.com",
    }
    headers = {
        "User-Agent": "day10-data-observability-lab/0.1 (mailto:student-lab@example.com)"
    }

    response: requests.Response | None = None
    for attempt in range(MAX_RETRIES + 1):
        response = requests.get(
            CROSSREF_API_URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        if response.status_code not in RETRY_STATUS_CODES or attempt == MAX_RETRIES:
            break

        retry_after = response.headers.get("Retry-After")
        wait_seconds = int(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        time.sleep(wait_seconds)

    if response is None:
        raise RuntimeError("Crossref request was not attempted.")

    response.raise_for_status()
    payload = response.json()

    write_json(settings.paths.raw_api_response, payload)

    records = parse_crossref_payload(payload)
    write_json(settings.paths.raw_records_json, [asdict(record) for record in records])

    return records


def load_raw_records(path: Path) -> list[PaperRecord]:
    """Doc JSON snapshot va map thanh PaperRecord."""
    data = read_json(path)
    return [PaperRecord(**record) for record in data]
