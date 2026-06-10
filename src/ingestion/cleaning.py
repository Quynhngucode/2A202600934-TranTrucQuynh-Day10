from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import html
import re
from typing import Any

import pandas as pd

from ingestion.crossref import PaperRecord


def parse_crossref_date(value: Any) -> date | None:
    if not isinstance(value, dict):
        return None

    date_parts = value.get("date-parts", [[None]])
    if not date_parts or not date_parts[0]:
        return None

    parts = date_parts[0]
    year = parts[0] if len(parts) > 0 else None
    month = parts[1] if len(parts) > 1 else 1
    day = parts[2] if len(parts) > 2 else 1

    if year is None:
        return None

    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    cleaned = html.unescape(str(text))
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def clean_summary(value: Any) -> str:
    return normalize_text(value)


def _record_to_dict(record: PaperRecord | dict[str, Any]) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return record
    return vars(record)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    text = normalize_text(value)
    return [text] if text else []


def build_clean_dataframe(records: list[PaperRecord], run_date: datetime) -> pd.DataFrame:
    """Clean raw records thanh dataframe san sang de embed."""
    run_day = run_date.date() if isinstance(run_date, datetime) else run_date
    rows: list[dict[str, Any]] = []

    for record in records:
        item = _record_to_dict(record)
        authors = _as_list(item.get("authors"))
        categories = _as_list(item.get("categories"))

        title = normalize_text(item.get("title"))
        summary = clean_summary(item.get("summary"))
        published_date = parse_crossref_date(item.get("published"))
        updated_date = parse_crossref_date(item.get("updated"))

        authors_joined = ", ".join(authors)
        categories_joined = ", ".join(categories)
        text_for_embedding = normalize_text(
            " ".join(
                part
                for part in [title, summary, authors_joined, categories_joined]
                if part
            )
        )

        rows.append(
            {
                "paper_id": normalize_text(item.get("paper_id")),
                "title": title,
                "summary": summary,
                "authors": authors,
                "categories": categories,
                "primary_category": normalize_text(
                    item.get("primary_category") or (categories[0] if categories else "")
                ),
                "published": published_date.isoformat() if published_date else "",
                "updated": updated_date.isoformat() if updated_date else "",
                "age_days": (run_day - published_date).days if published_date else None,
                "abs_url": normalize_text(item.get("abs_url")),
                "pdf_url": normalize_text(item.get("pdf_url")),
                "comment": normalize_text(item.get("comment")),
                "authors_joined": authors_joined,
                "categories_joined": categories_joined,
                "summary_chars": len(summary),
                "text_for_embedding": text_for_embedding,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["paper_id"])
    df = df[
        (df["paper_id"] != "")
        & (df["title"] != "")
        & (df["summary"] != "")
        & (df["text_for_embedding"] != "")
    ]
    df = df.sort_values(
        ["published", "paper_id"],
        ascending=[False, True],
        na_position="last",
    )
    return df.reset_index(drop=True)
