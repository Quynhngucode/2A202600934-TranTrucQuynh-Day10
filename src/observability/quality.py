from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import Settings
from core.utils import write_json


def run_data_quality_checks(df: pd.DataFrame, settings: Settings, report_name: str) -> dict[str, Any]:
    """Tao bo data quality checks va ghi ket qua vao data/quality."""
    settings.paths.quality_dir.mkdir(parents=True, exist_ok=True)

    paper_id_missing = int(df["paper_id"].isna().sum() + (df["paper_id"].astype(str).str.strip() == "").sum())
    duplicate_paper_ids = int(df["paper_id"].duplicated().sum())
    title_missing = int(df["title"].isna().sum() + (df["title"].astype(str).str.strip() == "").sum())
    summary_missing = int(df["summary"].isna().sum() + (df["summary"].astype(str).str.strip() == "").sum())
    short_summaries = int((df["summary"].astype(str).str.len() < 50).sum())

    age_series = pd.to_numeric(df.get("age_days"), errors="coerce")
    stale_rows = int((age_series > settings.freshness_threshold_days).sum())

    checks = {
        "report_name": report_name,
        "row_count": int(len(df)),
        "checks": {
            "has_rows": int(len(df)) > 0,
            "paper_id_not_null": paper_id_missing == 0,
            "paper_id_unique": duplicate_paper_ids == 0,
            "title_not_null": title_missing == 0,
            "summary_not_null": summary_missing == 0,
            "summary_min_length": short_summaries == 0,
            "freshness_threshold_days": settings.freshness_threshold_days,
            "fresh_enough": stale_rows == 0,
        },
        "stats": {
            "paper_id_missing": paper_id_missing,
            "duplicate_paper_ids": duplicate_paper_ids,
            "title_missing": title_missing,
            "summary_missing": summary_missing,
            "short_summaries": short_summaries,
            "stale_rows": stale_rows,
        },
    }
    checks["passed"] = all(
        value
        for key, value in checks["checks"].items()
        if key != "freshness_threshold_days"
    )

    write_json(settings.paths.quality_dir / f"{report_name}.json", checks)
    return checks


def build_freshness_report(df: pd.DataFrame, settings: Settings, report_path) -> dict[str, Any]:
    """Tong hop freshness report."""
    published = pd.to_datetime(df.get("published"), errors="coerce")
    age_days = pd.to_numeric(df.get("age_days"), errors="coerce")
    stale_mask = age_days > settings.freshness_threshold_days

    report = {
        "latest_published": published.max().date().isoformat() if published.notna().any() else None,
        "oldest_published": published.min().date().isoformat() if published.notna().any() else None,
        "stale_rows": int(stale_mask.sum()),
        "total_rows": int(len(df)),
        "freshness_threshold_days": settings.freshness_threshold_days,
        "is_fresh": bool(len(df) > 0 and stale_mask.sum() == 0),
    }

    write_json(report_path, report)
    return report
