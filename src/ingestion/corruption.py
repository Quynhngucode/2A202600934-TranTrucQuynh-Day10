from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from core.utils import normalize_whitespace, write_json


def _rebuild_text_for_embedding(row: pd.Series) -> str:
    return normalize_whitespace(
        " ".join(
            str(part)
            for part in [
                row.get("title", ""),
                row.get("summary", ""),
                row.get("authors_joined", ""),
                row.get("categories_joined", ""),
            ]
            if str(part).strip()
        )
    )


def _shift_date_back(value: Any, days: int) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value or "")

    return (parsed.date() - timedelta(days=days)).isoformat()


def _age_days(value: Any, run_day: date) -> int | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None

    return (run_day - parsed.date()).days


def corrupt_clean_dataframe(df: pd.DataFrame, output_log_path) -> pd.DataFrame:
    """Simulate nhieu dang data corruption va ghi corruption log."""
    if df.empty:
        write_json(output_log_path, {"corruptions": [], "input_rows": 0, "output_rows": 0})
        return df.copy()

    corrupted = df.copy().reset_index(drop=True)
    log: list[dict[str, Any]] = []
    run_day = pd.Timestamp.utcnow().date()

    latest_count = min(2, len(corrupted))
    if latest_count:
        dropped_ids = corrupted.head(latest_count)["paper_id"].astype(str).tolist()
        corrupted = corrupted.iloc[latest_count:].reset_index(drop=True)
        log.append(
            {
                "type": "drop_latest_records",
                "row_count": latest_count,
                "paper_ids": dropped_ids,
            }
        )

    if not corrupted.empty:
        blank_count = min(2, len(corrupted))
        blank_indexes = corrupted.head(blank_count).index.tolist()
        blank_ids = corrupted.loc[blank_indexes, "paper_id"].astype(str).tolist()
        corrupted.loc[blank_indexes, "summary"] = ""
        corrupted.loc[blank_indexes, "summary_chars"] = 0
        log.append(
            {
                "type": "blank_summary",
                "row_count": blank_count,
                "paper_ids": blank_ids,
            }
        )

    if len(corrupted) >= 2:
        noise_indexes = corrupted.tail(min(2, len(corrupted))).index.tolist()
        noise_ids = corrupted.loc[noise_indexes, "paper_id"].astype(str).tolist()
        corrupted.loc[noise_indexes, "summary"] = (
            corrupted.loc[noise_indexes, "summary"].astype(str)
            + " DATA_QUALITY_NOISE token_mismatch stale_context"
        )
        corrupted.loc[noise_indexes, "summary_chars"] = corrupted.loc[noise_indexes, "summary"].str.len()
        log.append(
            {
                "type": "inject_noise",
                "row_count": len(noise_indexes),
                "paper_ids": noise_ids,
            }
        )

    if not corrupted.empty:
        title_index = corrupted.index[min(2, len(corrupted) - 1)]
        paper_id = str(corrupted.loc[title_index, "paper_id"])
        corrupted.loc[title_index, "title"] = str(corrupted.loc[title_index, "title"])[:35]
        log.append(
            {
                "type": "truncate_title",
                "row_count": 1,
                "paper_ids": [paper_id],
            }
        )

    if not corrupted.empty and "published" in corrupted.columns:
        date_index = corrupted.index[min(3, len(corrupted) - 1)]
        paper_id = str(corrupted.loc[date_index, "paper_id"])
        corrupted.loc[date_index, "published"] = _shift_date_back(
            corrupted.loc[date_index, "published"],
            days=365 * 3,
        )
        if "age_days" in corrupted.columns:
            corrupted.loc[date_index, "age_days"] = _age_days(corrupted.loc[date_index, "published"], run_day)
        log.append(
            {
                "type": "stale_published_date",
                "row_count": 1,
                "paper_ids": [paper_id],
            }
        )

    duplicate_count = min(2, len(corrupted))
    if duplicate_count:
        duplicates = corrupted.tail(duplicate_count).copy()
        duplicated_ids = duplicates["paper_id"].astype(str).tolist()
        corrupted = pd.concat([corrupted, duplicates], ignore_index=True)
        log.append(
            {
                "type": "add_duplicate_rows",
                "row_count": duplicate_count,
                "paper_ids": duplicated_ids,
            }
        )

    if "text_for_embedding" in corrupted.columns:
        corrupted["text_for_embedding"] = corrupted.apply(_rebuild_text_for_embedding, axis=1)

    write_json(
        output_log_path,
        {
            "input_rows": int(len(df)),
            "output_rows": int(len(corrupted)),
            "corruptions": log,
        },
    )

    return corrupted.reset_index(drop=True)
