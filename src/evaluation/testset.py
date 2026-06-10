from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.utils import write_json


def build_test_set(df: pd.DataFrame, output_path) -> list[dict[str, Any]]:
    """Tao bo evaluation set tu cleaned dataframe."""
    required_cols = {"paper_id", "title", "summary"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataframe is missing required columns: {sorted(missing_cols)}")

    clean_df = df.copy()
    clean_df = clean_df.dropna(subset=["paper_id", "title", "summary"])
    clean_df = clean_df[
        (clean_df["paper_id"].astype(str).str.strip() != "")
        & (clean_df["title"].astype(str).str.strip() != "")
        & (clean_df["summary"].astype(str).str.strip() != "")
    ]

    if len(clean_df) < 3:
        raise ValueError("Need at least 3 valid documents to build a test set.")

    if "published" in clean_df.columns:
        clean_df = clean_df.sort_values("published", ascending=False, na_position="last")

    selected = clean_df.head(min(6, len(clean_df)))
    test_set: list[dict[str, Any]] = []

    def add_case(question_type: str, question: str, ground_truth: Any, paper_id: Any) -> None:
        if ground_truth is None:
            return

        ground_truth_text = str(ground_truth).strip()
        if not ground_truth_text:
            return

        test_set.append(
            {
                "id": f"q{len(test_set) + 1:03d}",
                "question_type": question_type,
                "question": question,
                "ground_truth": ground_truth_text,
                "ground_truth_doc_ids": [str(paper_id)],
            }
        )

    for _, row in selected.iterrows():
        paper_id = row["paper_id"]
        title = str(row["title"]).strip()

        add_case(
            "summary",
            f"What is the main idea of the paper titled '{title}'?",
            row.get("summary", ""),
            paper_id,
        )

        authors = row.get("authors_joined", "")
        if not authors and "authors" in row and isinstance(row.get("authors"), list):
            authors = ", ".join(str(author).strip() for author in row["authors"] if str(author).strip())
        add_case(
            "authors",
            f"List the authors of the paper titled '{title}'.",
            authors,
            paper_id,
        )

        add_case(
            "date",
            f"When was the paper titled '{title}' published?",
            row.get("published", ""),
            paper_id,
        )

        categories = row.get("categories_joined", "")
        if not categories and "categories" in row and isinstance(row.get("categories"), list):
            categories = ", ".join(
                str(category).strip() for category in row["categories"] if str(category).strip()
            )
        add_case(
            "categories",
            f"What categories are associated with the paper titled '{title}'?",
            categories,
            paper_id,
        )

    write_json(Path(output_path), test_set)
    return test_set
