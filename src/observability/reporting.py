from __future__ import annotations

from typing import Any

from core.utils import write_text


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def generate_phase1_report(
    report_path,
    source_summary: dict[str, Any],
    metrics: dict[str, Any],
    quality: dict[str, Any],
    freshness: dict[str, Any],
) -> None:
    """Viet markdown report cho baseline phase."""
    lines = [
        "# Phase 1 Baseline Report",
        "",
        "## Source",
        f"- Source: {source_summary.get('source')}",
        f"- Query: {source_summary.get('query')}",
        f"- Filter: {source_summary.get('filter')}",
        f"- Raw records: {source_summary.get('raw_records')}",
        f"- Clean records: {source_summary.get('clean_records')}",
        "",
        "## Evaluation",
        f"- Samples: {metrics.get('samples')}",
        f"- Retrieval hit rate: {_fmt(metrics.get('retrieval_hit_rate'))}",
        f"- Mean token F1: {_fmt(metrics.get('mean_token_f1'))}",
        f"- Judge accuracy: {_fmt(metrics.get('judge_accuracy'))}",
        f"- Mean judge score: {_fmt(metrics.get('mean_judge_score'))}",
        "",
        "## Data Quality",
        f"- Passed: {quality.get('passed')}",
        f"- Row count: {quality.get('row_count')}",
    ]

    for name, value in quality.get("stats", {}).items():
        lines.append(f"- {name}: {value}")

    lines.extend(
        [
            "",
            "## Freshness",
            f"- Latest published: {freshness.get('latest_published')}",
            f"- Oldest published: {freshness.get('oldest_published')}",
            f"- Stale rows: {freshness.get('stale_rows')}",
            f"- Total rows: {freshness.get('total_rows')}",
            f"- Is fresh: {freshness.get('is_fresh')}",
            "",
        ]
    )

    write_text(report_path, "\n".join(lines))


def generate_corruption_report(
    report_path,
    baseline_metrics: dict[str, Any],
    corrupted_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    corrupted_quality: dict[str, Any],
    repaired_quality: dict[str, Any],
    corrupted_freshness: dict[str, Any],
    repaired_freshness: dict[str, Any],
) -> None:
    """Viet markdown report so sanh baseline/corrupted/repaired."""
    lines = [
        "# Corruption Comparison Report",
        "",
        "## Metrics",
        "| Dataset | Retrieval hit rate | Mean token F1 | Judge accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, metrics in [
        ("Baseline", baseline_metrics),
        ("Corrupted", corrupted_metrics),
        ("Repaired", repaired_metrics),
    ]:
        lines.append(
            "| "
            f"{label} | "
            f"{_fmt(metrics.get('retrieval_hit_rate'))} | "
            f"{_fmt(metrics.get('mean_token_f1'))} | "
            f"{_fmt(metrics.get('judge_accuracy'))} |"
        )

    lines.extend(
        [
            "",
            "## Quality",
            f"- Corrupted passed: {corrupted_quality.get('passed')}",
            f"- Repaired passed: {repaired_quality.get('passed')}",
            "",
            "## Freshness",
            f"- Corrupted stale rows: {corrupted_freshness.get('stale_rows')}",
            f"- Repaired stale rows: {repaired_freshness.get('stale_rows')}",
            "",
        ]
    )

    write_text(report_path, "\n".join(lines))
