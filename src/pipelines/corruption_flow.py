from __future__ import annotations

import pandas as pd

from core.config import load_settings
from core.utils import now_utc, read_json, write_csv, write_json
from evaluation.metrics import evaluate_pipeline
from ingestion.cleaning import build_clean_dataframe
from ingestion.corruption import corrupt_clean_dataframe
from ingestion.crossref import load_raw_records
from observability.quality import build_freshness_report, run_data_quality_checks
from observability.reporting import generate_corruption_report
from retrieval.index import LocalEmbeddingIndex


def _require_file(path, message: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{message}: {path}")


def _load_clean_dataframe(settings) -> pd.DataFrame:
    if settings.paths.clean_json.exists():
        return pd.DataFrame(read_json(settings.paths.clean_json))

    _require_file(
        settings.paths.clean_csv,
        "Clean dataset not found. Run Phase 1 before corruption flow",
    )
    return pd.read_csv(settings.paths.clean_csv)


def main() -> None:
    """Xay dung corruption -> evaluate -> repair -> compare flow."""
    settings = load_settings()

    _require_file(
        settings.paths.baseline_metrics,
        "Baseline metrics not found. Run Phase 1 before corruption flow",
    )
    _require_file(
        settings.paths.eval_testset,
        "Evaluation test set not found. Run Phase 1 before corruption flow",
    )
    _require_file(
        settings.paths.raw_records_json,
        "Raw records not found. Fetch or run Phase 1 before corruption flow",
    )

    baseline_metrics = read_json(settings.paths.baseline_metrics)
    clean_df = _load_clean_dataframe(settings)

    corrupted_df = corrupt_clean_dataframe(clean_df, settings.paths.corruption_log)
    write_csv(corrupted_df, settings.paths.corrupted_clean_csv)
    write_json(settings.paths.corrupted_clean_json, corrupted_df.to_dict(orient="records"))

    corrupted_index = LocalEmbeddingIndex.build(
        corrupted_df,
        settings=settings,
        embeddings_output_path=settings.paths.corrupted_embeddings_json,
    )
    corrupted_eval = evaluate_pipeline(
        settings=settings,
        index=corrupted_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.corrupted_metrics,
        answers_output_path=settings.paths.corrupted_answers,
    )
    corrupted_quality = run_data_quality_checks(
        corrupted_df,
        settings=settings,
        report_name="corrupted_quality",
    )
    corrupted_freshness = build_freshness_report(
        corrupted_df,
        settings=settings,
        report_path=settings.paths.quality_dir / "corrupted_freshness_report.json",
    )

    raw_records = load_raw_records(settings.paths.raw_records_json)
    repaired_df = build_clean_dataframe(raw_records, now_utc())
    write_csv(repaired_df, settings.paths.repaired_clean_csv)
    write_json(settings.paths.repaired_clean_json, repaired_df.to_dict(orient="records"))

    repaired_index = LocalEmbeddingIndex.build(
        repaired_df,
        settings=settings,
        embeddings_output_path=settings.paths.repaired_embeddings_json,
    )
    repaired_eval = evaluate_pipeline(
        settings=settings,
        index=repaired_index,
        test_set_path=settings.paths.eval_testset,
        metrics_output_path=settings.paths.repaired_metrics,
        answers_output_path=settings.paths.repaired_answers,
    )
    repaired_quality = run_data_quality_checks(
        repaired_df,
        settings=settings,
        report_name="repaired_quality",
    )
    repaired_freshness = build_freshness_report(
        repaired_df,
        settings=settings,
        report_path=settings.paths.quality_dir / "repaired_freshness_report.json",
    )

    generate_corruption_report(
        report_path=settings.paths.comparison_report,
        baseline_metrics=baseline_metrics,
        corrupted_metrics=corrupted_eval.summary,
        repaired_metrics=repaired_eval.summary,
        corrupted_quality=corrupted_quality,
        repaired_quality=repaired_quality,
        corrupted_freshness=corrupted_freshness,
        repaired_freshness=repaired_freshness,
    )


if __name__ == "__main__":
    main()
