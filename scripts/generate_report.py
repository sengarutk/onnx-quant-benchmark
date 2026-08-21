"""
CLI Orchestrator for Comprehensive Report Generation.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.analysis.decision_matrix import synthesize_decision_matrix
from src.common.logging import setup_logger
from src.visualization.plots import plot_all_figures
from src.visualization.report_tables import generate_all_tables

logger = setup_logger("generate_report")


def main() -> None:
    raw_dir = PROJECT_ROOT / "results" / "raw"
    runs_csv = PROJECT_ROOT / "results" / "runs.csv"
    tables_dir = PROJECT_ROOT / "results" / "tables"
    figures_dir = PROJECT_ROOT / "results" / "figures"

    logger.info("--- Starting Comprehensive Report Generation ---")

    # 1. Aggregate and deduplicate benchmark runs
    df = aggregate_benchmark_runs(raw_dir, runs_csv)
    logger.info(f"Aggregated {len(df)} run records.")

    # 2. Generate Markdown report tables
    generate_all_tables(df, tables_dir)
    logger.info(f"Generated all 5 Markdown report tables in {tables_dir.relative_to(PROJECT_ROOT)}/.")

    # 3. Render 300-DPI publication figures
    figures = plot_all_figures(df, raw_dir, figures_dir)
    logger.info(f"Rendered all {len(figures)} 300-DPI publication figures in {figures_dir.relative_to(PROJECT_ROOT)}/.")

    # 4. Synthesize Deployment Decision Matrix
    decision_matrix_path = tables_dir / "deployment_decision_matrix.md"
    synthesize_decision_matrix(df, decision_matrix_path)
    logger.info("Synthesized Deployment Decision Matrix.")

    logger.info(">>> Report Generation Completed Successfully! <<<")


if __name__ == "__main__":
    main()
