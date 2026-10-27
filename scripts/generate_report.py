#!/usr/bin/env python3
"""
Aggregates one benchmark session and regenerates results/runs.csv, results/runs.json, tables 1-5,
the six session figures and the deployment decision matrix.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.aggregate_results import aggregate_benchmark_runs, list_sessions, load_session_records
from src.analysis.decision_matrix import synthesize_decision_matrix
from src.common.logging import setup_logger
from src.visualization.plots import plot_all_figures
from src.visualization.report_tables import generate_all_tables

logger = setup_logger("generate_report")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--session", default=None, help="Session id (default: latest completed full session)")
    parser.add_argument("--include-quick", action="store_true", help="Allow selecting a --quick smoke session")
    args = parser.parse_args()

    results = Path(args.results_dir)
    raw = results / "raw"
    session = args.session or list_sessions(raw, include_quick=args.include_quick)[-1].name
    df = aggregate_benchmark_runs(raw, results / "runs.csv", session_id=session)
    records = load_session_records(raw / session)

    # Commit-friendly copy of the selected session's full run records (raw/ itself is git-ignored)
    session_meta = json.loads((raw / session / "session.json").read_text(encoding="utf-8"))
    (results / "runs.json").write_text(json.dumps({"session": session_meta, "runs": records}, indent=2), encoding="utf-8")

    generate_all_tables(df, results / "tables")
    figures = plot_all_figures(df, records, results / "figures")
    synthesize_decision_matrix(df, results / "tables" / "deployment_decision_matrix.md")
    logger.info(f"Report for {session}: {len(df)} runs, 5 tables, {len(figures)} figures, decision matrix.")


if __name__ == "__main__":
    main()
