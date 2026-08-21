"""
Analysis & Operational Reporting Subsystem.
"""

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.analysis.pareto import identify_pareto_frontier, get_model_pareto_summary
from src.analysis.decision_matrix import synthesize_decision_matrix

__all__ = [
    "aggregate_benchmark_runs",
    "identify_pareto_frontier",
    "get_model_pareto_summary",
    "synthesize_decision_matrix",
]
