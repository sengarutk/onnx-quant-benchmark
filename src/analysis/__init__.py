"""
Analysis & Operational Reporting Subsystem.
"""

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.analysis.decision_flips import compute_detection_flips
from src.analysis.decision_matrix import synthesize_decision_matrix
from src.analysis.pareto import get_model_pareto_summary, identify_pareto_frontier
from src.analysis.stats import bootstrap_confidence_interval, wilcoxon_paired_test

# Convenience aliases
generate_decision_matrix = synthesize_decision_matrix
compute_pareto_frontier = identify_pareto_frontier

__all__ = [
    "aggregate_benchmark_runs",
    "compute_detection_flips",
    "synthesize_decision_matrix",
    "generate_decision_matrix",
    "identify_pareto_frontier",
    "compute_pareto_frontier",
    "get_model_pareto_summary",
    "bootstrap_confidence_interval",
    "wilcoxon_paired_test",
]
