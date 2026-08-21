"""
Visualization & Publication Table Reporting Engine.
"""

from src.visualization.plots import (
    plot_pareto_frontier,
    plot_latency_breakdown,
    plot_speedup_barchart,
    plot_tail_latencies,
    plot_memory_footprints,
    plot_stability_trends,
    plot_all_figures,
)
from src.visualization.report_tables import generate_all_tables

__all__ = [
    "plot_pareto_frontier",
    "plot_latency_breakdown",
    "plot_speedup_barchart",
    "plot_tail_latencies",
    "plot_memory_footprints",
    "plot_stability_trends",
    "plot_all_figures",
    "generate_all_tables",
]
