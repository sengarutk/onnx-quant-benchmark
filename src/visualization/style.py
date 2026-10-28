"""
Shared matplotlib style for all publication figures.

Figures are drawn at their printed size in the IEEE two-column paper (one column = 3.5 in, full text width =
7.16 in) so that the font sizes below are the sizes that appear in print; LaTeX includes them without scaling.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLUMN_WIDTH_IN = 3.5
TEXT_WIDTH_IN = 7.16
DPI = 300

# Smallest annotation size used anywhere; everything stays legible when printed at 100%.
ANNOTATION_FONTSIZE = 6.0


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8,
        "figure.titlesize": 8.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "patch.linewidth": 0.5,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
    })


apply_style()
