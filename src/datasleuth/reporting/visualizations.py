"""Plotly visualizations for distribution comparisons."""

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def numerical_distribution_figure(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    column: str,
) -> Figure:
    """Return an overlaid histogram identifying reference and comparison data."""
    data = pd.DataFrame(
        {
            column: pd.concat(
                [reference.rename("reference"), comparison.rename("comparison")]
            ).to_numpy(),
            "dataset": ["reference"] * len(reference)
            + ["comparison"] * len(comparison),
        }
    )
    return px.histogram(data, x=column, color="dataset", barmode="overlay")


def categorical_distribution_figure(
    reference: pd.Series,
    comparison: pd.Series,
    *,
    column: str,
) -> Figure:
    """Return a grouped category-count bar chart."""
    data = pd.DataFrame(
        {
            column: pd.concat(
                [reference.rename("reference"), comparison.rename("comparison")]
            ).astype("string"),
            "dataset": ["reference"] * len(reference)
            + ["comparison"] * len(comparison),
        }
    )
    return px.histogram(data, x=column, color="dataset", barmode="group")
