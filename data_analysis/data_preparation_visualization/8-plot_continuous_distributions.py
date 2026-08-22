#!/usr/bin/env python3
"""
This module provides a function to visualize the distributions
of continuous numerical features in a pandas DataFrame using
histograms with KDE curves and box plots.
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualizes the distributions of continuous numerical features.

    For each selected column, generates a histogram with a KDE
    overlay on the left and a corresponding box plot on the right.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        columns_to_plot (list, optional): List of continuous numeric
            column names to plot. If None, all numeric columns in
            the DataFrame are selected.

    Returns:
        None
    """
    if columns_to_plot is None:
        numeric_df = df.select_dtypes(include=[np.number])
        columns_to_plot = numeric_df.columns.tolist()

    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3 * n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(columns_to_plot):
        data = df[col].dropna()

        # Left subplot: Histogram + KDE
        ax_hist = axes[i, 0]
        ax_hist.hist(data, bins=30, density=True, alpha=0.7, edgecolor='black')

        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax_hist.plot(x_range, kde(x_range), color='red', linestyle='--')

        ax_hist.set_title(f"{col} Histogram + KDE")

        # Right subplot: Box Plot
        ax_box = axes[i, 1]
        ax_box.boxplot(data, vert=False, whis=(0, 100))
        ax_box.set_title(f"{col} Boxplot")

    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
