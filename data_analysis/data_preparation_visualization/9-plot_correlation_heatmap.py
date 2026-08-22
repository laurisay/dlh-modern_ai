#!/usr/bin/env python3
"""
This module provides a function to visualize the pairwise
correlations between continuous numeric features of a pandas
DataFrame using an annotated seaborn heatmap.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Visualizes correlations between continuous numeric features.

    Computes the pairwise correlation matrix of the numeric
    columns of the DataFrame and displays it as an annotated
    heatmap using the coolwarm colormap, with the color scale
    fixed between -1 and 1.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        None
    """
    plt.figure(figsize=(6, 5))

    corr = df.corr(numeric_only=True)

    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    plt.title("Correlation Matrix")
    plt.show()
