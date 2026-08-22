#!/usr/bin/env python3
"""
Module for plotting a correlation heatmap.
"""
import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Visualize correlations between continuous numerical features.
    """
    plt.figure(figsize=(6, 5))

    correlation_matrix = df.corr(numeric_only=True)

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    plt.show()
