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
    continuous_columns = [
        column for column in df.select_dtypes(
            include='number'
        ).columns
        if df[column].nunique() > 2
    ]

    correlation_matrix = df[continuous_columns].corr()

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1
    )

    plt.show()
