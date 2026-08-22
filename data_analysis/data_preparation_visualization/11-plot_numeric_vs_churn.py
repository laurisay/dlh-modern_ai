#!/usr/bin/env python3
"""
This module provides a function to compare the distribution of
a continuous numeric feature between churned and non-churned
customers in a pandas DataFrame.
"""
import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    Compares continuous numeric feature distributions by churn.

    Plots grouped histograms of the given numeric column, split
    by the Churn status ('Yes' vs 'No').

    Args:
        df (pandas.DataFrame): The input DataFrame, must contain
            a 'Churn' column.
        col (str): Name of the numeric column to plot.

    Returns:
        None
    """
    no_churn = df[df['Churn'] == 'No'][col]
    yes_churn = df[df['Churn'] == 'Yes'][col]

    plt.figure(figsize=(12, 8))
    plt.hist([no_churn, yes_churn], bins=30, label=['No', 'Yes'])

    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title='Churn')

    plt.show()
