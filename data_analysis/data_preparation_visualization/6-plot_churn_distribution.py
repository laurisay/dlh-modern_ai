#!/usr/bin/env python3
"""Plot the distribution of churn."""

import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """Plot the distribution of the Churn column."""
    counts = df['Churn'].value_counts()
    plt.bar(counts.index, counts.values,
            color=['skyblue', 'salmon'])
    plt.show()
