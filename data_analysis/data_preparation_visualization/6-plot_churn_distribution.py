#!/usr/bin/env python3
"""
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()

    plt.bar(
        counts.index,
        counts.values,
        color=['skyblue', 'salmon']
    )

    plt.show()
