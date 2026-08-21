#!/usr/bin/env python3
"""
Plot churn distribution
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Visualizes churn class distribution as a bar plot.
    """
    plt.figure(figsize=(12, 8))
    counts = df['Churn'].value_counts()
    plt.bar(counts.index, counts.values,
            color=['skyblue', 'salmon'])
    plt.show()
