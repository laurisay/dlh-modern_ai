#!/usr/bin/env python3
"""
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()

    colors = ['skyblue' if value == 'No' else 'salmon'
              for value in counts.index]

    plt.bar(counts.index, counts.values, color=colors)

    plt.show()
