#!/usr/bin/env python3
"""
Plot the distribution of churn.
"""
import matplotlib.pyplot as plt


def plot_churn_distribution(df):
    """
    Plot the distribution of the Churn column.
    """
    plt.figure(figsize=(12, 8))

    counts = df['Churn'].value_counts()

    plt.bar(
        counts.index,
        counts.values,
        color=['skyblue' if x == 'No' else 'salmon'
               for x in counts.index]
    )

    plt.title('Churn Distribution')
    plt.xlabel('Churn')
    plt.ylabel('Number of Customers')

    plt.show()
