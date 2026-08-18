#!/usr/bin/env python3
"""
Visualize missing values in a DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Visualizes missing values in a DataFrame.
    """
    plt.figure(figsize=(12, 8))

    missing = df.isnull()

    for col in range(len(df.columns)):
        rows = np.where(missing.iloc[:, col])[0]
        plt.scatter(rows, [col] * len(rows), marker='|')

    plt.yticks(np.arange(len(df.columns)), df.columns)

    plt.tight_layout()
    plt.show()
