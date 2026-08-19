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

    rows, cols = np.where(df.isnull())
    plt.scatter(rows, cols, marker='|')
    plt.yticks(np.arange(len(df.columns)), df.columns)
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()
