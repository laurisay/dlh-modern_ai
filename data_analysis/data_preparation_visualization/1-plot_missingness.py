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
        if len(rows) > 0:
            plt.scatter(rows, [col] * len(rows), marker='|', s=100, color='blue')

    plt.yticks(np.arange(len(df.columns)), df.columns)
    plt.xlabel('Row Index')
    plt.ylabel('Columns')
    plt.title('Missing Values Visualization')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
