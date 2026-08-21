#!/usr/bin/env python3
"""
Plot missing values in a DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Plot the missing values in a DataFrame.
    """
    plt.figure(figsize=(12, 8))

    missing = np.where(df.isnull())

    plt.scatter(missing[0], missing[1], marker='|')

    plt.yticks(np.arange(len(df.columns)), df.columns)
    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()
