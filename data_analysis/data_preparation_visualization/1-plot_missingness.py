#!/usr/bin/env python3
"""
Plot missing values in a DataFrame.
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """
    Plot the missing values of a DataFrame.
    """
    plt.figure(figsize=(12, 8))

    for i, column in enumerate(df.columns):
        missing = df[column].isna()
        rows = np.where(missing)[0]

        plt.plot(rows, [i] * len(rows), '|')

    plt.yticks(range(len(df.columns)), df.columns)
    plt.title("Missingness Plot")

    plt.tight_layout()
    plt.show()
