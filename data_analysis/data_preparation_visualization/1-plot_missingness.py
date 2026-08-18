#!/usr/bin/env python3
"""Function to visualize missing values in a DataFrame."""

import matplotlib.pyplot as plt
import numpy as np


def plot_missingness(df):
    """Visualizes missing values in a DataFrame."""
    plt.figure(figsize=(12, 8))

    plt.imshow(df.isnull().T, aspect='auto', interpolation='none')

    plt.xlabel('Rows')
    plt.ylabel('Columns')
    plt.yticks(np.arange(len(df.columns)), df.columns)

    plt.show()
