#!/usr/bin/env python3
"""
This module provides a function to visualize the churn rate
per category of a given categorical column in a pandas
DataFrame.
"""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Visualizes churn rates per category for a categorical column.

    Computes the proportion of customers with Churn == 'Yes' for
    each category of the given column and displays it as a bar
    plot.

    Args:
        df (pandas.DataFrame): The input DataFrame, must contain
            a 'Churn' column.
        col (str): Name of the categorical column to group by.

    Returns:
        None
    """
    churn_rate = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean())

    plt.figure(figsize=(12, 8))
    churn_rate.plot(kind='bar')

    plt.title(f"Churn Rate by {col}")
    plt.xlabel("")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)

    plt.show()
