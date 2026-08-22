#!/usr/bin/env python3
"""
This module provides a function to perform Welch's t-tests
comparing continuous numeric features between churned and
non-churned customers in a pandas DataFrame.
"""
from scipy import stats


def ttest_numeric(df):
    """
    Performs Welch's t-tests for continuous numeric features.

    For each continuous numeric column in the DataFrame, compares
    the Churn='Yes' group against the Churn='No' group using
    Welch's t-test (unequal variances assumed).

    H0: The means of the variable are equal in the Churn='Yes'
        and Churn='No' groups.
    H1: The means differ significantly.

    Args:
        df (pandas.DataFrame): The input DataFrame, must contain
            a 'Churn' column.

    Returns:
        dict: Mapping of feature_name -> p_value.
    """
    numeric_cols = df.select_dtypes(include='number').columns
    continuous_cols = [c for c in numeric_cols if df[c].nunique() > 2]

    yes_group = df[df['Churn'] == 'Yes']
    no_group = df[df['Churn'] == 'No']

    results = {}
    for col in continuous_cols:
        _, p_value = stats.ttest_ind(
            yes_group[col], no_group[col], equal_var=False
        )
        results[col] = p_value

    return results
