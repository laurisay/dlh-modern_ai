#!/usr/bin/env python3
"""
This module provides a function to perform chi-square tests of
independence between categorical features and the target
variable Churn in a pandas DataFrame.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Performs chi-square tests for categorical features vs Churn.

    For each categorical column in the DataFrame (excluding the
    target 'Churn' column, the customer identifier, and the
    continuous numeric columns), computes the chi-square p-value
    testing independence between that feature and 'Churn'.

    Args:
        df (pandas.DataFrame): The input DataFrame, must contain
            a 'Churn' column.

    Returns:
        dict: Mapping of feature_name -> p_value.
    """
    excluded = {'Churn', 'customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges'}

    results = {}
    for col in df.columns:
        if col in excluded:
            continue

        contingency_table = pd.crosstab(df[col], df['Churn'])
        _, p_value, _, _ = stats.chi2_contingency(contingency_table)
        results[col] = p_value

    return results
