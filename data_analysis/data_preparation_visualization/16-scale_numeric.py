#!/usr/bin/env python3
"""
This module provides a function to standardize the numeric
columns MonthlyCharges and TotalCharges of the Telco customer
churn dataset.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Standardizes numeric columns to mean 0 and std 1.

    Scales the 'MonthlyCharges' and 'TotalCharges' columns using
    scikit-learn's StandardScaler.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with scaled columns.
    """
    df = df.copy()

    cols = ['MonthlyCharges', 'TotalCharges']
    scaler = preprocessing.StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df
