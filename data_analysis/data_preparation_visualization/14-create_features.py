#!/usr/bin/env python3
"""
This module provides a function to engineer new features from
the Telco customer churn dataset: a count of subscribed
services and a binned tenure group.
"""
import pandas as pd


def create_features(df):
    """
    Engineers new features from the dataset.

    Creates two new columns:
        - NumServices: number of services the customer is
          subscribed to, counting 'Yes' in service-related
          columns (excluding PhoneService). For InternetService,
          'DSL' and 'Fiber optic' count as subscribed, and 'No'
          does not.
        - TenureGroup: categorical binning of the 'tenure' column
          into '0-12', '13-24', '25-48', '49-60', '60+' (0
          excluded, upper bounds inclusive).

    The original columns used to build these features are
    dropped from the returned DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        pandas.DataFrame: The DataFrame with the new features.
    """
    df = df.copy()

    service_cols = [
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
    ]

    def count_services(row):
        count = 0
        for col in service_cols:
            if col == 'InternetService':
                if row[col] in ('DSL', 'Fiber optic'):
                    count += 1
            elif row[col] == 'Yes':
                count += 1
        return count

    df['NumServices'] = df.apply(count_services, axis=1)

    bins = [0, 12, 24, 48, 60, float('inf')]
    labels = ['0-12', '13-24', '25-48', '49-60', '60+']
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=bins, labels=labels, right=True
    )

    df.drop(columns=service_cols + ['tenure'], inplace=True)

    return df
