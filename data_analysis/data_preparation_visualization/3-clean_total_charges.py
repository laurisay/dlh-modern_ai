#!/usr/bin/env python3
"""Handle missing values in TotalCharges."""


def clean_total_charges(df, method='drop'):
    """Handle missing TotalCharges using the specified method."""
    df = df.copy()

    if method == 'drop':
        return df.dropna(subset=['TotalCharges'])

    if method == 'median':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['TotalCharges'].median()
        )
        return df

    if method == 'impute':
        df['TotalCharges'] = df['TotalCharges'].fillna(
            df['MonthlyCharges'] * df['tenure']
        )
        return df

    return df
