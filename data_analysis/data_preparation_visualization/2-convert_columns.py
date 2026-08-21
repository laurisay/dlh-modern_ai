#!/usr/bin/env python3
"""
Convert specific DataFrame columns to appropriate types.
"""
import pandas as pd


def convert_columns(df):
    """
    Convert TotalCharges to numeric and SeniorCitizen to categorical strings.
    """
    df['TotalCharges'] = pd.to_numeric(
        df['TotalCharges'], errors='coerce'
    )

    df['SeniorCitizen'] = df['SeniorCitizen'].map({
        0: 'No',
        1: 'Yes'
    })

    return df
