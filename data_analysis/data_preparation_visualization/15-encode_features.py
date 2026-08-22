#!/usr/bin/env python3
"""
This module provides a function to encode the features of the
Telco customer churn dataset for modeling, using scikit-learn's
LabelEncoder and OrdinalEncoder alongside pandas one-hot
encoding.
"""
import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes features of the dataset for modeling.

    Encoding applied:
        - Churn: LabelEncoder (No -> 0, Yes -> 1)
        - Partner, Dependents, PaperlessBilling, SeniorCitizen:
          OrdinalEncoder (No -> 0, Yes -> 1)
        - Contract, PaymentMethod: one-hot encoding with the
          first category dropped
        - TenureGroup: OrdinalEncoder using alphabetical order

    Args:
        df (pandas.DataFrame): The input DataFrame.

    Returns:
        tuple:
            pandas.DataFrame: The encoded DataFrame.
            LabelEncoder: The fitted LabelEncoder for Churn.
            OrdinalEncoder: The fitted OrdinalEncoder for the
                binary columns.
            OrdinalEncoder: The fitted OrdinalEncoder for
                TenureGroup.
    """
    df = df.copy()

    churn_le = preprocessing.LabelEncoder()
    df['Churn'] = churn_le.fit_transform(df['Churn'])

    binary_cols = [
        'Partner', 'Dependents', 'PaperlessBilling', 'SeniorCitizen'
    ]
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for col in binary_cols:
        df[col] = binary_oe.fit_transform(df[[col]])
    df[binary_cols] = df[binary_cols].astype(int)

    tenure_oe = preprocessing.OrdinalEncoder()
    df['TenureGroup'] = tenure_oe.fit_transform(
        df[['TenureGroup']].astype(str)
    )
    df['TenureGroup'] = df['TenureGroup'].astype(int)

    df = pd.get_dummies(
        df, columns=['Contract', 'PaymentMethod'],
        drop_first=True, dtype=int
    )

    return df, churn_le, binary_oe, tenure_oe
