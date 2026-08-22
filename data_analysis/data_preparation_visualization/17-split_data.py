#!/usr/bin/env python3
"""
This module provides a function to split a dataset into
stratified train and test sets.
"""
from sklearn import model_selection


def split_data(df, target='Churn', test_size=0.2, random_state=42):
    """
    Splits data into train/test sets using stratified sampling.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        target (str): Name of the target column. Defaults to
            'Churn'.
        test_size (float): Proportion of the dataset to include
            in the test split. Defaults to 0.2.
        random_state (int): Random seed for reproducibility.
            Defaults to 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
