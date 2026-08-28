#!/usr/bin/env python3
"""Module that builds a Ridge Regression model."""
from sklearn import linear_model


def ridge_regression(random_state):
    """
    Build a Ridge Regression model using Scikit-learn.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained Ridge regression model instance.
    """
    model = linear_model.Ridge(random_state=random_state)

    return model
