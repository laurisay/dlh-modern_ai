#!/usr/bin/env python3
"""Module that builds a Lasso Regression model."""
from sklearn import linear_model


def lasso_regression(random_state):
    """
    Build a Lasso Regression model using Scikit-learn.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained Lasso regression model instance.
    """
    model = linear_model.Lasso(random_state=random_state)

    return model
