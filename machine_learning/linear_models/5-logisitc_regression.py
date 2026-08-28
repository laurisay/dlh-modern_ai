#!/usr/bin/env python3
"""Module that builds a Logistic Regression classifier."""
from sklearn import linear_model


def Logistic_Regression_Model(random_state):
    """
    Build a Logistic Regression model using Scikit-learn.

    Args:
        random_state: An integer used to set the random seed for
            reproducibility.

    Returns:
        model: An untrained LogisticRegression instance.
    """
    model = linear_model.LogisticRegression(random_state=random_state)

    return model
