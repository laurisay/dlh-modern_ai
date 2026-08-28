#!/usr/bin/env python3
"""Module that builds a Linear Regression model."""
from sklearn import linear_model


def Linear_Regression():
    """
    Build a Linear Regression model using Scikit-learn.

    Returns:
        model: An untrained LinearRegression instance.
    """
    model = linear_model.LinearRegression()

    return model
