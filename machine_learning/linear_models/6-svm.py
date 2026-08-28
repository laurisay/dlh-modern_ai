#!/usr/bin/env python3
"""Module that builds an SVM classifier with a specified kernel."""
from sklearn import svm


def get_SVM_model(name, random_state):
    """
    Build a Support Vector Machine (SVM) classifier using Scikit-learn
    with the specified kernel.

    Args:
        name: A string indicating the type of kernel to use. Accepted
            values are 'linear', 'poly', or 'rbf'.
        random_state: The seed used by the random number generator for
            reproducibility.

    Returns:
        An untrained instance of SVC.
    """
    model = svm.SVC(kernel=name, random_state=random_state)

    return model
