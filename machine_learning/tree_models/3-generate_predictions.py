#!/usr/bin/env python3
"""Module that generates predictions from a trained tree-based classifier."""


def generate_predictions(clf, X):
    """
    Generate predictions from a trained tree-based classifier.

    Args:
        clf: A trained Scikit-learn classifier instance.
        X: Feature matrix (NumPy array or pandas DataFrame).

    Returns:
        A NumPy array containing the predicted class labels for the
        input samples.
    """
    return clf.predict(X)
