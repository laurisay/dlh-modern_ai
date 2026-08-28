#!/usr/bin/env python3
"""Module that retrieves the cost-complexity pruning path."""


def get_pruning_path(clf, X, y):
    """
    Retrieve the cost-complexity pruning path for a decision tree
    classifier.

    Args:
        clf: A DecisionTreeClassifier instance.
        X: Input features.
        y: Target labels.

    Returns:
        ccp_alphas: A NumPy array containing the effective alpha values
            used for pruning.
        impurities: A NumPy array containing the total impurity of
            leaves at each corresponding alpha.
    """
    path = clf.cost_complexity_pruning_path(X, y)
    ccp_alphas = path.ccp_alphas
    impurities = path.impurities

    return ccp_alphas, impurities
