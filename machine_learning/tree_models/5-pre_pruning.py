#!/usr/bin/env python3
"""Module that performs a Grid Search for pre-pruning hyperparameters."""
from sklearn import model_selection


def prepruning(X, y, clf):
    """
    Perform a Grid Search for the best pre-pruning hyperparameters for a
    decision tree classifier.

    Args:
        X: Input features.
        y: Target labels.
        clf: An untrained DecisionTreeClassifier instance.

    Returns:
        A dictionary containing the best combination of hyperparameters
        found during the grid search.
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': range(2, 5),
        'min_samples_leaf': range(2, 5),
        'min_samples_split': range(2, 5),
    }

    grid_search = model_selection.GridSearchCV(clf, param_grid)
    grid_search.fit(X, y)

    return grid_search.best_params_
