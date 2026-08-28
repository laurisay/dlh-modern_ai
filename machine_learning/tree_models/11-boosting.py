#!/usr/bin/env python3
"""Module that initializes boosting classifiers."""
from sklearn import ensemble
import xgboost as xgb
import lightgbm as lgb


def compare_boosting_classifiers(name, n_estimators, random_state):
    """
    Initialize and return an untrained boosting classifier based on the
    specified algorithm name.

    Args:
        name: Name of the boosting algorithm. Must be one of 'adaboost',
            'gradientboosting', 'xgboost', or 'lightgbm'.
        n_estimators: Number of boosting iterations (trees).
        random_state: Random seed for reproducibility.

    Returns:
        An untrained instance of the selected boosting classifier.
    """
    if name == 'adaboost':
        model = ensemble.AdaBoostClassifier(
            n_estimators=n_estimators, random_state=random_state)
    elif name == 'gradientboosting':
        model = ensemble.GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state)
    elif name == 'xgboost':
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, random_state=random_state)
    elif name == 'lightgbm':
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators, random_state=random_state,
            verbose=-1)
    else:
        raise ValueError(f"Unknown model name '{name}'")

    return model
