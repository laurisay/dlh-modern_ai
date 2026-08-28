#!/usr/bin/env python3
"""Module that builds a Decision Tree Classifier."""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """
    Build a Decision Tree Classifier using Scikit-learn.

    Args:
        min_samples_leaf: Minimum number of samples required to be at a
            leaf node.
        min_samples_split: Minimum number of samples required to split an
            internal node.
        random_state: Seed used by the random number generator.

    Returns:
        model: A Scikit-learn DecisionTreeClassifier instance.
    """
    model = tree.DecisionTreeClassifier(
        criterion='gini',
        max_depth=None,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

    return model
