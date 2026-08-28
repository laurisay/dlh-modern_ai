#!/usr/bin/env python3
"""Module that displays the textual structure of a trained decision tree."""
from sklearn import tree


def draw(clf, feature_names, class_names):
    """
    Display the textual structure of a trained decision tree classifier.

    Args:
        clf: A trained DecisionTreeClassifier instance from Scikit-learn.
        feature_names: A list of the input feature names.
        class_names: A list of the target class names.

    Returns:
        None. Prints a readable text representation of the tree structure.
    """
    tree_rules = tree.export_text(
        clf, feature_names=feature_names, class_names=class_names)
    print(tree_rules)
