#!/usr/bin/env python3
"""Module that trains and evaluates decision trees with pruning."""
from sklearn import tree
train_tree = __import__('1-train').train_tree


def prune_and_evaluate_trees(X_train, y_train, X_test, y_test, ccp_alphas,
                              random_state, min_samples_leaf,
                              min_samples_split):
    """
    Train multiple decision tree classifiers over a range of
    cost-complexity pruning parameters and evaluate their performance.

    Args:
        X_train: Training data.
        y_train: Training labels.
        X_test: Testing data.
        y_test: Testing labels.
        ccp_alphas: A NumPy array of pruning alpha values to use for
            training different trees.
        random_state: Integer seed for reproducibility.
        min_samples_leaf: Minimum number of samples required at a leaf
            node.
        min_samples_split: Minimum number of samples required to split
            an internal node.

    Returns:
        clfs: A list of trained DecisionTreeClassifier instances, each
            corresponding to a ccp_alpha value.
        train_scores: A list of training accuracy scores for each
            classifier.
        test_scores: A list of testing accuracy scores for each
            classifier.
    """
    clfs = []

    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            ccp_alpha=ccp_alpha
        )
        train_tree(clf, X_train, y_train)
        clfs.append(clf)

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    return clfs, train_scores, test_scores
