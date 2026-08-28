#!/usr/bin/env python3
"""Module that selects the best ccp_alpha for pruning."""


def get_best_alpha(clfs, train_scores, test_scores, ccp_alphas):
    """
    Select the best pruning value ccp_alpha for a set of trained decision
    trees.

    Args:
        clfs: List of trained DecisionTreeClassifier instances, each
            trained with a different ccp_alpha.
        train_scores: List of training accuracy scores corresponding to
            each classifier in clfs.
        test_scores: List of test accuracy scores corresponding to each
            classifier in clfs.
        ccp_alphas: List or array of ccp_alpha values used to train the
            classifiers.

    Returns:
        best_alpha: The most appropriate ccp_alpha value based on test
            accuracy and generalization.
        best_clf: The trained classifier associated with the best alpha.
    """
    best_test_score = max(test_scores)

    candidates = [
        i for i in range(len(clfs)) if test_scores[i] == best_test_score
    ]

    min_diff = min(
        abs(train_scores[i] - test_scores[i]) for i in candidates
    )
    candidates = [
        i for i in candidates
        if abs(train_scores[i] - test_scores[i]) == min_diff
    ]

    best_index = max(candidates, key=lambda i: ccp_alphas[i])

    best_alpha = ccp_alphas[best_index]
    best_clf = clfs[best_index]

    return best_alpha, best_clf
