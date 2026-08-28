#!/usr/bin/env python3
"""Module that generates SHAP explanations for a trained model."""
import shap


def get_shap_explainer_and_values(model, X_train, X_test):
    """
    Generate model explanations using the SHAP library.

    Args:
        model: A trained regression model.
        X_train: Input data used to initialize the explainer.
        X_test: Input data to explain.

    Returns:
        explainer: SHAP explainer object.
        shap_values: SHAP values for the predictions on X_test.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    return explainer, shap_values
