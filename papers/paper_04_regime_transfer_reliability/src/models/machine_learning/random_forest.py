"""Reusable implementation of `random_forest` with locked parameters."""
from __future__ import annotations
from typing import Any, Mapping

from shared.models import estimator_protocol

DEFAULT_PARAMETERS: dict[str, Any] = {'n_estimators': 60, 'max_depth': 8, 'min_samples_leaf': 20, 'max_features': 'sqrt', 'max_samples': 0.8, 'class_weight': 'balanced_subsample', 'n_jobs': 1}

def get_default_parameters() -> dict[str, Any]:
    """Return a copy of the locked pre-refactor parameters."""
    return estimator_protocol.get_default_parameters(DEFAULT_PARAMETERS)

def validate_parameters(parameters: Mapping[str, Any]) -> None:
    """Reject unknown parameters and invalid non-positive iteration counts."""
    estimator_protocol.validate_parameters(parameters, DEFAULT_PARAMETERS)
    for key in ("n_estimators", "max_iter", "min_samples_leaf"):
        if key in parameters and int(parameters[key]) <= 0:
            raise ValueError(f"{key} must be positive")

def build_model(seed: int = 42, **overrides: Any) -> Any:
    """Build the estimator without hard-coded data or paths."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    parameters = get_default_parameters()
    parameters.update(overrides)
    validate_parameters(parameters)
    estimator = RandomForestClassifier(random_state=seed, **parameters)
    return estimator

def fit_model(model: Any, X: Any, y: Any, **fit_kwargs: Any) -> Any:
    """Fit an already-built estimator on caller-supplied data."""
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must be non-empty and aligned")
    model.fit(X, y, **fit_kwargs)
    return model

def predict(model: Any, X: Any) -> Any:
    """Return predictions at the estimator's native semantics."""
    if len(X) == 0:
        raise ValueError("X cannot be empty")
    return model.predict(X)

def predict_proba(model: Any, X: Any) -> Any:
    """Return positive-class probabilities when the estimator supports them."""
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)
        return probability[:, 1] if getattr(probability, "ndim", 1) == 2 else probability
    if hasattr(model, "decision_function"):
        import numpy as np
        score = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))
    raise TypeError("model does not expose probability or decision scores")
