"""Shared estimator protocol used by every generated model module.

Each ``papers/<id>/src/models/**`` module exposes the same six-symbol
interface — ``get_default_parameters``, ``validate_parameters``,
``build_model``, ``fit_model``, ``predict`` and ``predict_proba``. The bodies
were previously copy-pasted into 46 files; this module holds the one
implementation and the generated modules keep their paper-specific
``DEFAULT_PARAMETERS`` plus their own estimator class.

The behaviour is byte-for-byte equivalent to the previous per-file copies:
parameters are validated against the module's own defaults, ``build_model``
merges caller overrides on top of the defaults, and the fit/predict helpers
simply delegate to the estimator instance.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, TypeVar

ModelT = TypeVar("ModelT")


class SupportsPredict(Protocol):
    def predict(self, X: Any) -> Any: ...


class SupportsPredictProba(Protocol):
    def predict_proba(self, X: Any) -> Any: ...


def get_default_parameters(defaults: Mapping[str, Any]) -> dict[str, Any]:
    """Return a mutable copy of a module's default parameters."""
    return dict(defaults)


def validate_parameters(
    parameters: Mapping[str, Any],
    defaults: Mapping[str, Any],
) -> None:
    """Reject parameters that the estimator does not declare."""
    unknown = set(parameters).difference(defaults)
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")


def resolve_parameters(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge caller overrides onto the defaults and validate the result."""
    parameters = get_default_parameters(defaults)
    parameters.update(overrides or {})
    validate_parameters(parameters, defaults)
    return parameters


def fit_model(model: ModelT, X: Any, y: Any, **fit_kwargs: Any) -> ModelT:
    """Fit ``model`` on ``(X, y)`` and return the fitted instance."""
    return model.fit(X, y, **fit_kwargs)


def predict(model: Any, X: Any) -> Any:
    """Return hard predictions from a fitted estimator."""
    return model.predict(X)


def predict_proba(model: Any, X: Any) -> Any:
    """Return predicted probabilities from a fitted estimator."""
    return model.predict_proba(X)
