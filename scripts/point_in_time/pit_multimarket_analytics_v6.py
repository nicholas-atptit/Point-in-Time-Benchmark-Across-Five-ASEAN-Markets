#!/usr/bin/env python3
"""Locked V6 analytics for the selected-market PIT forecasting benchmark.

This module consumes only V6 observations and genuine OOS predictions.  Losses
are first averaged equally within date, then within interval, then within
market, and finally across included markets.  Security-weighted estimates are
retained only as explicitly labelled sensitivity results.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.metrics import balanced_accuracy_score

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_PATH = SCRIPT_PATH.parents[2]
if str(REPOSITORY_PATH) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_PATH))

from scripts.manifest_portability_v6 import validate_portable_record
from scripts.point_in_time import pit_multimarket_forecasting_v6 as core

REPO_ROOT = core.REPO_ROOT
PIT_ROOT = core.PIT_ROOT
REPORT_ROOT = core.REPORT_ROOT
REPORTS = PIT_ROOT / "reports"
PROCESSED = PIT_ROOT / "processed"
QA = PIT_ROOT / "qa"
MANIFEST_ROOT = PIT_ROOT / "manifests"

PIT = core.PIT_SEMANTICS
CURRENT = core.CURRENT_SEMANTICS
INTERSECTION = core.INTERSECTION_SEMANTICS
MODEL_IDS = list(core.MODEL_SPECS)

PERFORMANCE_BASES = [
    "pit_multimarket_performance_by_date_v6",
    "pit_multimarket_performance_by_interval_v6",
    "pit_multimarket_performance_by_market_v6",
    "pit_multimarket_equal_market_macro_v6",
    "pit_multimarket_current_minus_pit_v6",
    "pit_multimarket_bootstrap_inference_v6",
    "pit_multimarket_seed_robustness_v6",
    "pit_multimarket_model_rank_v6",
    "pit_multimarket_intersection_diagnostic_v6",
    "pit_multimarket_invalid_cells_v6",
    "pit_multimarket_leave_one_market_out_v6",
    "pit_multimarket_heterogeneity_v6",
]


def as_date(value: Any) -> date | None:
    return core.as_date(value)


def sha256_file(path: Path) -> str:
    return core.sha256_file(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return as_date(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        parsed = as_date(value)
        return parsed.isoformat() if parsed else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_csv_parquet(base: str, frame: pd.DataFrame) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / f"{base}.csv"
    parquet_path = REPORTS / f"{base}.parquet"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    frame.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def clip_probability(values: Iterable[float]) -> np.ndarray:
    return np.clip(np.asarray(list(values), dtype=float), 1e-15, 1 - 1e-15)


def calibration_parameters(
    actual: Sequence[int], probability: Sequence[float]
) -> tuple[float, float]:
    y = np.asarray(actual, dtype=int)
    p = clip_probability(probability)
    if len(y) < 20 or len(np.unique(y)) < 2 or np.nanstd(p) <= 1e-12:
        return np.nan, np.nan
    logit = np.log(p / (1.0 - p))
    design = np.column_stack([np.ones(len(y)), logit])
    beta = np.zeros(2, dtype=float)
    try:
        for _ in range(50):
            linear = np.clip(design @ beta, -30.0, 30.0)
            fitted = 1.0 / (1.0 + np.exp(-linear))
            weights = np.maximum(fitted * (1.0 - fitted), 1e-9)
            information = design.T @ (design * weights[:, None])
            information += np.eye(2) * 1e-10
            update = np.linalg.solve(
                information, design.T @ (y - fitted)
            )
            beta += update
            if np.max(np.abs(update)) < 1e-9:
                break
        return float(beta[0]), float(beta[1])
    except Exception:
        return np.nan, np.nan


def metric_record(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "prediction_count": 0,
            "class_0_count": 0,
            "class_1_count": 0,
            "brier_score": np.nan,
            "log_loss": np.nan,
            "balanced_accuracy": np.nan,
            "calibration_intercept": np.nan,
            "calibration_slope": np.nan,
        }
    y = frame["actual_class"].to_numpy(dtype=int)
    p = clip_probability(frame["prediction_probability"])
    predicted = (p >= core.PREDICTION_THRESHOLD).astype(int)
    intercept, slope = calibration_parameters(y, p)
    balanced = (
        float(balanced_accuracy_score(y, predicted))
        if len(np.unique(y)) == 2
        else np.nan
    )
    return {
        "prediction_count": int(len(frame)),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": float(
            np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p)))
        ),
        "balanced_accuracy": balanced,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def build_performance_by_date(
    predictions: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "interval_id",
        "market",
        "index_id",
        "universe_semantics",
        "model_id",
        "prediction_date",
    ]
    rows: list[dict[str, Any]] = []
    count_columns = {
        PIT: "PIT_constituent_count",
        CURRENT: "current_constituent_count",
        INTERSECTION: "intersection_count",
    }
    composition_lookup = composition.set_index(
        ["interval_id", "prediction_date"]
    )
    for values, group in predictions.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(metric_record(group))
        denominator = np.nan
        lookup_key = (str(row["interval_id"]), as_date(row["prediction_date"]))
        if lookup_key in composition_lookup.index:
            comp = composition_lookup.loc[lookup_key]
            if isinstance(comp, pd.DataFrame):
                comp = comp.iloc[0]
            denominator = float(
                comp[count_columns[str(row["universe_semantics"])]]
            )
        row["universe_security_count"] = denominator
        row["prediction_coverage"] = (
            float(row["prediction_count"] / denominator)
            if denominator and np.isfinite(denominator)
            else np.nan
        )
        row["aggregation_rule"] = "equal_security_within_date"
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["prediction_date"] = pd.to_datetime(
            result["prediction_date"], errors="coerce"
        ).dt.date
    return result


def _aggregate_performance(
    lower: pd.DataFrame,
    *,
    group_keys: Sequence[str],
    unit_column: str,
    rule: str,
) -> pd.DataFrame:
    metrics = [
        "brier_score",
        "log_loss",
        "balanced_accuracy",
        "calibration_intercept",
        "calibration_slope",
        "prediction_coverage",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in lower.groupby(list(group_keys), sort=True):
        row = dict(zip(group_keys, values if isinstance(values, tuple) else (values,)))
        row["aggregation_units"] = int(group[unit_column].nunique())
        for metric in metrics:
            numeric = pd.to_numeric(group[metric], errors="coerce")
            row[metric] = float(numeric.mean()) if numeric.notna().any() else np.nan
        row["prediction_count"] = int(group["prediction_count"].sum())
        row["aggregation_rule"] = rule
        rows.append(row)
    return pd.DataFrame(rows)


def build_performance_hierarchy(
    by_date: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval = _aggregate_performance(
        by_date,
        group_keys=[
            "interval_id",
            "market",
            "index_id",
            "universe_semantics",
            "model_id",
        ],
        unit_column="prediction_date",
        rule="equal_date_within_interval",
    ).rename(columns={"aggregation_units": "oos_trading_dates"})
    market = _aggregate_performance(
        interval,
        group_keys=["market", "index_id", "universe_semantics", "model_id"],
        unit_column="interval_id",
        rule="equal_interval_within_market",
    ).rename(columns={"aggregation_units": "eligible_model_valid_intervals"})
    macro = _aggregate_performance(
        market,
        group_keys=["universe_semantics", "model_id"],
        unit_column="market",
        rule="equal_market_selected_market_macro_exploratory",
    ).rename(columns={"aggregation_units": "model_valid_markets"})
    macro["inference_scope"] = "EXPLORATORY_SELECTED_MARKET"
    return interval, market, macro


def benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return adjusted


def moving_block_means(
    values: Sequence[float],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> np.ndarray:
    vector = np.asarray(values, dtype=float)
    vector = vector[np.isfinite(vector)]
    if not len(vector):
        return np.array([], dtype=float)
    block = min(int(block_length), len(vector))
    blocks_needed = int(math.ceil(len(vector) / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(vector), size=(samples, blocks_needed))
    offsets = np.arange(block)
    output = np.empty(samples, dtype=float)
    for idx in range(samples):
        positions = (starts[idx, :, None] + offsets[None, :]) % len(vector)
        output[idx] = float(vector[positions.ravel()[: len(vector)]].mean())
    return output


def bootstrap_summary(
    point: float,
    replicates: np.ndarray,
) -> dict[str, float]:
    if not len(replicates):
        return {
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "raw_p_value": np.nan,
        }
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    less = (float(np.sum(replicates <= 0.0)) + 1.0) / (len(replicates) + 1.0)
    greater = (float(np.sum(replicates >= 0.0)) + 1.0) / (
        len(replicates) + 1.0
    )
    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "raw_p_value": float(min(1.0, 2.0 * min(less, greater))),
    }


def build_current_minus_pit(
    by_date: pd.DataFrame,
    by_interval: pd.DataFrame,
    by_market: pd.DataFrame,
    macro: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "brier_score",
        "log_loss",
        "balanced_accuracy",
        "calibration_intercept",
        "calibration_slope",
        "prediction_coverage",
    ]
    date_keys = [
        "interval_id",
        "market",
        "index_id",
        "model_id",
        "prediction_date",
    ]
    pit = by_date[by_date["universe_semantics"].eq(PIT)].set_index(date_keys)
    current = by_date[by_date["universe_semantics"].eq(CURRENT)].set_index(
        date_keys
    )
    common = pit.index.intersection(current.index)
    date_rows: list[dict[str, Any]] = []
    for key in common:
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(date_keys, key_tuple))
        for metric in metrics:
            row[f"delta_{metric}_current_minus_pit"] = float(
                current.loc[key, metric] - pit.loc[key, metric]
            )
        date_rows.append(row)
    date_diff = pd.DataFrame(date_rows)

    rows: list[dict[str, Any]] = []
    for level, frame, keys in [
        (
            "interval",
            by_interval,
            ["interval_id", "market", "index_id", "model_id"],
        ),
        ("market", by_market, ["market", "index_id", "model_id"]),
        ("selected_market_macro", macro, ["model_id"]),
    ]:
        pit_frame = frame[frame["universe_semantics"].eq(PIT)].set_index(keys)
        current_frame = frame[
            frame["universe_semantics"].eq(CURRENT)
        ].set_index(keys)
        for key in pit_frame.index.intersection(current_frame.index):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = {"aggregation_level": level, **dict(zip(keys, key_tuple))}
            for metric in metrics:
                row[f"delta_{metric}_current_minus_pit"] = float(
                    current_frame.loc[key, metric] - pit_frame.loc[key, metric]
                )
            row["interpretation"] = (
                "negative means lower current-universe loss for loss metrics; "
                "positive Brier difference means lower PIT loss"
            )
            rows.append(row)
    return pd.DataFrame(rows), date_diff


def build_universe_composition(
    inventory: pd.DataFrame,
    arm_members: Mapping[str, pd.DataFrame],
    observations: pd.DataFrame,
    rejections: pd.DataFrame,
) -> pd.DataFrame:
    """Build one explicit composition record for every eligible interval-date."""
    all_observations = pd.concat([observations, rejections], ignore_index=True)
    all_observations["prediction_date"] = pd.to_datetime(
        all_observations["prediction_date"], errors="coerce"
    ).dt.date
    all_observations = all_observations.drop_duplicates(
        [
            "interval_id",
            "universe_semantics",
            "security_id",
            "prediction_date",
        ],
        keep="first",
    )
    observations = observations.copy()
    observations["prediction_date"] = pd.to_datetime(
        observations["prediction_date"], errors="coerce"
    ).dt.date
    inventory_lookup = inventory.set_index("interval_id")
    frames: dict[str, dict[tuple[str, date], set[str]]] = {}
    for arm, frame in arm_members.items():
        working = frame.copy()
        working["membership_date"] = pd.to_datetime(
            working["membership_date"], errors="coerce"
        ).dt.date
        frames[arm] = {
            (str(interval_id), as_date(on_date)): set(
                group["security_id"].astype(str)
            )
            for (interval_id, on_date), group in working.groupby(
                ["interval_id", "membership_date"], sort=True
            )
        }
    keys = sorted(
        set(frames["pit"]).union(frames["current"]),
        key=lambda value: (value[0], value[1]),
    )
    rows: list[dict[str, Any]] = []
    prior_pit: dict[str, set[str]] = {}
    prior_current: dict[str, set[str]] = {}
    for interval_id, on_date in keys:
        pit = frames["pit"].get((interval_id, on_date), set())
        current = frames["current"].get((interval_id, on_date), set())
        intersection = pit.intersection(current)
        union = pit.union(current)
        pit_previous = prior_pit.get(interval_id, pit)
        current_previous = prior_current.get(interval_id, current)
        pit_additions = pit - pit_previous
        pit_removals = pit_previous - pit
        current_additions = current - current_previous
        current_removals = current_previous - current
        interval = inventory_lookup.loc[interval_id]
        row: dict[str, Any] = {
            "interval_id": interval_id,
            "market": interval["market"],
            "index_id": interval["index_id"],
            "prediction_date": on_date,
            "PIT_constituent_count": len(pit),
            "current_constituent_count": len(current),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard_overlap": (
                float(len(intersection) / len(union)) if union else np.nan
            ),
            "PIT_only_count": len(pit - current),
            "current_only_count": len(current - pit),
            "PIT_only_share": (
                float(len(pit - current) / len(union)) if union else np.nan
            ),
            "current_only_share": (
                float(len(current - pit) / len(union)) if union else np.nan
            ),
            "PIT_additions": len(pit_additions),
            "PIT_removals": len(pit_removals),
            "current_additions": len(current_additions),
            "current_removals": len(current_removals),
            "turnover_events_on_date": (
                len(pit_additions)
                + len(pit_removals)
                + len(current_additions)
                + len(current_removals)
            ),
            "interval_turnover_events": int(interval["turnover_events"]),
        }
        for semantics, prefix, denominator in [
            (PIT, "PIT", len(pit)),
            (CURRENT, "current", len(current)),
            (INTERSECTION, "intersection", len(intersection)),
        ]:
            candidates = all_observations[
                all_observations["interval_id"].astype(str).eq(interval_id)
                & all_observations["universe_semantics"].astype(str).eq(
                    semantics
                )
                & all_observations["prediction_date"].eq(on_date)
            ]
            joined = candidates[
                ~candidates["rejection_reason"]
                .astype(str)
                .eq("MISSING_PREDICTION_RAW_OHLC")
            ]
            valid = observations[
                observations["interval_id"].astype(str).eq(interval_id)
                & observations["universe_semantics"].astype(str).eq(semantics)
                & observations["prediction_date"].eq(on_date)
                & observations["valid_for_model"].astype(bool)
            ]
            row[f"{prefix}_price_join_rows"] = int(len(joined))
            row[f"{prefix}_price_join_coverage"] = (
                float(len(joined) / denominator) if denominator else np.nan
            )
            row[f"{prefix}_valid_forecasting_observations"] = int(len(valid))
        row["price_join_coverage"] = row["PIT_price_join_coverage"]
        row["valid_forecasting_observations"] = int(
            row["PIT_valid_forecasting_observations"]
            + row["current_valid_forecasting_observations"]
            + row["intersection_valid_forecasting_observations"]
        )
        rows.append(row)
        prior_pit[interval_id] = pit
        prior_current[interval_id] = current
    return pd.DataFrame(rows)


def _hierarchical_replicates(
    date_diff: pd.DataFrame,
    *,
    metric_column: str,
    level: str,
    scope: Any,
    block_length: int,
    samples: int,
    seed: int,
) -> tuple[float, np.ndarray, int, int, list[str]]:
    data = date_diff.copy()
    if level == "interval":
        data = data[data["interval_id"].astype(str).eq(str(scope))]
    elif level == "market":
        data = data[data["index_id"].astype(str).eq(str(scope))]
    elif level == "selected_market_macro":
        pass
    else:
        raise ValueError(f"unknown inference level {level}")
    data = data.sort_values(
        ["index_id", "interval_id", "prediction_date"], kind="stable"
    )
    interval_vectors: dict[tuple[str, str], np.ndarray] = {}
    for (market, interval_id), group in data.groupby(
        ["index_id", "interval_id"], sort=True
    ):
        vector = pd.to_numeric(group[metric_column], errors="coerce").dropna()
        if not vector.empty:
            interval_vectors[(str(market), str(interval_id))] = (
                vector.to_numpy(dtype=float)
            )
    if not interval_vectors:
        return np.nan, np.array([], dtype=float), 0, 0, []
    point_by_market: dict[str, list[float]] = {}
    for (market, _), vector in interval_vectors.items():
        point_by_market.setdefault(market, []).append(float(vector.mean()))
    point = float(
        np.mean([np.mean(values) for values in point_by_market.values()])
    )
    replicate_by_interval: dict[tuple[str, str], np.ndarray] = {}
    for ordinal, (key, vector) in enumerate(sorted(interval_vectors.items())):
        replicate_by_interval[key] = moving_block_means(
            vector,
            samples=samples,
            block_length=block_length,
            seed=seed + ordinal * 1009,
        )
    replicate_by_market: dict[str, list[np.ndarray]] = {}
    for (market, _), vector in replicate_by_interval.items():
        replicate_by_market.setdefault(market, []).append(vector)
    market_replicates = [
        np.mean(np.vstack(vectors), axis=0)
        for _, vectors in sorted(replicate_by_market.items())
    ]
    replicates = np.mean(np.vstack(market_replicates), axis=0)
    return (
        point,
        replicates,
        sum(len(vector) for vector in interval_vectors.values()),
        len(interval_vectors),
        sorted(point_by_market),
    )


def classify_inference(row: Mapping[str, Any]) -> str:
    if not bool(row.get("valid", False)):
        return "INSUFFICIENT_INFORMATION"
    low = float(row["ci_lower"])
    high = float(row["ci_upper"])
    adjusted = float(row["adjusted_p_value"])
    if adjusted <= 0.05 and (low > 0.0 or high < 0.0):
        return "SUPPORTED_CURRENT_PIT_DIFFERENCE"
    return "NO_SUPPORTED_DIFFERENCE"


def build_bootstrap_inference(
    date_diff: pd.DataFrame,
) -> pd.DataFrame:
    metric_columns = {
        "brier_score": "delta_brier_score_current_minus_pit",
        "log_loss": "delta_log_loss_current_minus_pit",
        "balanced_accuracy": "delta_balanced_accuracy_current_minus_pit",
        "calibration_slope": (
            "delta_calibration_slope_current_minus_pit"
        ),
        "prediction_coverage": (
            "delta_prediction_coverage_current_minus_pit"
        ),
    }
    rows: list[dict[str, Any]] = []
    intervals = (
        date_diff[["interval_id", "market", "index_id"]]
        .drop_duplicates()
        .sort_values(["index_id", "interval_id"])
        .to_dict("records")
    )
    markets = sorted(set(date_diff["index_id"].astype(str)))
    scopes: list[tuple[str, Any, dict[str, Any]]] = []
    for interval in intervals:
        scopes.append(
            (
                "interval",
                interval["interval_id"],
                {
                    "interval_id": interval["interval_id"],
                    "market": interval["market"],
                    "index_id": interval["index_id"],
                },
            )
        )
    for market in markets:
        label = date_diff[date_diff["index_id"].astype(str).eq(market)][
            "market"
        ].iloc[0]
        scopes.append(
            ("market", market, {"market": label, "index_id": market})
        )
    scopes.append(
        (
            "selected_market_macro",
            "SELECTED_MARKET",
            {"scope_label": "SELECTED_MARKET_EXPLORATORY"},
        )
    )
    for model_ordinal, model_id in enumerate(MODEL_IDS):
        model = date_diff[date_diff["model_id"].astype(str).eq(model_id)]
        for level, scope, identifiers in scopes:
            for metric_ordinal, (metric, column) in enumerate(
                metric_columns.items()
            ):
                for block in [
                    core.PRIMARY_BLOCK_LENGTH,
                    *core.ROBUSTNESS_BLOCK_LENGTHS,
                ]:
                    subset = model
                    if level == "interval":
                        subset = subset[
                            subset["interval_id"].astype(str).eq(str(scope))
                        ]
                    elif level == "market":
                        subset = subset[
                            subset["index_id"].astype(str).eq(str(scope))
                        ]
                    interval_lengths = (
                        subset.groupby("interval_id")["prediction_date"].nunique()
                        if not subset.empty
                        else pd.Series(dtype=int)
                    )
                    enough = (
                        not interval_lengths.empty
                        and (
                            block != 40
                            or bool((interval_lengths >= 2 * block).all())
                        )
                    )
                    base = {
                        "inference_level": level,
                        **identifiers,
                        "model_id": model_id,
                        "metric": metric,
                        "block_length": block,
                        "is_primary_block": block
                        == core.PRIMARY_BLOCK_LENGTH,
                        "bootstrap_samples": core.BOOTSTRAP_SAMPLES,
                        "bootstrap_seed": core.BOOTSTRAP_SEED,
                    }
                    if not enough:
                        rows.append(
                            {
                                **base,
                                "valid": False,
                                "point_estimate": np.nan,
                                "ci_lower": np.nan,
                                "ci_upper": np.nan,
                                "raw_p_value": np.nan,
                                "adjusted_p_value": np.nan,
                                "n_dates": int(interval_lengths.sum()),
                                "n_intervals": int(len(interval_lengths)),
                                "n_markets": int(
                                    subset["index_id"].nunique()
                                ),
                                "direction_agreement": np.nan,
                                "reason": (
                                    "40-session robustness requires at least "
                                    "two independent 40-session blocks in "
                                    "every contributing interval"
                                ),
                                "result_classification": (
                                    "INSUFFICIENT_INFORMATION"
                                ),
                            }
                        )
                        continue
                    seed = (
                        core.BOOTSTRAP_SEED
                        + model_ordinal * 100_003
                        + metric_ordinal * 10_007
                        + block
                    )
                    point, replicates, n_dates, n_intervals, valid_markets = (
                        _hierarchical_replicates(
                            model,
                            metric_column=column,
                            level=level,
                            scope=scope,
                            block_length=block,
                            samples=core.BOOTSTRAP_SAMPLES,
                            seed=seed,
                        )
                    )
                    summary = bootstrap_summary(point, replicates)
                    market_points = []
                    if level == "selected_market_macro":
                        for index_id, group in model.groupby(
                            "index_id", sort=True
                        ):
                            market_points.append(
                                float(
                                    group.groupby("interval_id")[column]
                                    .mean()
                                    .mean()
                                )
                            )
                    direction_agreement = (
                        float(
                            max(
                                np.mean(np.asarray(market_points) > 0),
                                np.mean(np.asarray(market_points) < 0),
                            )
                        )
                        if market_points
                        else np.nan
                    )
                    rows.append(
                        {
                            **base,
                            "valid": True,
                            "point_estimate": point,
                            **summary,
                            "adjusted_p_value": np.nan,
                            "n_dates": n_dates,
                            "n_intervals": n_intervals,
                            "n_markets": len(valid_markets),
                            "direction_agreement": direction_agreement,
                            "reason": None,
                            "result_classification": None,
                        }
                    )
    output = pd.DataFrame(rows)
    # The preregistered family is formed separately by metric and block length.
    for _, indexes in output[output["valid"].astype(bool)].groupby(
        ["inference_level", "metric", "block_length"], dropna=False
    ).groups.items():
        output.loc[indexes, "adjusted_p_value"] = benjamini_hochberg(
            output.loc[indexes, "raw_p_value"].to_numpy(dtype=float)
        )
    output["result_classification"] = [
        classify_inference(row) for row in output.to_dict("records")
    ]
    return output


def build_seed_robustness(seed_predictions: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "interval_id",
        "market",
        "index_id",
        "universe_semantics",
        "model_id",
        "seed",
    ]
    rows: list[dict[str, Any]] = []
    for values, group in seed_predictions.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(metric_record(group))
        rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail
    detail["brier_rank_within_seed"] = detail.groupby(
        [
            "interval_id",
            "universe_semantics",
            "seed",
        ]
    )["brier_score"].rank(method="min")
    summary_rows: list[dict[str, Any]] = []
    group_keys = keys[:-1]
    for values, group in detail.groupby(group_keys, sort=True):
        row = dict(zip(group_keys, values))
        row["seed"] = "SUMMARY"
        row["seed_count"] = int(group["seed"].nunique())
        for metric in [
            "brier_score",
            "log_loss",
            "balanced_accuracy",
            "calibration_slope",
            "brier_rank_within_seed",
        ]:
            values_numeric = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values_numeric.mean())
            row[f"{metric}_std"] = float(values_numeric.std(ddof=0))
            row[f"{metric}_min"] = float(values_numeric.min())
            row[f"{metric}_max"] = float(values_numeric.max())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    detail["seed"] = detail["seed"].astype(str)
    detail["record_type"] = "seed"
    summary["record_type"] = "summary"
    return pd.concat([detail, summary], ignore_index=True, sort=False)


def _rank_bootstrap(
    predictions: pd.DataFrame,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Return model rank summaries using a common date resample per draw."""
    date_loss = (
        predictions.assign(
            loss=(
                predictions["prediction_probability"].astype(float)
                - predictions["actual_class"].astype(int)
            )
            ** 2
        )
        .groupby(["model_id", "prediction_date"], as_index=False)["loss"]
        .mean()
    )
    pivot = date_loss.pivot(
        index="prediction_date", columns="model_id", values="loss"
    ).dropna(subset=MODEL_IDS)
    if pivot.empty:
        return {}
    pivot = pivot.sort_index()
    n = len(pivot)
    block = min(block_length, n)
    blocks_needed = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    counts = {model: [] for model in MODEL_IDS}
    for _ in range(samples):
        starts = rng.integers(0, n, size=blocks_needed)
        positions = np.concatenate(
            [(start + np.arange(block)) % n for start in starts]
        )[:n]
        scores = pivot.iloc[positions][MODEL_IDS].mean()
        ranks = scores.rank(method="min")
        for model in MODEL_IDS:
            counts[model].append(float(ranks[model]))
    result: dict[str, dict[str, float]] = {}
    for model, ranks in counts.items():
        array = np.asarray(ranks, dtype=float)
        result[model] = {
            "bootstrap_rank_mean": float(array.mean()),
            "bootstrap_rank_std": float(array.std(ddof=0)),
            "bootstrap_rank_ci_lower": float(np.quantile(array, 0.025)),
            "bootstrap_rank_ci_upper": float(np.quantile(array, 0.975)),
            "bootstrap_top_probability": float(np.mean(array == 1.0)),
        }
    return result


def build_model_rank(
    interval_performance: pd.DataFrame,
    market_performance: pd.DataFrame,
    collapsed: pd.DataFrame,
    seed_robustness: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_level, frame, scope_keys in [
        (
            "interval",
            interval_performance,
            ["interval_id", "market", "index_id"],
        ),
        ("market", market_performance, ["market", "index_id"]),
    ]:
        for scope_values, scoped in frame.groupby(scope_keys, sort=True):
            scope_values = (
                scope_values
                if isinstance(scope_values, tuple)
                else (scope_values,)
            )
            scope = dict(zip(scope_keys, scope_values))
            pit_scores = scoped[scoped["universe_semantics"].eq(PIT)].set_index(
                "model_id"
            )["brier_score"]
            current_scores = scoped[
                scoped["universe_semantics"].eq(CURRENT)
            ].set_index("model_id")["brier_score"]
            common_models = [
                model
                for model in MODEL_IDS
                if model in pit_scores.index and model in current_scores.index
            ]
            if len(common_models) >= 2:
                pit_rank = pit_scores[common_models].rank(method="min")
                current_rank = current_scores[common_models].rank(method="min")
                correlation = float(
                    spearmanr(pit_rank, current_rank).statistic
                )
                pit_winner = str(pit_scores[common_models].idxmin())
                current_winner = str(current_scores[common_models].idxmin())
                top_agreement = pit_winner == current_winner
                current_to_pit_regret = float(
                    pit_scores[current_winner] - pit_scores.min()
                )
                pit_to_current_regret = float(
                    current_scores[pit_winner] - current_scores.min()
                )
            else:
                correlation = np.nan
                pit_winner = None
                current_winner = None
                top_agreement = False
                current_to_pit_regret = np.nan
                pit_to_current_regret = np.nan
            for semantics, arm_group in scoped.groupby(
                "universe_semantics", sort=True
            ):
                arm_group = arm_group.copy()
                arm_group["brier_rank"] = arm_group["brier_score"].rank(
                    method="min"
                )
                for record in arm_group.to_dict("records"):
                    row = {
                        "scope_level": scope_level,
                        **scope,
                        "universe_semantics": semantics,
                        "model_id": record["model_id"],
                        "brier_score": record["brier_score"],
                        "brier_rank": float(record["brier_rank"]),
                        "PIT_current_spearman_rank_correlation": correlation,
                        "PIT_top_model": pit_winner,
                        "current_top_model": current_winner,
                        "top_model_agreement": top_agreement,
                        "current_winner_deployed_on_PIT_regret": (
                            current_to_pit_regret
                        ),
                        "PIT_winner_evaluated_on_current_regret": (
                            pit_to_current_regret
                        ),
                    }
                    rows.append(row)
    output = pd.DataFrame(rows)
    seed_detail = seed_robustness[
        seed_robustness["record_type"].astype(str).eq("seed")
    ]
    seed_summary = seed_detail.groupby(
        ["interval_id", "universe_semantics", "model_id"]
    )["brier_rank_within_seed"].agg(
        seed_rank_mean="mean",
        seed_rank_std="std",
        seed_rank_min="min",
        seed_rank_max="max",
    ).reset_index()
    if not output.empty:
        output = output.merge(
            seed_summary,
            on=["interval_id", "universe_semantics", "model_id"],
            how="left",
        )
    market_seed = (
        seed_detail.groupby(
            [
                "market",
                "index_id",
                "universe_semantics",
                "model_id",
                "seed",
            ],
            as_index=False,
        )["brier_score"]
        .mean()
    )
    market_seed["market_seed_rank"] = market_seed.groupby(
        ["index_id", "universe_semantics", "seed"]
    )["brier_score"].rank(method="min")
    market_seed_summary = market_seed.groupby(
        ["market", "index_id", "universe_semantics", "model_id"]
    )["market_seed_rank"].agg(
        market_seed_rank_mean="mean",
        market_seed_rank_std="std",
        market_seed_rank_min="min",
        market_seed_rank_max="max",
    ).reset_index()
    if not output.empty:
        output = output.merge(
            market_seed_summary,
            on=["market", "index_id", "universe_semantics", "model_id"],
            how="left",
        )
    bootstrap_rows: list[dict[str, Any]] = []
    for (interval_id, semantics), group in collapsed.groupby(
        ["interval_id", "universe_semantics"], sort=True
    ):
        summary = _rank_bootstrap(
            group,
            samples=core.BOOTSTRAP_SAMPLES,
            block_length=core.PRIMARY_BLOCK_LENGTH,
            seed=core.BOOTSTRAP_SEED
            + int(
                hashlib.sha256(
                    f"{interval_id}|{semantics}".encode()
                ).hexdigest()[:8],
                16,
            ),
        )
        for model_id, values in summary.items():
            bootstrap_rows.append(
                {
                    "interval_id": interval_id,
                    "universe_semantics": semantics,
                    "model_id": model_id,
                    **values,
                }
            )
    if bootstrap_rows:
        output = output.merge(
            pd.DataFrame(bootstrap_rows),
            on=["interval_id", "universe_semantics", "model_id"],
            how="left",
        )
    output["rank_uncertainty_interpretation"] = (
        "rank changes are descriptive unless supported by bootstrap uncertainty"
    )
    return output


def _restricted_date_performance(
    predictions: pd.DataFrame,
    valid_keys: set[tuple[str, str, date]],
) -> pd.DataFrame:
    selected = predictions[
        [
            (
                str(row["interval_id"]),
                str(row["security_id"]),
                as_date(row["prediction_date"]),
            )
            in valid_keys
            for row in predictions.to_dict("records")
        ]
    ].copy()
    keys = [
        "interval_id",
        "market",
        "index_id",
        "universe_semantics",
        "model_id",
        "prediction_date",
    ]
    rows = []
    for values, group in selected.groupby(keys, sort=True):
        row = dict(zip(keys, values))
        row.update(metric_record(group))
        row["prediction_coverage"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_intersection_diagnostic(
    collapsed: pd.DataFrame,
    by_interval: pd.DataFrame,
) -> pd.DataFrame:
    intersection = collapsed[
        collapsed["universe_semantics"].eq(INTERSECTION)
    ]
    keys = {
        (
            str(row["interval_id"]),
            str(row["security_id"]),
            as_date(row["prediction_date"]),
        )
        for row in intersection.to_dict("records")
    }
    full = collapsed[collapsed["universe_semantics"].isin([PIT, CURRENT])]
    restricted_date = _restricted_date_performance(full, keys)
    restricted_interval, _, _ = build_performance_hierarchy(restricted_date)
    rows: list[dict[str, Any]] = []
    for interval_id in sorted(set(collapsed["interval_id"].astype(str))):
        for model_id in MODEL_IDS:
            full_group = by_interval[
                by_interval["interval_id"].astype(str).eq(interval_id)
                & by_interval["model_id"].astype(str).eq(model_id)
            ]
            restricted_group = restricted_interval[
                restricted_interval["interval_id"].astype(str).eq(interval_id)
                & restricted_interval["model_id"].astype(str).eq(model_id)
            ]
            arm_c = by_interval[
                by_interval["interval_id"].astype(str).eq(interval_id)
                & by_interval["model_id"].astype(str).eq(model_id)
                & by_interval["universe_semantics"].eq(INTERSECTION)
            ]
            if (
                not {PIT, CURRENT}
                <= set(full_group["universe_semantics"].astype(str))
                or not {PIT, CURRENT}
                <= set(restricted_group["universe_semantics"].astype(str))
            ):
                continue
            full_values = full_group.set_index("universe_semantics")
            restricted_values = restricted_group.set_index(
                "universe_semantics"
            )
            full_delta = float(
                full_values.loc[CURRENT, "brier_score"]
                - full_values.loc[PIT, "brier_score"]
            )
            intersection_delta = float(
                restricted_values.loc[CURRENT, "brier_score"]
                - restricted_values.loc[PIT, "brier_score"]
            )
            rows.append(
                {
                    "interval_id": interval_id,
                    "market": full_group["market"].iloc[0],
                    "index_id": full_group["index_id"].iloc[0],
                    "model_id": model_id,
                    "full_universe_delta_brier_current_minus_pit": full_delta,
                    "common_security_evaluation_delta_brier_current_minus_pit": (
                        intersection_delta
                    ),
                    "diagnostic_composition_component": (
                        full_delta - intersection_delta
                    ),
                    "intersection_trained_arm_brier": (
                        float(arm_c["brier_score"].iloc[0])
                        if not arm_c.empty
                        else np.nan
                    ),
                    "intersection_method": (
                        "native full-arm models restricted to common "
                        "security-date evaluation keys; separately trained "
                        "Arm C reported as composition control"
                    ),
                    "interpretation_boundary": (
                        "descriptive diagnostic; not a causal decomposition"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_leave_one_market_out(
    current_minus_pit: pd.DataFrame,
    included_markets: Sequence[str],
) -> pd.DataFrame:
    market = current_minus_pit[
        current_minus_pit["aggregation_level"].eq("market")
    ].copy()
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = market[market["model_id"].astype(str).eq(model_id)]
        for excluded in included_markets:
            retained = model[~model["index_id"].astype(str).eq(str(excluded))]
            values = pd.to_numeric(
                retained["delta_brier_score_current_minus_pit"],
                errors="coerce",
            ).dropna()
            rows.append(
                {
                    "model_id": model_id,
                    "excluded_market": excluded,
                    "model_valid_markets_remaining": int(
                        retained["index_id"].nunique()
                    ),
                    "equal_market_delta_brier_current_minus_pit": (
                        float(values.mean()) if not values.empty else np.nan
                    ),
                    "direction": (
                        "CURRENT_LOWER_BRIER"
                        if not values.empty and values.mean() < 0
                        else (
                            "PIT_LOWER_BRIER"
                            if not values.empty and values.mean() > 0
                            else "NO_DIRECTION_OR_INFORMATION"
                        )
                    ),
                    "valid": not values.empty,
                    "scope": (
                        "selected-market leave-one-market-out exploratory"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_heterogeneity(
    current_minus_pit: pd.DataFrame,
    inference: pd.DataFrame,
) -> pd.DataFrame:
    market_effects = current_minus_pit[
        current_minus_pit["aggregation_level"].eq("market")
    ]
    market_inference = inference[
        inference["inference_level"].eq("market")
        & inference["metric"].eq("brier_score")
        & inference["block_length"].eq(core.PRIMARY_BLOCK_LENGTH)
        & inference["valid"].astype(bool)
    ]
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        effects = market_effects[
            market_effects["model_id"].astype(str).eq(model_id)
        ][
            [
                "index_id",
                "delta_brier_score_current_minus_pit",
            ]
        ].copy()
        intervals = market_inference[
            market_inference["model_id"].astype(str).eq(model_id)
        ][["index_id", "ci_lower", "ci_upper"]]
        merged = effects.merge(intervals, on="index_id", how="left")
        merged["standard_error"] = (
            pd.to_numeric(merged["ci_upper"], errors="coerce")
            - pd.to_numeric(merged["ci_lower"], errors="coerce")
        ) / (2 * 1.96)
        merged["standard_error"] = merged["standard_error"].where(
            merged["standard_error"] > 1e-12
        )
        valid = merged.dropna(
            subset=[
                "delta_brier_score_current_minus_pit",
                "standard_error",
            ]
        )
        if len(valid) >= 2:
            effects_array = valid[
                "delta_brier_score_current_minus_pit"
            ].to_numpy(dtype=float)
            weights = 1.0 / (
                valid["standard_error"].to_numpy(dtype=float) ** 2
            )
            weighted = float(np.sum(weights * effects_array) / weights.sum())
            q = float(np.sum(weights * (effects_array - weighted) ** 2))
            df = len(valid) - 1
            p_value = float(chi2.sf(q, df))
            i2 = float(max(0.0, (q - df) / q) * 100.0) if q > 0 else 0.0
            directions = np.sign(effects_array)
            disagree = bool(
                np.any(directions > 0) and np.any(directions < 0)
            )
            classification = (
                "HETEROGENEOUS_ACROSS_MARKETS"
                if disagree and i2 >= 50.0
                else "NO_MATERIAL_HETEROGENEITY_SUPPORTED"
            )
        else:
            weighted = q = p_value = i2 = np.nan
            df = max(0, len(valid) - 1)
            disagree = False
            classification = "INSUFFICIENT_INFORMATION"
        rows.append(
            {
                "model_id": model_id,
                "model_valid_markets": int(len(valid)),
                "inverse_variance_weighted_effect": weighted,
                "cochran_Q": q,
                "Q_df": df,
                "Q_p_value": p_value,
                "I2_percent": i2,
                "market_direction_disagreement": disagree,
                "heterogeneity_classification": classification,
                "classification_rule": (
                    "HETEROGENEOUS_ACROSS_MARKETS only when market signs "
                    "disagree and I2 >= 50%; exploratory with few markets"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_composition_associations(
    date_diff: pd.DataFrame,
    composition: pd.DataFrame,
) -> pd.DataFrame:
    merged = date_diff.merge(
        composition,
        on=["interval_id", "market", "index_id", "prediction_date"],
        how="left",
        validate="many_to_one",
    )
    variables = [
        "jaccard_overlap",
        "interval_turnover_events",
        "PIT_only_share",
        "current_only_share",
    ]
    rows: list[dict[str, Any]] = []
    for model_id, group in merged.groupby("model_id", sort=True):
        for variable in variables:
            pair = group[
                [
                    variable,
                    "delta_brier_score_current_minus_pit",
                ]
            ].dropna()
            correlation = (
                float(
                    spearmanr(
                        pair[variable],
                        pair["delta_brier_score_current_minus_pit"].abs(),
                    ).statistic
                )
                if len(pair) >= 3 and pair[variable].nunique() >= 2
                else np.nan
            )
            rows.append(
                {
                    "model_id": model_id,
                    "composition_variable": variable,
                    "n_dates": len(pair),
                    "spearman_with_absolute_brier_difference": correlation,
                    "analysis_scope": "descriptive_exploratory_noncausal",
                }
            )
    return pd.DataFrame(rows)


def _security_equal_difference(collapsed: pd.DataFrame) -> pd.DataFrame:
    working = collapsed.copy()
    working["brier_loss"] = (
        working["prediction_probability"].astype(float)
        - working["actual_class"].astype(int)
    ) ** 2
    security = (
        working.groupby(
            [
                "interval_id",
                "market",
                "index_id",
                "universe_semantics",
                "model_id",
                "security_id",
            ],
            as_index=False,
        )["brier_loss"]
        .mean()
    )
    interval = (
        security.groupby(
            [
                "interval_id",
                "market",
                "index_id",
                "universe_semantics",
                "model_id",
            ],
            as_index=False,
        )["brier_loss"]
        .mean()
    )
    pit = interval[interval["universe_semantics"].eq(PIT)].set_index(
        ["interval_id", "market", "index_id", "model_id"]
    )
    current = interval[interval["universe_semantics"].eq(CURRENT)].set_index(
        ["interval_id", "market", "index_id", "model_id"]
    )
    rows = []
    for key in pit.index.intersection(current.index):
        rows.append(
            {
                "interval_id": key[0],
                "market": key[1],
                "index_id": key[2],
                "model_id": key[3],
                "delta_brier_current_minus_pit": float(
                    current.loc[key, "brier_loss"]
                    - pit.loc[key, "brier_loss"]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_sensitivity(
    inventory: pd.DataFrame,
    by_interval: pd.DataFrame,
    bootstrap: pd.DataFrame,
    collapsed: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid_intervals = set(collapsed["interval_id"].astype(str))
    coverage = inventory.set_index("interval_id")["price_coverage"]
    oos_dates = (
        collapsed.groupby(["interval_id", "model_id"])[
            "prediction_date"
        ].nunique()
    )
    primary = bootstrap[
        bootstrap["inference_level"].eq("interval")
        & bootstrap["metric"].eq("brier_score")
        & bootstrap["block_length"].eq(core.PRIMARY_BLOCK_LENGTH)
        & bootstrap["valid"].astype(bool)
    ]
    for record in primary.to_dict("records"):
        common = {
            "interval_id": record["interval_id"],
            "market": record["market"],
            "index_id": record["index_id"],
            "model_id": record["model_id"],
        }
        rows.append(
            {
                **common,
                "sensitivity": "complete_price_join_only",
                "included": True,
                "estimate": record["point_estimate"],
                "notes": (
                    "all prediction rows require complete raw OHLC by "
                    "construction"
                ),
            }
        )
        price_ok = float(coverage.loc[record["interval_id"]]) >= 0.98
        rows.append(
            {
                **common,
                "sensitivity": "minimum_0.98_interval_price_coverage",
                "included": price_ok,
                "estimate": record["point_estimate"] if price_ok else np.nan,
                "notes": "predefined V5 interval price-coverage filter",
            }
        )
        enough_dates = (
            int(
                oos_dates.loc[
                    (record["interval_id"], record["model_id"])
                ]
            )
            >= 60
        )
        rows.append(
            {
                **common,
                "sensitivity": "exclude_intervals_under_60_OOS_dates",
                "included": enough_dates,
                "estimate": (
                    record["point_estimate"] if enough_dates else np.nan
                ),
                "notes": "predefined sensitivity; primary gate remains 40",
            }
        )
    block10 = bootstrap[
        bootstrap["inference_level"].eq("interval")
        & bootstrap["metric"].eq("brier_score")
        & bootstrap["block_length"].eq(10)
        & bootstrap["valid"].astype(bool)
    ]
    for record in block10.to_dict("records"):
        rows.append(
            {
                "interval_id": record["interval_id"],
                "market": record["market"],
                "index_id": record["index_id"],
                "model_id": record["model_id"],
                "sensitivity": "10_session_bootstrap_blocks",
                "included": True,
                "estimate": record["point_estimate"],
                "ci_lower": record["ci_lower"],
                "ci_upper": record["ci_upper"],
                "notes": "predefined moving-block robustness",
            }
        )
    intersection = by_interval[
        by_interval["universe_semantics"].eq(INTERSECTION)
    ]
    for record in intersection.to_dict("records"):
        rows.append(
            {
                "interval_id": record["interval_id"],
                "market": record["market"],
                "index_id": record["index_id"],
                "model_id": record["model_id"],
                "sensitivity": "intersection_arm",
                "included": True,
                "estimate": record["brier_score"],
                "notes": "composition-control Arm C performance",
            }
        )
    for record in _security_equal_difference(collapsed).to_dict("records"):
        rows.append(
            {
                **record,
                "sensitivity": "equal_security_aggregation",
                "included": True,
                "estimate": record["delta_brier_current_minus_pit"],
                "notes": "secondary robustness; not primary hierarchy",
            }
        )
    for record in leave_one_out.to_dict("records"):
        rows.append(
            {
                "model_id": record["model_id"],
                "sensitivity": (
                    f"exclude_market_{record['excluded_market']}"
                ),
                "included": bool(record["valid"]),
                "estimate": record[
                    "equal_market_delta_brier_current_minus_pit"
                ],
                "notes": "predefined leave-one-market-out sensitivity",
            }
        )
    return pd.DataFrame(rows)


def build_set50_v4_v6_reconciliation(
    interval_performance: pd.DataFrame,
) -> pd.DataFrame:
    v4_path = REPORTS / "set50_forecast_performance_by_arm_model_v4.csv"
    v4 = pd.read_csv(v4_path)
    v4 = v4[
        v4["period"].astype(str).eq("final_test")
        & v4["seed"].astype(str).eq("COLLAPSED")
        & v4["universe_semantics"].isin([PIT, CURRENT])
    ].copy()
    v6 = interval_performance[
        interval_performance["index_id"].astype(str).eq("SET50")
        & interval_performance["universe_semantics"].isin([PIT, CURRENT])
    ].copy()
    rows: list[dict[str, Any]] = []
    for v6_row in v6.to_dict("records"):
        matching = v4[
            v4["model_id"].astype(str).eq(str(v6_row["model_id"]))
            & v4["universe_semantics"].astype(str).eq(
                str(v6_row["universe_semantics"])
            )
        ]
        v4_brier = (
            float(matching["brier_score"].iloc[0])
            if not matching.empty
            else np.nan
        )
        rows.append(
            {
                "v4_experiment_id": "SET50_PIT_FORECASTING_BENCHMARK_V4",
                "v4_locked_result": "SET50_SHORT_INTERVAL_NULL_DIAGNOSTIC",
                "v4_protocol": (
                    "locked SET50 V4 calibration/final-test protocol"
                ),
                "v4_brier_score": v4_brier,
                "v6_experiment_id": core.EXPERIMENT_ID,
                "v6_interval_id": v6_row["interval_id"],
                "v6_universe_semantics": v6_row[
                    "universe_semantics"
                ],
                "model_id": v6_row["model_id"],
                "v6_protocol": (
                    "60 train, 20 calibration, 20 test, 20 step; "
                    "interval bounded"
                ),
                "v6_brier_score": v6_row["brier_score"],
                "v6_minus_v4_brier": (
                    float(v6_row["brier_score"] - v4_brier)
                    if np.isfinite(v4_brier)
                    else np.nan
                ),
                "interpretation": (
                    "protocol reconciliation only; V6 does not replace, "
                    "merge, or relabel locked V4 results"
                ),
            }
        )
    return pd.DataFrame(rows)


def add_rank_difference(
    contrasts: pd.DataFrame,
    interval: pd.DataFrame,
    market: pd.DataFrame,
    macro: pd.DataFrame,
) -> pd.DataFrame:
    output = contrasts.copy()
    for level, frame, keys in [
        (
            "interval",
            interval,
            ["interval_id", "market", "index_id", "model_id"],
        ),
        ("market", market, ["market", "index_id", "model_id"]),
        ("selected_market_macro", macro, ["model_id"]),
    ]:
        ranked = frame.copy()
        ranked["model_rank"] = ranked.groupby(
            [
                column
                for column in keys
                if column != "model_id"
            ]
            + ["universe_semantics"]
        )["brier_score"].rank(method="min")
        pit = ranked[ranked["universe_semantics"].eq(PIT)].set_index(keys)
        current = ranked[ranked["universe_semantics"].eq(CURRENT)].set_index(
            keys
        )
        for key in pit.index.intersection(current.index):
            key_tuple = key if isinstance(key, tuple) else (key,)
            mask = output["aggregation_level"].eq(level)
            for column, value in zip(keys, key_tuple):
                mask &= output[column].astype(str).eq(str(value))
            output.loc[
                mask,
                "delta_model_rank_current_minus_pit",
            ] = float(
                current.loc[key, "model_rank"] - pit.loc[key, "model_rank"]
            )
    return output


def verify_locked_v4() -> dict[str, Any]:
    lock_root = PIT_ROOT / "locked_results" / "set50_pilot_v4"
    lock_manifest_path = lock_root / "locked_artifacts_manifest.json"
    manifest = json.loads(lock_manifest_path.read_text(encoding="utf-8"))
    failures = []
    for artifact in manifest["artifacts"]:
        original = REPO_ROOT / artifact["original_path"]
        locked = lock_root / artifact["locked_path"]
        for label, path in [("original", original), ("locked", locked)]:
            if not path.is_file():
                failures.append(f"{label} missing: {path}")
            elif sha256_file(path) != artifact["locked_sha256"]:
                failures.append(f"{label} checksum mismatch: {path}")
    return {
        "lock_id": manifest["lock_id"],
        "locked_result": manifest["status"],
        "artifact_count": int(manifest["artifact_count"]),
        "all_checksums_valid": not failures,
        "failures": failures,
    }


def required_artifact_paths() -> list[Path]:
    paths: list[Path] = [
        core.CONFIG_PATH,
        core.INVENTORY_PATH,
        core.TERMINOLOGY_PATH,
        REPO_ROOT
        / "research_data"
        / "shared"
        / "manifests"
        / "manifest_portability_v6.json",
        REPORT_ROOT / "MANIFEST_PORTABILITY_V6.md",
        PROCESSED / "pit_multimarket_observation_panel_v6.parquet",
        PROCESSED / "pit_multimarket_feature_panel_v6.parquet",
        PROCESSED / "pit_multimarket_fold_calendar_v6.parquet",
        PROCESSED / "pit_multimarket_predictions_seed_level_v6.parquet",
        PROCESSED / "pit_multimarket_predictions_seed_collapsed_v6.parquet",
        QA / "pit_multimarket_observation_rejections_v6.parquet",
        QA / "pit_multimarket_all_cells_v6.parquet",
        QA / "pit_v6_preflight_validation.json",
        QA / "pit_v6_test_summary.json",
        REPO_ROOT / "scripts" / "manifest_portability_v6.py",
        REPO_ROOT
        / "scripts"
        / "point_in_time"
        / "pit_multimarket_forecasting_v6.py",
        REPO_ROOT
        / "scripts"
        / "point_in_time"
        / "pit_multimarket_analytics_v6.py",
        REPO_ROOT / "tests" / "test_manifest_portability_v6.py",
        REPO_ROOT
        / "tests"
        / "point_in_time"
        / "test_pit_multimarket_forecasting_v6.py",
        REPORTS / "pit_multimarket_universe_composition_v6.parquet",
        REPORTS / "pit_multimarket_overlap_by_date_v6.csv",
        REPORTS / "pit_multimarket_composition_associations_v6.csv",
        REPORTS / "pit_multimarket_sensitivity_v6.csv",
        REPORTS / "pit_multimarket_sensitivity_v6.parquet",
        REPORTS / "set50_v4_v6_protocol_reconciliation.csv",
        REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V6.md",
        REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V6.json",
    ]
    for base in PERFORMANCE_BASES:
        paths.extend([REPORTS / f"{base}.csv", REPORTS / f"{base}.parquet"])
    return paths


def build_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(set(paths), key=lambda item: item.as_posix()):
        if not path.is_file():
            raise FileNotFoundError(f"required V6 artifact missing: {path}")
        relative = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        records.append(
            {
                "artifact_relative_path": relative,
                "repository_root_hint": ".",
                "original_absolute_path": str(path.resolve()),
                "checksum": sha256_file(path),
                "size_bytes": int(path.stat().st_size),
                "manifest_version": core.MANIFEST_VERSION,
            }
        )
    payload = {
        "experiment_id": core.EXPERIMENT_ID,
        "manifest_version": core.MANIFEST_VERSION,
        "validation_key": "artifact_relative_path",
        "repository_root_hint": ".",
        "artifact_count": len(records),
        "records": records,
    }
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_ROOT / "pit_multimarket_forecasting_v6_manifest.json"
    write_json(path, payload)
    for record in records:
        validate_portable_record(record, REPO_ROOT)
    return payload


def validate_artifacts(
    manifest: Mapping[str, Any],
    *,
    inventory: pd.DataFrame,
    folds: pd.DataFrame,
    collapsed: pd.DataFrame,
    all_cells: pd.DataFrame,
) -> dict[str, Any]:
    failures: list[str] = []
    for record in manifest["records"]:
        try:
            validate_portable_record(record, REPO_ROOT)
        except Exception as exc:
            failures.append(f"manifest validation: {exc}")
    for base in PERFORMANCE_BASES:
        for suffix in ("csv", "parquet"):
            if not (REPORTS / f"{base}.{suffix}").is_file():
                failures.append(f"missing required output {base}.{suffix}")
    ready = inventory[inventory["eligible_for_forecasting"].astype(bool)]
    if len(ready) != 6:
        failures.append(f"eligible inventory expected 6, found {len(ready)}")
    if set(ready["index_id"].astype(str)) != {"SET50", "VN30", "IDX30"}:
        failures.append("eligible market scope differs from SET50/VN30/IDX30")
    if collapsed.duplicated(
        [
            "interval_id",
            "universe_semantics",
            "fold_id",
            "security_id",
            "prediction_date",
            "model_id",
        ]
    ).any():
        failures.append("duplicate collapsed prediction keys")
    probability = collapsed["prediction_probability"].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or (probability < 0).any()
        or (probability > 1).any()
    ):
        failures.append("prediction probability validation failed")
    if not (
        pd.to_datetime(collapsed["target_date"])
        > pd.to_datetime(collapsed["prediction_date"])
    ).all():
        failures.append("target chronology validation failed")
    if not (
        pd.to_datetime(collapsed["label_maturity_cutoff"])
        < pd.to_datetime(collapsed["prediction_date"])
    ).all():
        failures.append("label maturity validation failed")
    valid_folds = folds[folds["valid_fold"].astype(bool)]
    for _, group in valid_folds.groupby("interval_id"):
        ordered = group.sort_values("test_start")
        starts = pd.to_datetime(ordered["test_start"])
        ends = pd.to_datetime(ordered["test_end"])
        if len(group) > 1 and not (starts.iloc[1:].to_numpy() > ends.iloc[:-1].to_numpy()).all():
            failures.append("overlapping interval test blocks")
    valid_cells = int(all_cells["valid"].astype(bool).sum())
    invalid_cells = int((~all_cells["valid"].astype(bool)).sum())
    if valid_cells != 36 or invalid_cells != 36:
        failures.append(
            f"unexpected V6 cell counts valid={valid_cells} invalid={invalid_cells}"
        )
    v4 = verify_locked_v4()
    if not v4["all_checksums_valid"]:
        failures.extend(v4["failures"])
    payload = {
        "experiment_id": core.EXPERIMENT_ID,
        "status": "PASS" if not failures else "FAIL",
        "checks": {
            "manifest_records": len(manifest["records"]),
            "eligible_intervals": len(ready),
            "valid_cells": valid_cells,
            "invalid_cells": invalid_cells,
            "collapsed_predictions": len(collapsed),
            "probabilities_valid": "prediction probability validation failed"
            not in failures,
            "chronology_valid": not any(
                "chronology" in failure for failure in failures
            ),
            "locked_v4_immutable": v4["all_checksums_valid"],
        },
        "failures": failures,
    }
    write_json(QA / "pit_multimarket_artifact_validation_v6.json", payload)
    return payload


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    selected = frame.reindex(columns=columns).copy()
    if selected.empty:
        return "_No model-valid rows._"
    headers = [str(column) for column in selected.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in selected.to_dict("records"):
        values = []
        for column in headers:
            value = row[column]
            if value is None or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                text = "NA"
            elif isinstance(value, float):
                text = f"{value:.6f}"
            else:
                text = str(value)
            values.append(text.replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_test_summary() -> dict[str, Any]:
    path = QA / "pit_v6_test_summary.json"
    if not path.is_file():
        return {
            "status": "PENDING",
            "passed": 0,
            "failed": 0,
            "repository_tests_passed": 0,
            "repository_tests_failed": 0,
            "commands": [],
        }
    return json.loads(path.read_text(encoding="utf-8"))


def build_status_payload(
    *,
    inventory: pd.DataFrame,
    observations: pd.DataFrame,
    features: pd.DataFrame,
    rejections: pd.DataFrame,
    folds: pd.DataFrame,
    seed_predictions: pd.DataFrame,
    collapsed: pd.DataFrame,
    all_cells: pd.DataFrame,
    by_interval: pd.DataFrame,
    by_market: pd.DataFrame,
    macro: pd.DataFrame,
    inference: pd.DataFrame,
    rank: pd.DataFrame,
    diagnostic: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    heterogeneity: pd.DataFrame,
    composition: pd.DataFrame,
    v4_integrity: Mapping[str, Any],
    artifact_validation_status: str = "PENDING",
) -> dict[str, Any]:
    ready = inventory[inventory["eligible_for_forecasting"].astype(bool)]
    primary = inference[
        inference["metric"].eq("brier_score")
        & inference["block_length"].eq(core.PRIMARY_BLOCK_LENGTH)
        & inference["valid"].astype(bool)
    ]
    market_primary = primary[primary["inference_level"].eq("market")]
    macro_primary = primary[
        primary["inference_level"].eq("selected_market_macro")
    ]
    rank_market = rank[
        rank["scope_level"].eq("market")
        & rank["universe_semantics"].eq(PIT)
    ]
    rank_interval = rank[rank["scope_level"].eq("interval")]
    test_summary = load_test_summary()
    required_present = all(path.is_file() for path in required_artifact_paths())
    complete = (
        required_present
        and int(all_cells["valid"].astype(bool).sum()) > 0
        and len(seed_predictions) > 0
        and len(collapsed) > 0
        and v4_integrity["all_checksums_valid"]
        and artifact_validation_status in {"PASS", "PENDING"}
        and test_summary.get("status") == "PASS"
    )
    status = "COMPLETE" if complete else "PARTIAL"
    scope_rows = []
    for record in ready.sort_values(
        ["index_id", "interval_start"]
    ).to_dict("records"):
        scope_rows.append(
            {
                key: record[key]
                for key in [
                    "interval_id",
                    "market",
                    "index_id",
                    "interval_start",
                    "interval_end_exclusive",
                    "trading_dates",
                    "price_coverage",
                    "turnover_events",
                    "mean_jaccard_overlap",
                ]
            }
        )
    payload = {
        "status": status,
        "experiment_id": core.EXPERIMENT_ID,
        "study_scope": (
            "selected-market partial-period ASEAN benchmark; not five-market"
        ),
        "portability": {
            "legacy_manifest_failure": (
                "legacy model-decay manifests stored WSL /mnt/k absolute paths"
            ),
            "resolution_method": (
                "repository-relative validation key, configured root, legacy "
                "absolute metadata, then explicit Windows/WSL translation"
            ),
            "manifest_path_issue_fixed": True,
            "repository_tests_passed": int(
                test_summary.get("repository_tests_passed", 0)
            ),
            "repository_tests_failed": int(
                test_summary.get("repository_tests_failed", 0)
            ),
        },
        "eligible_scope": {
            "included_markets": sorted(set(ready["index_id"].astype(str))),
            "excluded_markets": ["PSEI", "FBM_KLCI"],
            "exclusion_reasons": {
                "PSEI": "no V5 interval passes eligible_for_forecasting",
                "FBM_KLCI": "no V5 interval passes eligible_for_forecasting",
            },
            "eligible_intervals_by_market": {
                key: int(value)
                for key, value in ready.groupby("index_id").size().items()
            },
            "total_eligible_trading_dates": int(ready["trading_dates"].sum()),
            "intervals": scope_rows,
        },
        "forecasting_data": {
            "observation_rows": len(observations),
            "feature_rows": len(features),
            "rejection_rows": len(rejections),
            "fold_rows": len(folds),
            "valid_folds": int(folds["valid_fold"].astype(bool).sum()),
            "seed_level_predictions": len(seed_predictions),
            "seed_collapsed_predictions": len(collapsed),
            "valid_cells": int(all_cells["valid"].astype(bool).sum()),
            "invalid_cells": int((~all_cells["valid"].astype(bool)).sum()),
        },
        "performance": {
            "by_interval": by_interval.to_dict("records"),
            "by_market": by_market.to_dict("records"),
            "equal_market_macro": macro.to_dict("records"),
        },
        "current_versus_pit": {
            "primary_market_inference": market_primary.to_dict("records"),
            "primary_equal_market_macro_inference": (
                macro_primary.to_dict("records")
            ),
            "leave_one_market_out": leave_one_out.to_dict("records"),
            "heterogeneity": heterogeneity.to_dict("records"),
            "interpretation": (
                "negative Brier delta means current-universe loss is lower; "
                "positive means PIT loss is lower; CI crossing zero is no "
                "supported difference, not equivalence"
            ),
        },
        "model_selection": {
            "market_rank_summary": rank_market.to_dict("records"),
            "interval_rank_uncertainty": rank_interval.to_dict("records"),
            "uncertainty_boundary": (
                "rank changes are not labelled meaningful when unsupported "
                "by seed or bootstrap uncertainty"
            ),
        },
        "intersection_control": diagnostic.to_dict("records"),
        "composition": {
            "date_rows": len(composition),
            "interpretation_boundary": (
                "descriptive and exploratory; no causal decomposition"
            ),
        },
        "locked_v4": dict(v4_integrity),
        "tests": test_summary,
        "artifact_validation_status": artifact_validation_status,
        "overall_five_market_pit_status": "PARTIAL",
        "main_blockers": [
            "PSEi has no V5 forecasting-ready interval",
            "FBM KLCI has no V5 forecasting-ready interval",
            (
                "both IDX30 intervals and one VN30 interval are too short for "
                "two 60/20/20 rolling-origin folds"
            ),
            "historical price joins remain incomplete in some V5 intervals",
        ],
        "limitations": [
            "partial market coverage",
            "heterogeneous date intervals",
            "short intervals",
            "unresolved PSEi and FBM KLCI",
            "missing historical price joins",
            "no profitability test",
            "no causal interpretation",
            "macro inference is exploratory with few model-valid markets",
        ],
        "output_report": (
            "reports/data_execution/PIT_MULTIMARKET_FORECASTING_STATUS_V6.md"
        ),
    }
    return json_safe(payload)


def write_status(payload: Mapping[str, Any]) -> None:
    json_path = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V6.json"
    md_path = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V6.md"
    write_json(json_path, payload)
    scope = pd.DataFrame(payload["eligible_scope"]["intervals"])
    market_inference = pd.DataFrame(
        payload["current_versus_pit"]["primary_market_inference"]
    )
    macro_inference = pd.DataFrame(
        payload["current_versus_pit"][
            "primary_equal_market_macro_inference"
        ]
    )
    interval_performance = pd.DataFrame(
        payload["performance"]["by_interval"]
    )
    market_performance = pd.DataFrame(payload["performance"]["by_market"])
    macro_performance = pd.DataFrame(
        payload["performance"]["equal_market_macro"]
    )
    rank = pd.DataFrame(payload["model_selection"]["market_rank_summary"])
    interval_rank = pd.DataFrame(
        payload["model_selection"]["interval_rank_uncertainty"]
    )
    diagnostic = pd.DataFrame(payload["intersection_control"])
    leave_one_out = pd.DataFrame(
        payload["current_versus_pit"]["leave_one_market_out"]
    )
    heterogeneity = pd.DataFrame(
        payload["current_versus_pit"]["heterogeneity"]
    )
    tests = payload["tests"]
    lines = [
        "# PIT Multimarket Forecasting Status V6",
        "",
        f"**STATUS: {payload['status']}**",
        "",
        (
            "This is a selected-market, partial-period ASEAN PIT benchmark. "
            "It is not a complete five-market benchmark. No profitability or "
            "causal claim is made."
        ),
        "",
        "## Portability",
        "",
        (
            "The legacy environmental failure came from WSL `/mnt/k/...` "
            "absolute paths. Validation now keys on repository-relative paths "
            "under the configured root, retaining absolute paths only as "
            "legacy metadata and translating Windows/WSL drive forms only "
            "when explicitly recognized."
        ),
        "",
        f"- Manifest path issue fixed: yes.",
        (
            f"- Repository tests: {payload['portability']['repository_tests_passed']} "
            f"passed, {payload['portability']['repository_tests_failed']} failed."
        ),
        (
            f"- Locked V4 artifacts: "
            f"{'valid' if payload['locked_v4']['all_checksums_valid'] else 'INVALID'} "
            f"({payload['locked_v4']['artifact_count']} checksums)."
        ),
        "",
        "## Eligible scope",
        "",
        markdown_table(
            scope,
            [
                "index_id",
                "interval_id",
                "interval_start",
                "interval_end_exclusive",
                "trading_dates",
                "price_coverage",
                "turnover_events",
                "mean_jaccard_overlap",
            ],
        ),
        "",
        (
            "Included: SET50, VN30, IDX30. Excluded: PSEi and FBM KLCI "
            "because neither has a V5 forecasting-ready interval."
        ),
        "",
        "## Forecasting data",
        "",
        (
            f"- Observations: {payload['forecasting_data']['observation_rows']:,}; "
            f"features: {payload['forecasting_data']['feature_rows']:,}; "
            f"explicit rejections: {payload['forecasting_data']['rejection_rows']:,}."
        ),
        (
            f"- Valid folds: {payload['forecasting_data']['valid_folds']}; "
            f"seed-level predictions: "
            f"{payload['forecasting_data']['seed_level_predictions']:,}; "
            f"collapsed predictions: "
            f"{payload['forecasting_data']['seed_collapsed_predictions']:,}."
        ),
        (
            f"- Valid cells: {payload['forecasting_data']['valid_cells']}; "
            f"invalid cells retained: "
            f"{payload['forecasting_data']['invalid_cells']}."
        ),
        "",
        "## Performance and current versus PIT",
        "",
        (
            "Losses are aggregated equally across securities within date, "
            "then equally across dates within interval, intervals within "
            "market, and model-valid markets at the exploratory macro level."
        ),
        "",
        "Interval-level performance:",
        "",
        markdown_table(
            interval_performance,
            [
                "index_id",
                "interval_id",
                "universe_semantics",
                "model_id",
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "calibration_intercept",
                "calibration_slope",
                "prediction_coverage",
            ],
        ),
        "",
        "Market-level performance:",
        "",
        markdown_table(
            market_performance,
            [
                "index_id",
                "universe_semantics",
                "model_id",
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "calibration_intercept",
                "calibration_slope",
                "prediction_coverage",
            ],
        ),
        "",
        "Exploratory equal-market performance:",
        "",
        markdown_table(
            macro_performance,
            [
                "universe_semantics",
                "model_id",
                "model_valid_markets",
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "calibration_intercept",
                "calibration_slope",
                "prediction_coverage",
            ],
        ),
        "",
        "Primary market-level Brier inference (current minus PIT):",
        "",
        markdown_table(
            market_inference,
            [
                "index_id",
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "adjusted_p_value",
                "result_classification",
            ],
        ),
        "",
        "Equal-market macro inference is exploratory:",
        "",
        markdown_table(
            macro_inference,
            [
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "adjusted_p_value",
                "direction_agreement",
                "result_classification",
            ],
        ),
        "",
        (
            "Negative Brier differences mean lower loss under the deliberately "
            "backward-applied current universe. Positive values mean lower PIT "
            "loss. A confidence interval crossing zero is classified as no "
            "supported difference, not equivalence."
        ),
        "",
        "Leave-one-market-out sensitivity:",
        "",
        markdown_table(
            leave_one_out,
            [
                "model_id",
                "excluded_market",
                "model_valid_markets_remaining",
                "equal_market_delta_brier_current_minus_pit",
                "direction",
                "valid",
            ],
        ),
        "",
        "Exploratory market heterogeneity:",
        "",
        markdown_table(
            heterogeneity,
            [
                "model_id",
                "model_valid_markets",
                "cochran_Q",
                "Q_p_value",
                "I2_percent",
                "market_direction_disagreement",
                "heterogeneity_classification",
            ],
        ),
        "",
        "## Model selection",
        "",
        markdown_table(
            rank,
            [
                "index_id",
                "model_id",
                "PIT_current_spearman_rank_correlation",
                "top_model_agreement",
                "current_winner_deployed_on_PIT_regret",
                "PIT_winner_evaluated_on_current_regret",
            ],
        ),
        "",
        "Interval-level seed and bootstrap rank uncertainty:",
        "",
        markdown_table(
            interval_rank,
            [
                "index_id",
                "interval_id",
                "universe_semantics",
                "model_id",
                "brier_rank",
                "seed_rank_std",
                "bootstrap_rank_std",
                "bootstrap_top_probability",
            ],
        ),
        "",
        (
            "Seed and moving-block bootstrap rank distributions are retained "
            "in the V6 rank and seed-robustness artifacts. Rank changes are "
            "not treated as meaningful when uncertainty does not support them."
        ),
        "",
        "## Intersection control",
        "",
        markdown_table(
            diagnostic,
            [
                "index_id",
                "interval_id",
                "model_id",
                "full_universe_delta_brier_current_minus_pit",
                "common_security_evaluation_delta_brier_current_minus_pit",
                "diagnostic_composition_component",
            ],
        ),
        "",
        (
            "The common-security result restricts native full-arm models to "
            "common security-date evaluation keys; the separately trained Arm "
            "C is also reported. This is descriptive, not a causal "
            "decomposition."
        ),
        "",
        "## Limitations",
        "",
        *[f"- {item}." for item in payload["limitations"]],
        "",
        "## Tests and artifact validation",
        "",
        (
            f"- Tests: {tests.get('passed', 0)} passed, "
            f"{tests.get('failed', 0)} failed; status "
            f"`{tests.get('status', 'PENDING')}`."
        ),
        (
            f"- Artifact validation: "
            f"`{payload['artifact_validation_status']}`."
        ),
        "",
        "## Overall five-market PIT status",
        "",
        "**PARTIAL.** PSEi and FBM KLCI remain blocked from forecasting.",
        "",
    ]
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines), encoding="utf-8")


def run_analytics() -> dict[str, Any]:
    if not core.CONFIG_PATH.is_file():
        raise RuntimeError("V6 frozen configuration is missing")
    config = json_safe(
        core.yaml.safe_load(core.CONFIG_PATH.read_text(encoding="utf-8"))
    )
    expected = json_safe(core.expected_config(core._ready_inventory()))
    if config != expected:
        raise RuntimeError(
            "frozen V6 configuration drift detected; V6.1 is required"
        )
    inventory_all = pd.read_csv(core.INVENTORY_PATH)
    inventory_all["eligible_for_forecasting"] = (
        inventory_all["eligible_for_forecasting"]
        .astype(str)
        .str.lower()
        .eq("true")
    )
    inventory = inventory_all[
        inventory_all["eligible_for_forecasting"]
    ].copy()
    observations = pd.read_parquet(
        PROCESSED / "pit_multimarket_observation_panel_v6.parquet"
    )
    features = pd.read_parquet(
        PROCESSED / "pit_multimarket_feature_panel_v6.parquet"
    )
    rejections = pd.read_parquet(
        QA / "pit_multimarket_observation_rejections_v6.parquet"
    )
    folds = pd.read_parquet(
        PROCESSED / "pit_multimarket_fold_calendar_v6.parquet"
    )
    seed_predictions = pd.read_parquet(
        PROCESSED / "pit_multimarket_predictions_seed_level_v6.parquet"
    )
    collapsed = pd.read_parquet(
        PROCESSED / "pit_multimarket_predictions_seed_collapsed_v6.parquet"
    )
    all_cells = pd.read_parquet(
        QA / "pit_multimarket_all_cells_v6.parquet"
    )
    invalid = all_cells[~all_cells["valid"].astype(bool)].copy()

    arm_members = core.build_arm_membership_dates(inventory)
    composition = build_universe_composition(
        inventory, arm_members, observations, rejections
    )
    composition.to_parquet(
        REPORTS / "pit_multimarket_universe_composition_v6.parquet",
        index=False,
    )
    composition.to_csv(
        REPORTS / "pit_multimarket_overlap_by_date_v6.csv",
        index=False,
        lineterminator="\n",
    )
    by_date = build_performance_by_date(collapsed, composition)
    by_interval, by_market, macro = build_performance_hierarchy(by_date)
    current_minus_pit, date_diff = build_current_minus_pit(
        by_date, by_interval, by_market, macro
    )
    current_minus_pit = add_rank_difference(
        current_minus_pit, by_interval, by_market, macro
    )
    inference = build_bootstrap_inference(date_diff)
    seed_robustness = build_seed_robustness(seed_predictions)
    rank = build_model_rank(
        by_interval,
        by_market,
        collapsed,
        seed_robustness,
    )
    diagnostic = build_intersection_diagnostic(collapsed, by_interval)
    leave_one_out = build_leave_one_market_out(
        current_minus_pit, config["included_markets"]
    )
    heterogeneity = build_heterogeneity(current_minus_pit, inference)
    sensitivity = build_sensitivity(
        inventory,
        by_interval,
        inference,
        collapsed,
        leave_one_out,
    )
    associations = build_composition_associations(date_diff, composition)
    reconciliation = build_set50_v4_v6_reconciliation(by_interval)

    outputs = {
        PERFORMANCE_BASES[0]: by_date,
        PERFORMANCE_BASES[1]: by_interval,
        PERFORMANCE_BASES[2]: by_market,
        PERFORMANCE_BASES[3]: macro,
        PERFORMANCE_BASES[4]: current_minus_pit,
        PERFORMANCE_BASES[5]: inference,
        PERFORMANCE_BASES[6]: seed_robustness,
        PERFORMANCE_BASES[7]: rank,
        PERFORMANCE_BASES[8]: diagnostic,
        PERFORMANCE_BASES[9]: invalid,
        PERFORMANCE_BASES[10]: leave_one_out,
        PERFORMANCE_BASES[11]: heterogeneity,
    }
    for base, frame in outputs.items():
        write_csv_parquet(base, frame)
    write_csv_parquet("pit_multimarket_sensitivity_v6", sensitivity)
    associations.to_csv(
        REPORTS / "pit_multimarket_composition_associations_v6.csv",
        index=False,
        lineterminator="\n",
    )
    reconciliation.to_csv(
        REPORTS / "set50_v4_v6_protocol_reconciliation.csv",
        index=False,
        lineterminator="\n",
    )

    v4_integrity = verify_locked_v4()
    initial_status = build_status_payload(
        inventory=inventory_all,
        observations=observations,
        features=features,
        rejections=rejections,
        folds=folds,
        seed_predictions=seed_predictions,
        collapsed=collapsed,
        all_cells=all_cells,
        by_interval=by_interval,
        by_market=by_market,
        macro=macro,
        inference=inference,
        rank=rank,
        diagnostic=diagnostic,
        leave_one_out=leave_one_out,
        heterogeneity=heterogeneity,
        composition=composition,
        v4_integrity=v4_integrity,
        artifact_validation_status="PENDING",
    )
    write_status(initial_status)
    manifest = build_manifest(required_artifact_paths())
    validation = validate_artifacts(
        manifest,
        inventory=inventory_all,
        folds=folds,
        collapsed=collapsed,
        all_cells=all_cells,
    )
    final_status = build_status_payload(
        inventory=inventory_all,
        observations=observations,
        features=features,
        rejections=rejections,
        folds=folds,
        seed_predictions=seed_predictions,
        collapsed=collapsed,
        all_cells=all_cells,
        by_interval=by_interval,
        by_market=by_market,
        macro=macro,
        inference=inference,
        rank=rank,
        diagnostic=diagnostic,
        leave_one_out=leave_one_out,
        heterogeneity=heterogeneity,
        composition=composition,
        v4_integrity=v4_integrity,
        artifact_validation_status=validation["status"],
    )
    write_status(final_status)
    # Status changed after the first manifest pass; refresh checksums once.
    manifest = build_manifest(required_artifact_paths())
    validation = validate_artifacts(
        manifest,
        inventory=inventory_all,
        folds=folds,
        collapsed=collapsed,
        all_cells=all_cells,
    )
    return {
        "status": final_status["status"],
        "eligible_intervals": len(inventory),
        "valid_cells": int(all_cells["valid"].astype(bool).sum()),
        "invalid_cells": int((~all_cells["valid"].astype(bool)).sum()),
        "seed_level_predictions": len(seed_predictions),
        "collapsed_predictions": len(collapsed),
        "manifest_records": manifest["artifact_count"],
        "artifact_validation": validation["status"],
    }


def final_console_text() -> str:
    status_path = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V6.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    market = pd.DataFrame(
        payload["current_versus_pit"]["primary_market_inference"]
    )
    macro = pd.DataFrame(
        payload["current_versus_pit"][
            "primary_equal_market_macro_inference"
        ]
    )
    rank = pd.DataFrame(payload["model_selection"]["market_rank_summary"])
    diagnostic = pd.DataFrame(payload["intersection_control"])
    heterogeneity = pd.DataFrame(
        payload["current_versus_pit"]["heterogeneity"]
    )
    lines = [
        f"STATUS: {payload['status']}",
        "",
        "PORTABILITY:",
        (
            "* manifest path issue fixed: "
            + (
                "yes"
                if payload["portability"]["manifest_path_issue_fixed"]
                else "no"
            )
        ),
        (
            "* repository tests passed: "
            f"{payload['portability']['repository_tests_passed']}"
        ),
        (
            "* repository tests failed: "
            f"{payload['portability']['repository_tests_failed']}"
        ),
        "",
        "ELIGIBLE SCOPE:",
        (
            "* included markets: "
            + ", ".join(payload["eligible_scope"]["included_markets"])
        ),
        (
            "* excluded markets: "
            + ", ".join(payload["eligible_scope"]["excluded_markets"])
        ),
        (
            "* forecasting-ready intervals by market: "
            + json.dumps(
                payload["eligible_scope"]["eligible_intervals_by_market"],
                sort_keys=True,
            )
        ),
        (
            "* total eligible trading dates: "
            f"{payload['eligible_scope']['total_eligible_trading_dates']}"
        ),
        "",
        "FORECASTING:",
        f"* valid model cells: {payload['forecasting_data']['valid_cells']}",
        f"* invalid model cells: {payload['forecasting_data']['invalid_cells']}",
        (
            "* seed-level prediction rows: "
            f"{payload['forecasting_data']['seed_level_predictions']}"
        ),
        (
            "* collapsed prediction rows: "
            f"{payload['forecasting_data']['seed_collapsed_predictions']}"
        ),
        "",
        "PRIMARY CURRENT-MINUS-PIT RESULTS:",
    ]
    for record in market.sort_values(["index_id", "model_id"]).to_dict(
        "records"
    ):
        lines.append(
            f"* {record['index_id']} / {record['model_id']}: "
            f"{record['point_estimate']:+.6f}, "
            f"95% CI [{record['ci_lower']:+.6f}, "
            f"{record['ci_upper']:+.6f}], "
            f"BH p={record['adjusted_p_value']:.4f}"
        )
    for record in macro.sort_values("model_id").to_dict("records"):
        lines.append(
            f"* equal-market macro / {record['model_id']}: "
            f"{record['point_estimate']:+.6f}, "
            f"95% CI [{record['ci_lower']:+.6f}, "
            f"{record['ci_upper']:+.6f}], "
            f"BH p={record['adjusted_p_value']:.4f}, "
            f"direction agreement={record['direction_agreement']:.3f}"
        )
    lines.append(
        "* heterogeneity classification: "
        + ", ".join(
            f"{row['model_id']}={row['heterogeneity_classification']}"
            for row in heterogeneity.to_dict("records")
        )
    )
    lines.extend(["", "MODEL RANKING:"])
    for (index_id,), group in rank.groupby(["index_id"], sort=True):
        first = group.iloc[0]
        lines.append(
            f"* {index_id}: rank correlation="
            f"{first['PIT_current_spearman_rank_correlation']:.3f}, "
            f"top-model agreement={first['top_model_agreement']}, "
            f"current-to-PIT regret="
            f"{first['current_winner_deployed_on_PIT_regret']:+.6f}, "
            f"PIT-to-current regret="
            f"{first['PIT_winner_evaluated_on_current_regret']:+.6f}"
        )
    lines.extend(["", "INTERSECTION CONTROL:"])
    for model_id, group in diagnostic.groupby("model_id", sort=True):
        lines.append(
            f"* {model_id}: full-universe contrast="
            f"{group['full_universe_delta_brier_current_minus_pit'].mean():+.6f}, "
            f"intersection contrast="
            f"{group['common_security_evaluation_delta_brier_current_minus_pit'].mean():+.6f}, "
            f"diagnostic difference="
            f"{group['diagnostic_composition_component'].mean():+.6f}"
        )
    lines.extend(
        [
            "",
            "TESTS:",
            f"* passed: {payload['tests']['passed']}",
            f"* failed: {payload['tests']['failed']}",
            "",
            "OVERALL FIVE-MARKET PIT STATUS:",
            "PARTIAL",
            "",
            "MAIN BLOCKERS:",
            *[f"* {item}" for item in payload["main_blockers"]],
            "",
            "OUTPUT REPORT:",
            (
                "reports/data_execution/"
                "PIT_MULTIMARKET_FORECASTING_STATUS_V6.md"
            ),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    result = run_analytics()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
