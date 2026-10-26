#!/usr/bin/env python3
"""Preregistered selected-market ASEAN PIT forecasting benchmark V6.

The runner is additive: it reads V5 data gates, preserves the locked SET50 V4
experiment, and writes V6 artifacts only.  PSEi and FBM KLCI remain excluded
unless the V5 gate itself makes an interval forecasting-ready.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
SCRIPT_PATH = Path(__file__).resolve()
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pit_pipeline_v2 as v2  # noqa: E402
import pit_pipeline_v3 as v3  # noqa: E402

EXPERIMENT_ID = "SELECTED_ASEAN_PIT_PARTIAL_INTERVAL_BENCHMARK_V6"
EXPERIMENT_VERSION = "v6"
PIPELINE_VERSION = "6.0.0"
RUN_DATE = "2026-08-02"
DATA_VERSION = "pit-data-v5"
MEMBERSHIP_VERSION = "pit-membership-v5"
SNAPSHOT_VERSION = "pit-universe-v5"
PRICE_VERSION = "pit-data-v5"
MANIFEST_VERSION = "manifest-portability-v6.0.0"
ELIGIBILITY_VERSION = "pit-interval-eligibility-v5"
FEATURE_VERSION = "raw_ohlc_registry_d16ba71d917fc1f9ce020345"
FEATURE_COLUMNS = [
    "lagged_open_close_return",
    "lagged_close_close_return",
    "lagged_overnight_gap",
    "lagged_high_low_range",
    "lagged_rolling_volatility_5",
    "lagged_momentum_5",
    "lagged_downside_rate_5",
    "lagged_market_return",
    "lagged_market_breadth",
]
FEATURE_LOOKBACK = 5
PIT_SEMANTICS = "POINT_IN_TIME"
CURRENT_SEMANTICS = "CURRENT_2026_APPLIED_BACKWARD"
INTERSECTION_SEMANTICS = "PIT_CURRENT_INTERSECTION_CONTROL"
ARM_SEMANTICS = {
    "pit": PIT_SEMANTICS,
    "current": CURRENT_SEMANTICS,
    "intersection": INTERSECTION_SEMANTICS,
}
SEEDS = [11, 23, 37, 53, 71]
MODEL_SPECS = {
    "historical_class_probability": {
        "family": "historical_class_probability",
        "module": "information_set_baseline",
        "parameters": {
            "history": "matured_labels_before_calibration_start",
            "fallback_probability": 0.5,
        },
    },
    "logistic_regression": {
        "family": "logistic_regression",
        "module": (
            "papers.paper_03_raw_ohlc_forecasting.src.models."
            "machine_learning.logistic_regression"
        ),
        "parameters": {"C": 1.0, "max_iter": 500},
    },
    "random_forest": {
        "family": "random_forest",
        "module": (
            "papers.paper_04_regime_transfer_reliability.src.models."
            "machine_learning.random_forest"
        ),
        "parameters": {
            "n_estimators": 60,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "max_samples": 0.8,
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
        },
    },
    "lightgbm": {
        "family": "lightgbm",
        "module": (
            "papers.paper_03_raw_ohlc_forecasting.src.models."
            "machine_learning.lightgbm"
        ),
        "parameters": {
            "n_estimators": 50,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 30,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
            "verbosity": -1,
        },
    },
}
INITIAL_TRAINING_DATES = 60
CALIBRATION_DATES = 20
TEST_BLOCK_DATES = 20
STEP_DATES = 20
MIN_VALID_FOLDS = 2
MIN_OOS_DATES = 40
MIN_OOS_ROWS = 800
MAX_TARGET_GAP_DAYS = 14
PREDICTION_THRESHOLD = 0.5
PRICE_COVERAGE_THRESHOLD = 0.95
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 271828
PRIMARY_BLOCK_LENGTH = 20
ROBUSTNESS_BLOCK_LENGTHS = [10, 40]
STUDY_LABEL = "SELECTED_ASEAN_PIT_PARTIAL_INTERVAL_BENCHMARK_V6"

INVENTORY_PATH = (
    PIT_ROOT / "reports" / "forecasting_ready_interval_inventory_v6.csv"
)
TERMINOLOGY_PATH = REPORT_ROOT / "PIT_V6_TERMINOLOGY_RECONCILIATION.md"
CONFIG_PATH = PIT_ROOT / "config" / "pit_multimarket_forecasting_v6.yaml"

INVENTORY_COLUMNS = [
    "interval_id",
    "market",
    "index_id",
    "chain_id",
    "interval_start",
    "interval_end_exclusive",
    "calendar_days",
    "trading_dates",
    "PIT_security_count",
    "current_security_count",
    "common_security_count",
    "mean_jaccard_overlap",
    "turnover_events",
    "required_active_member_days",
    "covered_active_member_days",
    "price_coverage",
    "feature_lookback_available",
    "target_coverage_available",
    "high_severity_conflicts",
    "PIT_raw_ohlcv_rows",
    "current_raw_ohlcv_rows",
    "eligible_for_forecasting",
    "eligibility_version",
    "notes",
]


def as_date(value: Any) -> date | None:
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(str(value), errors="coerce")
    return None if pd.isna(parsed) else parsed.date()


def iso(value: Any) -> str | None:
    parsed = as_date(value)
    return parsed.isoformat() if parsed else None


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return iso(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
) -> None:
    frame = pd.DataFrame(list(rows), columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def write_parquet(
    path: Path,
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    frame.to_parquet(path, index=False, engine="pyarrow")


def repository_commit_hash() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return completed.stdout.strip()
    return "NOT_A_GIT_WORKTREE"


def current_universe_tickers() -> dict[str, set[str]]:
    frames = v3.current_tickers(REPO_ROOT)
    return {
        index_id: {
            value
            for value in frame.get("local_symbol", pd.Series(dtype=str))
            .astype(str)
            .str.strip()
            if value
        }
        for index_id, frame in frames.items()
    }


def _membership_interval_sets(
    membership: pd.DataFrame,
) -> dict[str, list[tuple[date, date, set[str], str]]]:
    result: dict[str, list[tuple[date, date, set[str], str]]] = defaultdict(list)
    for (
        index_id,
        start_value,
        end_value,
        chain_id,
    ), group in membership.groupby(
        [
            "index_id",
            "effective_from",
            "effective_to",
            "interval_chain_id",
        ],
        sort=True,
    ):
        result[str(index_id)].append(
            (
                as_date(start_value),
                as_date(end_value),
                set(group["ticker_on_effective_date"].astype(str)),
                str(chain_id),
            )
        )
    for index_id in result:
        result[index_id].sort(key=lambda item: (item[0], item[1]))
    return result


def _recalculated_ready(record: Mapping[str, Any]) -> bool:
    return bool(
        record.get("event_chain_complete")
        and int(record.get("required_securities") or 0) > 0
        and int(record.get("resolved_securities") or 0)
        == int(record.get("required_securities") or 0)
        and math.isclose(
            float(record.get("identity_resolution_rate") or 0.0),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and float(record.get("price_coverage") or 0.0)
        >= PRICE_COVERAGE_THRESHOLD
        and bool(record.get("feature_lookback_available"))
        and bool(record.get("target_coverage_available"))
        and int(record.get("high_severity_conflicts") or 0) == 0
        and bool(record.get("eligible_for_descriptive_analysis"))
    )


def build_interval_inventory() -> pd.DataFrame:
    eligibility = pd.read_parquet(
        PIT_ROOT / "qa" / "pit_interval_eligibility_v5.parquet"
    )
    membership = pd.read_parquet(
        PIT_ROOT / "processed" / "index_membership_v5.parquet"
    )
    snapshots = pd.read_parquet(
        PIT_ROOT / "snapshots" / "pit_universe_daily_v5.parquet"
    )
    pit_panel = pd.read_parquet(
        PIT_ROOT / "processed" / "pit_raw_ohlc_panel_v5.parquet"
    )
    current_panel = pd.read_parquet(
        PIT_ROOT / "processed" / "current_universe_matched_panel_v5.parquet"
    )
    current = current_universe_tickers()
    membership_sets = _membership_interval_sets(membership)
    rows: list[dict[str, Any]] = []
    for gate in eligibility.sort_values(
        ["market", "index_id", "interval_start"], kind="stable"
    ).to_dict("records"):
        interval_id = str(gate["interval_id"])
        index_id = str(gate["index_id"])
        start = as_date(gate["interval_start"])
        end = as_date(gate["interval_end_exclusive"])
        candidates = [
            item
            for item in membership_sets.get(index_id, [])
            if item[0] == start and item[1] == end
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{interval_id} must map to one V5 membership chain, got {len(candidates)}"
            )
        _, _, pit_tickers, chain_id = candidates[0]
        current_tickers = current.get(index_id, set())
        common = pit_tickers.intersection(current_tickers)
        union = pit_tickers.union(current_tickers)
        interval_snapshots = snapshots[
            snapshots["interval_id"].astype(str).eq(interval_id)
        ]
        trading_dates = int(interval_snapshots["date"].nunique())
        prior = [
            item
            for item in membership_sets[index_id]
            if item[1] <= start
        ]
        previous_tickers = prior[-1][2] if prior else set()
        turnover = (
            len(pit_tickers.symmetric_difference(previous_tickers))
            if previous_tickers
            else 0
        )
        recalculated = _recalculated_ready(gate)
        recorded = bool(gate["eligible_for_forecasting"])
        if recalculated != recorded:
            raise RuntimeError(
                f"V5 readiness mismatch for {interval_id}: "
                f"recalculated={recalculated}, recorded={recorded}"
            )
        notes = (
            "RECALCULATED_FORECASTING_READY"
            if recalculated
            else "DESCRIPTIVE_READY_ONLY: "
            + str(gate.get("blocking_reasons") or "one or more locked gates failed")
        )
        rows.append(
            {
                "interval_id": interval_id,
                "market": gate["market"],
                "index_id": index_id,
                "chain_id": chain_id,
                "interval_start": start,
                "interval_end_exclusive": end,
                "calendar_days": (end - start).days,
                "trading_dates": trading_dates,
                "PIT_security_count": len(pit_tickers),
                "current_security_count": len(current_tickers),
                "common_security_count": len(common),
                "mean_jaccard_overlap": (
                    len(common) / len(union) if union else 0.0
                ),
                "turnover_events": turnover,
                "required_active_member_days": int(
                    gate["required_active_member_days"]
                ),
                "covered_active_member_days": int(
                    gate["covered_active_member_days"]
                ),
                "price_coverage": float(gate["price_coverage"]),
                "feature_lookback_available": bool(
                    gate["feature_lookback_available"]
                ),
                "target_coverage_available": bool(
                    gate["target_coverage_available"]
                ),
                "high_severity_conflicts": int(
                    gate["high_severity_conflicts"]
                ),
                "PIT_raw_ohlcv_rows": int(
                    len(
                        pit_panel[
                            pit_panel["interval_id"].astype(str).eq(interval_id)
                        ]
                    )
                ),
                "current_raw_ohlcv_rows": int(
                    len(
                        current_panel[
                            current_panel["interval_id"].astype(str).eq(
                                interval_id
                            )
                        ]
                    )
                ),
                "eligible_for_forecasting": recalculated,
                "eligibility_version": ELIGIBILITY_VERSION,
                "notes": notes,
            }
        )
    frame = pd.DataFrame(rows, columns=INVENTORY_COLUMNS)
    ready = frame[frame["eligible_for_forecasting"].astype(bool)]
    expected = {"SET50": 2, "VN30": 2, "IDX30": 2}
    actual = ready.groupby("index_id").size().to_dict()
    if actual != expected or len(ready) != 6:
        raise RuntimeError(
            f"unexpected V6 ready scope: expected {expected}, observed {actual}"
        )
    if set(ready["index_id"]) != {"SET50", "VN30", "IDX30"}:
        raise RuntimeError("an unexpected market passed the recomputed V5 gates")
    if len(frame) != 59:
        raise RuntimeError(f"expected 59 chain-level intervals, got {len(frame)}")
    write_csv(INVENTORY_PATH, frame.to_dict("records"), INVENTORY_COLUMNS)
    return frame


def write_terminology_reconciliation() -> None:
    text = """# PIT V6 Terminology Reconciliation

V6 uses each term at one fixed grain:

- **Review period:** a scheduled or extraordinary index review represented by
  dated source evidence. A review period is evidence input, not automatically
  an eligible interval.
- **Event chain:** the ordered anchor and dated membership evidence supporting
  one bounded market interval without crossing an unresolved gap.
- **Market-level eligible interval:** one index/date range evaluated once in
  `pit_interval_eligibility_v5.parquet`. V5 has 59 chain-level market
  intervals; six independently pass every locked forecasting gate.
- **Security membership interval:** one security's membership record inside a
  market-level chain. V5 has 2,010 security-level membership rows. This term is
  never used for the count 59.
- **Daily PIT snapshot:** one evidence-backed active security on one observed
  market trading date, with `POINT_IN_TIME` semantics.
- **Active-member day:** one expected security/date pair inside a supported
  membership interval, whether or not raw OHLC joined.
- **Forecasting observation:** one valid prediction-date/next-valid-target-date
  pair after chronology, membership, ticker, gap, price, and feature gates.

Subsequent V6 reports use **market-level interval** for the 59/6 interval
counts and **security membership row** for the 2,010-row membership table.
"""
    TERMINOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    TERMINOLOGY_PATH.write_text(text, encoding="utf-8")


def validate_preflight(inventory: pd.DataFrame) -> dict[str, Any]:
    ready = inventory[inventory["eligible_for_forecasting"].astype(bool)]
    result = {
        "validated": True,
        "inventory_rows": len(inventory),
        "eligible_rows": len(ready),
        "eligible_by_index": ready.groupby("index_id").size().to_dict(),
        "included_markets": sorted(set(ready["index_id"])),
        "excluded_markets": ["PSEI", "FBM_KLCI"],
        "descriptive_not_forecasting": int(
            (~inventory["eligible_for_forecasting"].astype(bool)).sum()
        ),
        "inventory_sha256": sha256_file(INVENTORY_PATH),
        "terminology_sha256": sha256_file(TERMINOLOGY_PATH),
    }
    write_json(PIT_ROOT / "qa" / "pit_v6_preflight_validation.json", result)
    return result


def package_versions() -> dict[str, str]:
    modules = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "pyarrow"]
    result = {"python": platform.python_version()}
    for name in modules:
        module = importlib.import_module(name)
        result[name] = str(getattr(module, "__version__", "UNKNOWN"))
    return result


def expected_config(inventory: pd.DataFrame) -> dict[str, Any]:
    ready = inventory[inventory["eligible_for_forecasting"].astype(bool)]
    feature_source = (
        REPO_ROOT
        / "papers"
        / "paper_03_raw_ohlc_forecasting"
        / "src"
        / "features"
        / "raw_ohlc.py"
    )
    excluded = sorted(set(v2.INDEX_CONFIG) - set(ready["index_id"].astype(str)))
    return {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "code_commit_hash": repository_commit_hash(),
        "data_version": DATA_VERSION,
        "membership_version": MEMBERSHIP_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "price_version": PRICE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "eligibility_version": ELIGIBILITY_VERSION,
        "eligible_interval_ids": sorted(ready["interval_id"].astype(str)),
        "included_markets": sorted(set(ready["index_id"].astype(str))),
        "excluded_markets": excluded,
        "exclusion_reasons": {
            "PSEI": "no V5 market-level interval passes eligible_for_forecasting",
            "FBM_KLCI": "no V5 market-level interval passes eligible_for_forecasting",
        },
        "universe_arms": {
            "pit": PIT_SEMANTICS,
            "current": CURRENT_SEMANTICS,
            "intersection": INTERSECTION_SEMANTICS,
        },
        "feature_registry_hash": sha256_file(feature_source),
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "feature_lookback": FEATURE_LOOKBACK,
        "target_definition": {
            "name": "next-session open-to-close direction",
            "return": "target_close / target_open - 1",
            "direction": "1 if target_return > 0 else 0",
            "next_observation": "next valid same-security observation",
            "max_gap_calendar_days": MAX_TARGET_GAP_DAYS,
            "adjusted_close_allowed": False,
            "volume_allowed": False,
            "imputation_allowed": False,
        },
        "model_registry": MODEL_SPECS,
        "hyperparameters": {
            model_id: spec["parameters"] for model_id, spec in MODEL_SPECS.items()
        },
        "seed_registry": SEEDS,
        "rolling_origin_protocol": {
            "initial_training_dates": INITIAL_TRAINING_DATES,
            "calibration_dates": CALIBRATION_DATES,
            "test_block_dates": TEST_BLOCK_DATES,
            "step_dates": STEP_DATES,
            "training_window": "expanding",
            "labels_must_mature_before_fit": True,
        },
        "minimum_sample_gates": {
            "valid_folds": MIN_VALID_FOLDS,
            "oos_trading_dates": MIN_OOS_DATES,
            "oos_security_date_predictions": MIN_OOS_ROWS,
            "both_target_classes_train": True,
            "both_target_classes_test": True,
        },
        "primary_metric": "brier_score",
        "secondary_metrics": [
            "log_loss",
            "balanced_accuracy",
            "calibration_intercept",
            "calibration_slope",
            "coverage",
            "model_rank",
        ],
        "aggregation_rules": {
            "level_1": "equal_security_mean_within_date",
            "level_2": "equal_date_mean_within_interval",
            "level_3": "equal_interval_mean_within_market",
            "level_4": "equal_market_mean_selected_market_macro",
            "security_weighted": "secondary_robustness_only",
        },
        "bootstrap_rules": {
            "method": "paired_moving_block_bootstrap_common_trading_dates",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "primary_block_length": PRIMARY_BLOCK_LENGTH,
            "robustness_block_lengths": ROBUSTNESS_BLOCK_LENGTHS,
        },
        "multiple_testing_family": (
            "Benjamini-Hochberg within block length across all valid "
            "interval-model current-minus-PIT comparisons"
        ),
        "result_classification_rules": {
            "SUPPORTED_CURRENT_PIT_DIFFERENCE": (
                "BH-adjusted p <= 0.05 and 95% CI excludes zero"
            ),
            "NO_SUPPORTED_DIFFERENCE": (
                "valid information and 95% CI includes zero; not equivalence"
            ),
            "INSUFFICIENT_INFORMATION": "cell fails any minimum validity gate",
            "HETEROGENEOUS_ACROSS_MARKETS": (
                "valid market directions disagree and heterogeneity is material"
            ),
        },
        "package_versions": package_versions(),
        "frozen_on": RUN_DATE,
        "correction_policy": "any change requires V6.1 or later experiment ID",
    }


def freeze_config(inventory: pd.DataFrame) -> dict[str, Any]:
    config = expected_config(inventory)
    rendered = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CONFIG_PATH.is_file():
        existing = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError(
                "frozen V6 config differs from current preregistered inputs; "
                "create V6.1 instead of modifying V6"
            )
    else:
        CONFIG_PATH.write_text(rendered, encoding="utf-8")
    return config


def run_preflight(*, freeze: bool) -> dict[str, Any]:
    inventory = build_interval_inventory()
    write_terminology_reconciliation()
    validation = validate_preflight(inventory)
    if freeze:
        config = freeze_config(inventory)
        validation["config_sha256"] = sha256_file(CONFIG_PATH)
        validation["experiment_id"] = config["experiment_id"]
        write_json(PIT_ROOT / "qa" / "pit_v6_preflight_validation.json", validation)
    return validation


OBSERVATION_COLUMNS = [
    "experiment_id",
    "interval_id",
    "market",
    "index_id",
    "universe_semantics",
    "security_id",
    "ticker_on_date",
    "prediction_date",
    "target_date",
    "open",
    "high",
    "low",
    "close",
    "target_open",
    "target_close",
    "target_return",
    "target_direction",
    "membership_chain_id",
    "membership_version",
    "price_version",
    "valid_for_model",
    "rejection_reason",
]

FEATURE_METADATA_COLUMNS = [
    "feature_version",
    "feature_registry_hash",
    "feature_lookback",
    "feature_available_date",
    "null_reason",
    "universe_semantics",
    "interval_id",
]

FOLD_COLUMNS = [
    "interval_id",
    "fold_id",
    "training_start",
    "training_end",
    "calibration_start",
    "calibration_end",
    "test_start",
    "test_end",
    "label_maturity_cutoff",
    "training_dates",
    "test_dates",
    "valid_fold",
    "invalid_reason",
]


def security_id(index_id: str, ticker: str) -> str:
    config = v2.INDEX_CONFIG[index_id]
    return stable_id("SEC", config["market"], config["exchange"], ticker)


def _ready_inventory() -> pd.DataFrame:
    inventory = pd.read_csv(INVENTORY_PATH)
    inventory["eligible_for_forecasting"] = (
        inventory["eligible_for_forecasting"].astype(str).str.lower().eq("true")
    )
    return inventory[inventory["eligible_for_forecasting"]].copy()


def build_arm_membership_dates(
    inventory: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    snapshots = pd.read_parquet(
        PIT_ROOT / "snapshots" / "pit_universe_daily_v5.parquet"
    )
    current = current_universe_tickers()
    membership = pd.read_parquet(
        PIT_ROOT / "processed" / "index_membership_v5.parquet"
    )
    chain_by_interval: dict[str, str] = {}
    for record in inventory.to_dict("records"):
        interval_id = str(record["interval_id"])
        start, end = as_date(record["interval_start"]), as_date(
            record["interval_end_exclusive"]
        )
        chains = membership[
            membership["index_id"].astype(str).eq(str(record["index_id"]))
            & pd.to_datetime(
                membership["effective_from"], errors="coerce"
            ).dt.date.eq(start)
            & pd.to_datetime(
                membership["effective_to"], errors="coerce"
            ).dt.date.eq(end)
        ]["interval_chain_id"].dropna().astype(str).unique()
        if len(chains) != 1:
            raise RuntimeError(
                f"{interval_id} maps to {len(chains)} membership chains"
            )
        chain_by_interval[interval_id] = chains[0]

    pit = snapshots[
        snapshots["interval_id"].astype(str).isin(
            set(inventory["interval_id"].astype(str))
        )
    ][
        [
            "interval_id",
            "market",
            "index_id",
            "security_id",
            "ticker_on_date",
            "date",
            "interval_chain_id",
        ]
    ].copy()
    pit = pit.rename(
        columns={
            "date": "membership_date",
            "interval_chain_id": "membership_chain_id",
        }
    )
    pit["universe_semantics"] = PIT_SEMANTICS
    pit["membership_version"] = MEMBERSHIP_VERSION

    current_rows: list[dict[str, Any]] = []
    intersection_rows: list[dict[str, Any]] = []
    for record in inventory.to_dict("records"):
        interval_id = str(record["interval_id"])
        index_id = str(record["index_id"])
        market = str(record["market"])
        interval_pit = pit[pit["interval_id"].astype(str).eq(interval_id)]
        dates = sorted(
            {
                as_date(value)
                for value in interval_pit["membership_date"]
                if as_date(value)
            }
        )
        current_tickers = current.get(index_id, set())
        current_ids = {
            security_id(index_id, ticker): ticker for ticker in current_tickers
        }
        pit_ids = {
            str(row["security_id"]): str(row["ticker_on_date"])
            for row in interval_pit[
                ["security_id", "ticker_on_date"]
            ].drop_duplicates().to_dict("records")
        }
        common_ids = set(pit_ids).intersection(current_ids)
        for on_date in dates:
            for sec, ticker in sorted(current_ids.items()):
                current_rows.append(
                    {
                        "interval_id": interval_id,
                        "market": market,
                        "index_id": index_id,
                        "security_id": sec,
                        "ticker_on_date": ticker,
                        "membership_date": on_date,
                        "membership_chain_id": (
                            f"{chain_by_interval[interval_id]}+"
                            "CURRENT_2026_CONFIG"
                        ),
                        "universe_semantics": CURRENT_SEMANTICS,
                        "membership_version": (
                            "current-universe-2026-config-v1"
                        ),
                    }
                )
            for sec in sorted(common_ids):
                intersection_rows.append(
                    {
                        "interval_id": interval_id,
                        "market": market,
                        "index_id": index_id,
                        "security_id": sec,
                        "ticker_on_date": pit_ids[sec],
                        "membership_date": on_date,
                        "membership_chain_id": (
                            f"{chain_by_interval[interval_id]}+"
                            "CURRENT_2026_CONFIG+INTERSECTION"
                        ),
                        "universe_semantics": INTERSECTION_SEMANTICS,
                        "membership_version": (
                            f"{MEMBERSHIP_VERSION}+"
                            "current-universe-2026-config-v1"
                        ),
                    }
                )
    columns = [
        "interval_id",
        "market",
        "index_id",
        "security_id",
        "ticker_on_date",
        "membership_date",
        "membership_chain_id",
        "universe_semantics",
        "membership_version",
    ]
    current_frame = pd.DataFrame(current_rows, columns=columns)
    intersection = pd.DataFrame(intersection_rows, columns=columns)
    for frame in (pit, current_frame, intersection):
        if frame.duplicated(
            ["interval_id", "membership_date", "security_id"]
        ).any():
            raise RuntimeError("duplicate arm membership date key")
    return {
        "pit": pit.reindex(columns=columns).reset_index(drop=True),
        "current": current_frame,
        "intersection": intersection,
    }


def construct_observations(
    inventory: pd.DataFrame,
    arm_members: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_parquet(
        PIT_ROOT / "processed" / "historical_raw_ohlc_v5.parquet"
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
    raw_columns = [
        "market",
        "security_id",
        "date",
        "ticker_used",
        "open",
        "high",
        "low",
        "close",
    ]
    raw = raw[raw_columns].drop_duplicates(
        ["market", "security_id", "date"], keep="first"
    )
    valid_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    for arm, members in arm_members.items():
        semantics = ARM_SEMANTICS[arm]
        for (
            interval_id,
            market,
            index_id,
            sec,
        ), member_group in members.groupby(
            ["interval_id", "market", "index_id", "security_id"],
            sort=True,
        ):
            member_group = member_group.sort_values(
                "membership_date", kind="stable"
            )
            ticker_by_date = {
                as_date(row["membership_date"]): str(row["ticker_on_date"])
                for row in member_group.to_dict("records")
            }
            member_dates = set(ticker_by_date)
            security_raw = raw[
                raw["market"].astype(str).eq(str(market))
                & raw["security_id"].astype(str).eq(str(sec))
                & raw["date"].isin(member_dates)
            ].sort_values("date", kind="stable")
            raw_by_date = {
                row["date"]: row for row in security_raw.to_dict("records")
            }
            available_dates = sorted(raw_by_date)
            next_date = {
                current_date: target_date
                for current_date, target_date in zip(
                    available_dates, available_dates[1:]
                )
            }
            chain_id = str(member_group.iloc[0]["membership_chain_id"])
            membership_version = str(
                member_group.iloc[0]["membership_version"]
            )
            for prediction_date in sorted(member_dates):
                ticker = ticker_by_date[prediction_date]
                base = {
                    "experiment_id": EXPERIMENT_ID,
                    "interval_id": interval_id,
                    "market": market,
                    "index_id": index_id,
                    "universe_semantics": semantics,
                    "security_id": sec,
                    "ticker_on_date": ticker,
                    "prediction_date": prediction_date,
                    "target_date": None,
                    "open": np.nan,
                    "high": np.nan,
                    "low": np.nan,
                    "close": np.nan,
                    "target_open": np.nan,
                    "target_close": np.nan,
                    "target_return": np.nan,
                    "target_direction": np.nan,
                    "membership_chain_id": chain_id,
                    "membership_version": membership_version,
                    "price_version": PRICE_VERSION,
                    "valid_for_model": False,
                    "rejection_reason": None,
                }
                prediction_raw = raw_by_date.get(prediction_date)
                reason = None
                if prediction_raw is None:
                    reason = "MISSING_PREDICTION_RAW_OHLC"
                else:
                    for column in ("open", "high", "low", "close"):
                        base[column] = prediction_raw[column]
                    target_date = next_date.get(prediction_date)
                    base["target_date"] = target_date
                    if target_date is None:
                        reason = "NO_NEXT_VALID_SAME_SECURITY_OBSERVATION"
                    elif target_date not in member_dates:
                        reason = "TARGET_OUTSIDE_MEMBERSHIP_INTERVAL"
                    elif (target_date - prediction_date).days > MAX_TARGET_GAP_DAYS:
                        reason = "LONG_TARGET_GAP"
                    else:
                        target_raw = raw_by_date[target_date]
                        base["target_open"] = target_raw["open"]
                        base["target_close"] = target_raw["close"]
                        if (
                            str(prediction_raw["ticker_used"]) != ticker
                            or str(target_raw["ticker_used"])
                            != ticker_by_date[target_date]
                        ):
                            reason = "TICKER_VALIDITY_BOUNDARY"
                if reason is None:
                    target_return = (
                        float(base["target_close"]) / float(base["target_open"])
                        - 1.0
                    )
                    base["target_return"] = target_return
                    base["target_direction"] = int(target_return > 0.0)
                    base["valid_for_model"] = True
                    valid_rows.append(base)
                else:
                    base["rejection_reason"] = reason
                    rejection_rows.append(base)
    observations = pd.DataFrame(valid_rows, columns=OBSERVATION_COLUMNS)
    rejections = pd.DataFrame(rejection_rows, columns=OBSERVATION_COLUMNS)
    if not observations.empty:
        observations = observations.sort_values(
            [
                "interval_id",
                "universe_semantics",
                "prediction_date",
                "security_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
    if observations.duplicated(
        [
            "interval_id",
            "universe_semantics",
            "prediction_date",
            "security_id",
        ]
    ).any():
        raise RuntimeError("duplicate V6 observation keys")
    if not observations.empty:
        prediction = pd.to_datetime(observations["prediction_date"])
        target = pd.to_datetime(observations["target_date"])
        if not (target > prediction).all():
            raise RuntimeError("target chronology failure")
    return observations, rejections


def construct_features(
    observations: pd.DataFrame,
    initial_rejections: pd.DataFrame,
    arm_members: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from papers.paper_03_raw_ohlc_forecasting.src.features.raw_ohlc import (
        build_lagged_raw_features,
    )

    raw = pd.read_parquet(
        PIT_ROOT / "processed" / "historical_raw_ohlc_v5.parquet"
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
    feature_source = (
        REPO_ROOT
        / "papers"
        / "paper_03_raw_ohlc_forecasting"
        / "src"
        / "features"
        / "raw_ohlc.py"
    )
    registry_hash = sha256_file(feature_source)
    feature_frames: list[pd.DataFrame] = []
    feature_rejections: list[pd.DataFrame] = []
    observation_updates: list[pd.DataFrame] = []
    for arm, members in arm_members.items():
        semantics = ARM_SEMANTICS[arm]
        for interval_id, member_group in members.groupby(
            "interval_id", sort=True
        ):
            interval_obs = observations[
                observations["interval_id"].astype(str).eq(str(interval_id))
                & observations["universe_semantics"].astype(str).eq(semantics)
            ].copy()
            if interval_obs.empty:
                continue
            end = max(
                as_date(value) for value in member_group["membership_date"]
            ) + timedelta(days=1)
            tickers_by_security = {
                str(sec): set(group["ticker_on_date"].astype(str))
                for sec, group in member_group.groupby(
                    "security_id", sort=False
                )
            }
            full = raw[
                raw["market"].astype(str).eq(
                    str(member_group.iloc[0]["market"])
                )
                & raw["security_id"].astype(str).isin(tickers_by_security)
                & (raw["date"] < end)
            ].copy()
            full = full[
                [
                    "market",
                    "security_id",
                    "ticker_used",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ]
            full = full[
                [
                    str(row["ticker_used"])
                    in tickers_by_security.get(str(row["security_id"]), set())
                    for row in full.to_dict("records")
                ]
            ]
            full_input = full.rename(
                columns={
                    "security_id": "symbol_canonical",
                    "open": "open_raw",
                    "high": "high_raw",
                    "low": "low_raw",
                    "close": "close_raw",
                }
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full_featured, built_columns = build_lagged_raw_features(
                    full_input
                )
            if built_columns != FEATURE_COLUMNS:
                raise RuntimeError("locked raw-OHLC feature registry changed")
            full_featured["date"] = pd.to_datetime(
                full_featured["date"], errors="coerce"
            ).dt.date
            full_featured = full_featured.rename(
                columns={"symbol_canonical": "security_id"}
            )

            eligible_raw = member_group.merge(
                raw[
                    [
                        "market",
                        "security_id",
                        "date",
                        "open",
                        "high",
                        "low",
                        "close",
                    ]
                ],
                left_on=["market", "security_id", "membership_date"],
                right_on=["market", "security_id", "date"],
                how="inner",
                validate="one_to_one",
            )
            eligible_input = eligible_raw.rename(
                columns={
                    "security_id": "symbol_canonical",
                    "open": "open_raw",
                    "high": "high_raw",
                    "low": "low_raw",
                    "close": "close_raw",
                }
            )[
                [
                    "market",
                    "symbol_canonical",
                    "date",
                    "open_raw",
                    "high_raw",
                    "low_raw",
                    "close_raw",
                ]
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eligible_featured, _ = build_lagged_raw_features(
                    eligible_input
                )
            eligible_featured["date"] = pd.to_datetime(
                eligible_featured["date"], errors="coerce"
            ).dt.date
            market_lookup = eligible_featured[
                [
                    "symbol_canonical",
                    "date",
                    "lagged_market_return",
                    "lagged_market_breadth",
                ]
            ].rename(columns={"symbol_canonical": "security_id"})
            individual = full_featured.drop(
                columns=["lagged_market_return", "lagged_market_breadth"],
                errors="ignore",
            )
            feature_lookup = individual[
                ["security_id", "date", *FEATURE_COLUMNS[:-2]]
            ].merge(
                market_lookup,
                on=["security_id", "date"],
                how="left",
                validate="one_to_one",
            )
            merged = interval_obs.merge(
                feature_lookup,
                left_on=["security_id", "prediction_date"],
                right_on=["security_id", "date"],
                how="left",
                validate="one_to_one",
            ).drop(columns=["date"], errors="ignore")
            feature_values = merged[FEATURE_COLUMNS].apply(
                pd.to_numeric, errors="coerce"
            )
            finite = pd.Series(
                np.isfinite(feature_values.to_numpy()).all(axis=1),
                index=merged.index,
            )
            missing_names = feature_values.apply(
                lambda row: ";".join(
                    column
                    for column in FEATURE_COLUMNS
                    if not np.isfinite(row[column])
                ),
                axis=1,
            )
            merged["feature_version"] = FEATURE_VERSION
            merged["feature_registry_hash"] = registry_hash
            merged["feature_lookback"] = FEATURE_LOOKBACK
            merged["feature_available_date"] = merged["prediction_date"]
            merged["null_reason"] = np.where(
                finite, None, "MISSING_FEATURES:" + missing_names
            )
            merged.loc[~finite, "valid_for_model"] = False
            merged.loc[~finite, "rejection_reason"] = "MISSING_FEATURES"
            observation_updates.append(
                merged[OBSERVATION_COLUMNS].copy()
            )
            rejected = merged.loc[~finite, OBSERVATION_COLUMNS].copy()
            if not rejected.empty:
                feature_rejections.append(rejected)
            feature_frames.append(merged.loc[finite].copy())
    features = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame()
    )
    updated_observations = (
        pd.concat(observation_updates, ignore_index=True)
        if observation_updates
        else observations.copy()
    )
    rejections = pd.concat(
        [initial_rejections, *feature_rejections],
        ignore_index=True,
    )
    if not features.empty:
        features = features.sort_values(
            [
                "interval_id",
                "universe_semantics",
                "prediction_date",
                "security_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
    return updated_observations, features, rejections


def write_observation_feature_artifacts(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    rejections: pd.DataFrame,
) -> None:
    observation_path = (
        PIT_ROOT
        / "processed"
        / "pit_multimarket_observation_panel_v6.parquet"
    )
    feature_path = (
        PIT_ROOT / "processed" / "pit_multimarket_feature_panel_v6.parquet"
    )
    rejection_path = (
        PIT_ROOT
        / "qa"
        / "pit_multimarket_observation_rejections_v6.parquet"
    )
    write_parquet(observation_path, observations, OBSERVATION_COLUMNS)
    write_parquet(feature_path, features)
    write_parquet(rejection_path, rejections, OBSERVATION_COLUMNS)


def build_fold_calendar(
    inventory: pd.DataFrame,
    arm_members: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    pit_members = arm_members["pit"]
    for record in inventory.sort_values(
        ["market", "interval_start"], kind="stable"
    ).to_dict("records"):
        interval_id = str(record["interval_id"])
        dates = sorted(
            {
                as_date(value)
                for value in pit_members.loc[
                    pit_members["interval_id"].astype(str).eq(interval_id),
                    "membership_date",
                ]
                if as_date(value)
            }
        )
        fold_number = 0
        while True:
            training_end_position = INITIAL_TRAINING_DATES + fold_number * STEP_DATES
            calibration_end_position = (
                training_end_position + CALIBRATION_DATES
            )
            test_end_position = calibration_end_position + TEST_BLOCK_DATES
            if test_end_position > len(dates):
                break
            training_dates = dates[:training_end_position]
            calibration = dates[
                training_end_position:calibration_end_position
            ]
            test = dates[calibration_end_position:test_end_position]
            rows.append(
                {
                    "interval_id": interval_id,
                    "fold_id": stable_id(
                        "FOLD", interval_id, fold_number, length=16
                    ),
                    "training_start": training_dates[0],
                    "training_end": training_dates[-1],
                    "calibration_start": calibration[0],
                    "calibration_end": calibration[-1],
                    "test_start": test[0],
                    "test_end": test[-1],
                    "label_maturity_cutoff": training_dates[-1],
                    "training_dates": len(training_dates),
                    "test_dates": len(test),
                    "valid_fold": True,
                    "invalid_reason": None,
                }
            )
            fold_number += 1
        if fold_number == 0:
            rows.append(
                {
                    "interval_id": interval_id,
                    "fold_id": stable_id(
                        "FOLD", interval_id, "INVALID", length=16
                    ),
                    "training_start": dates[0] if dates else None,
                    "training_end": (
                        dates[min(len(dates), INITIAL_TRAINING_DATES) - 1]
                        if dates
                        else None
                    ),
                    "calibration_start": None,
                    "calibration_end": None,
                    "test_start": None,
                    "test_end": None,
                    "label_maturity_cutoff": None,
                    "training_dates": min(len(dates), INITIAL_TRAINING_DATES),
                    "test_dates": 0,
                    "valid_fold": False,
                    "invalid_reason": (
                        f"interval has {len(dates)} trading dates; requires at "
                        f"least {INITIAL_TRAINING_DATES + CALIBRATION_DATES + TEST_BLOCK_DATES}"
                    ),
                }
            )
    calendar = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    valid = calendar[calendar["valid_fold"].astype(bool)]
    for interval_id, group in valid.groupby("interval_id"):
        ordered = group.sort_values("test_start")
        prior_end: date | None = None
        for fold in ordered.to_dict("records"):
            test_start, test_end = as_date(fold["test_start"]), as_date(
                fold["test_end"]
            )
            if prior_end and test_start <= prior_end:
                raise RuntimeError(
                    f"overlapping V6 test blocks for {interval_id}"
                )
            prior_end = test_end
    path = (
        PIT_ROOT / "processed" / "pit_multimarket_fold_calendar_v6.parquet"
    )
    write_parquet(path, calendar, FOLD_COLUMNS)
    return calendar


PREDICTION_COLUMNS = [
    "experiment_id",
    "interval_id",
    "market",
    "index_id",
    "universe_semantics",
    "fold_id",
    "security_id",
    "ticker_on_date",
    "prediction_date",
    "target_date",
    "model_id",
    "model_family",
    "model_version",
    "seed",
    "training_start",
    "training_end",
    "training_row_count",
    "training_class_0",
    "training_class_1",
    "label_maturity_cutoff",
    "prediction_probability",
    "predicted_class",
    "actual_class",
    "feature_version",
    "membership_version",
    "price_version",
    "is_valid_prediction",
]


def _build_model(model_id: str, seed: int) -> Any:
    spec = MODEL_SPECS[model_id]
    if model_id == "historical_class_probability":
        raise ValueError("historical probability baseline has no estimator")
    module = importlib.import_module(str(spec["module"]))
    return module.build_model(seed=seed)


def _predict_probability(model_id: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    module = importlib.import_module(str(MODEL_SPECS[model_id]["module"]))
    probability = np.asarray(module.predict_proba(model, X), dtype=float)
    return np.clip(probability, 0.0, 1.0)


def run_oos_models(
    features: pd.DataFrame,
    fold_calendar: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = features[features["valid_for_model"].astype(bool)].copy()
    features["prediction_date"] = pd.to_datetime(
        features["prediction_date"], errors="coerce"
    ).dt.date
    features["target_date"] = pd.to_datetime(
        features["target_date"], errors="coerce"
    ).dt.date
    rows: list[dict[str, Any]] = []
    execution_failures: list[dict[str, Any]] = []
    valid_folds = fold_calendar[fold_calendar["valid_fold"].astype(bool)]
    for fold in valid_folds.sort_values(
        ["interval_id", "test_start"], kind="stable"
    ).to_dict("records"):
        interval_id = str(fold["interval_id"])
        training_start = as_date(fold["training_start"])
        training_end = as_date(fold["training_end"])
        calibration_start = as_date(fold["calibration_start"])
        test_start = as_date(fold["test_start"])
        test_end = as_date(fold["test_end"])
        for semantics in ARM_SEMANTICS.values():
            data = features[
                features["interval_id"].astype(str).eq(interval_id)
                & features["universe_semantics"].astype(str).eq(semantics)
            ].copy()
            training = data[
                (data["prediction_date"] >= training_start)
                & (data["prediction_date"] <= training_end)
                & (data["target_date"] < calibration_start)
            ].copy()
            test = data[
                (data["prediction_date"] >= test_start)
                & (data["prediction_date"] <= test_end)
            ].copy()
            y_train = training["target_direction"].astype(int)
            y_test = test["target_direction"].astype(int)
            base_failure = None
            if training.empty:
                base_failure = "NO_MATURED_TRAINING_ROWS"
            elif test.empty:
                base_failure = "NO_TEST_ROWS"
            elif y_train.nunique() < 2:
                base_failure = "TRAINING_ONE_CLASS"
            elif y_test.nunique() < 2:
                base_failure = "TEST_ONE_CLASS"
            if base_failure:
                for model_id in MODEL_SPECS:
                    execution_failures.append(
                        {
                            "interval_id": interval_id,
                            "universe_semantics": semantics,
                            "fold_id": fold["fold_id"],
                            "model_id": model_id,
                            "reason": base_failure,
                        }
                    )
                continue
            X_train = training[FEATURE_COLUMNS]
            X_test = test[FEATURE_COLUMNS]
            train_counts = y_train.value_counts().to_dict()
            for model_id, spec in MODEL_SPECS.items():
                for seed in SEEDS:
                    try:
                        if model_id == "historical_class_probability":
                            probability = np.full(
                                len(test), float(y_train.mean()), dtype=float
                            )
                        else:
                            estimator = _build_model(model_id, seed)
                            estimator.fit(X_train, y_train)
                            probability = _predict_probability(
                                model_id, estimator, X_test
                            )
                        if (
                            not np.isfinite(probability).all()
                            or (probability < 0).any()
                            or (probability > 1).any()
                        ):
                            raise ValueError("PROBABILITY_OUT_OF_BOUNDS")
                        for base, value in zip(
                            test.to_dict("records"), probability
                        ):
                            rows.append(
                                {
                                    "experiment_id": EXPERIMENT_ID,
                                    "interval_id": interval_id,
                                    "market": base["market"],
                                    "index_id": base["index_id"],
                                    "universe_semantics": semantics,
                                    "fold_id": fold["fold_id"],
                                    "security_id": base["security_id"],
                                    "ticker_on_date": base["ticker_on_date"],
                                    "prediction_date": base["prediction_date"],
                                    "target_date": base["target_date"],
                                    "model_id": model_id,
                                    "model_family": spec["family"],
                                    "model_version": (
                                        f"{EXPERIMENT_VERSION}-{model_id}-locked"
                                    ),
                                    "seed": seed,
                                    "training_start": min(
                                        training["prediction_date"]
                                    ),
                                    "training_end": max(
                                        training["prediction_date"]
                                    ),
                                    "training_row_count": len(training),
                                    "training_class_0": int(
                                        train_counts.get(0, 0)
                                    ),
                                    "training_class_1": int(
                                        train_counts.get(1, 0)
                                    ),
                                    "label_maturity_cutoff": max(
                                        training["target_date"]
                                    ),
                                    "prediction_probability": float(value),
                                    "predicted_class": int(
                                        float(value) >= PREDICTION_THRESHOLD
                                    ),
                                    "actual_class": int(
                                        base["target_direction"]
                                    ),
                                    "feature_version": base[
                                        "feature_version"
                                    ],
                                    "membership_version": base[
                                        "membership_version"
                                    ],
                                    "price_version": base["price_version"],
                                    "is_valid_prediction": True,
                                }
                            )
                    except Exception as exc:
                        execution_failures.append(
                            {
                                "interval_id": interval_id,
                                "universe_semantics": semantics,
                                "fold_id": fold["fold_id"],
                                "model_id": model_id,
                                "seed": seed,
                                "reason": f"{type(exc).__name__}:{exc}",
                            }
                        )
    seed_level = pd.DataFrame(rows, columns=PREDICTION_COLUMNS)
    if not seed_level.empty:
        seed_level = seed_level.sort_values(
            [
                "interval_id",
                "universe_semantics",
                "fold_id",
                "prediction_date",
                "model_id",
                "security_id",
                "seed",
            ],
            kind="stable",
        ).reset_index(drop=True)
    key = [
        "experiment_id",
        "interval_id",
        "market",
        "index_id",
        "universe_semantics",
        "fold_id",
        "security_id",
        "ticker_on_date",
        "prediction_date",
        "target_date",
        "model_id",
        "model_family",
        "model_version",
        "training_start",
        "training_end",
        "training_row_count",
        "training_class_0",
        "training_class_1",
        "label_maturity_cutoff",
        "actual_class",
        "feature_version",
        "membership_version",
        "price_version",
        "is_valid_prediction",
    ]
    collapsed_rows: list[dict[str, Any]] = []
    if not seed_level.empty:
        for values, group in seed_level.groupby(
            key, dropna=False, sort=True
        ):
            record = dict(zip(key, values))
            record["seed"] = "COLLAPSED"
            record["prediction_probability"] = float(
                group["prediction_probability"].mean()
            )
            record["predicted_class"] = int(
                record["prediction_probability"] >= PREDICTION_THRESHOLD
            )
            record["seed_count"] = int(group["seed"].nunique())
            collapsed_rows.append(record)
    collapsed = pd.DataFrame(
        collapsed_rows, columns=PREDICTION_COLUMNS + ["seed_count"]
    )
    if not collapsed.empty:
        collapsed = collapsed.sort_values(
            [
                "interval_id",
                "universe_semantics",
                "fold_id",
                "prediction_date",
                "model_id",
                "security_id",
            ],
            kind="stable",
        ).reset_index(drop=True)
    failures = pd.DataFrame(execution_failures)
    return seed_level, collapsed, failures


def validate_prediction_cells(
    inventory: pd.DataFrame,
    fold_calendar: pd.DataFrame,
    features: pd.DataFrame,
    collapsed: pd.DataFrame,
    execution_failures: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows: list[dict[str, Any]] = []
    feature_keys = set(
        map(
            tuple,
            features[
                [
                    "interval_id",
                    "universe_semantics",
                    "security_id",
                    "prediction_date",
                ]
            ].itertuples(index=False, name=None),
        )
    )
    for interval in inventory.to_dict("records"):
        interval_id = str(interval["interval_id"])
        scheduled_folds = int(
            fold_calendar[
                fold_calendar["interval_id"].astype(str).eq(interval_id)
                & fold_calendar["valid_fold"].astype(bool)
            ]["fold_id"].nunique()
        )
        for semantics in ARM_SEMANTICS.values():
            for model_id in MODEL_SPECS:
                frame = collapsed[
                    collapsed["interval_id"].astype(str).eq(interval_id)
                    & collapsed["universe_semantics"].astype(str).eq(semantics)
                    & collapsed["model_id"].astype(str).eq(model_id)
                ].copy()
                reasons: list[str] = []
                folds = int(frame["fold_id"].nunique()) if not frame.empty else 0
                dates = (
                    int(frame["prediction_date"].nunique())
                    if not frame.empty
                    else 0
                )
                rows = len(frame)
                classes = (
                    int(frame["actual_class"].nunique())
                    if not frame.empty
                    else 0
                )
                if scheduled_folds < MIN_VALID_FOLDS:
                    reasons.append(
                        f"valid rolling folds {scheduled_folds} < {MIN_VALID_FOLDS}"
                    )
                if folds < MIN_VALID_FOLDS:
                    reasons.append(
                        f"prediction folds {folds} < {MIN_VALID_FOLDS}"
                    )
                if dates < MIN_OOS_DATES:
                    reasons.append(f"OOS dates {dates} < {MIN_OOS_DATES}")
                if rows < MIN_OOS_ROWS:
                    reasons.append(f"OOS rows {rows} < {MIN_OOS_ROWS}")
                if classes < 2:
                    reasons.append("both target classes not represented in test")
                if not frame.empty and (
                    (frame["training_class_0"] <= 0).any()
                    or (frame["training_class_1"] <= 0).any()
                ):
                    reasons.append("both target classes not represented in train")
                duplicate = (
                    frame.duplicated(
                        [
                            "interval_id",
                            "universe_semantics",
                            "fold_id",
                            "security_id",
                            "prediction_date",
                            "model_id",
                        ]
                    ).any()
                    if not frame.empty
                    else False
                )
                if duplicate:
                    reasons.append("duplicate collapsed prediction keys")
                if not frame.empty:
                    probability = frame["prediction_probability"].to_numpy(
                        dtype=float
                    )
                    if (
                        not np.isfinite(probability).all()
                        or (probability < 0).any()
                        or (probability > 1).any()
                    ):
                        reasons.append("invalid prediction probability")
                    if not (
                        pd.to_datetime(frame["target_date"])
                        > pd.to_datetime(frame["prediction_date"])
                    ).all():
                        reasons.append("target chronology failure")
                    if not (
                        pd.to_datetime(frame["label_maturity_cutoff"])
                        < pd.to_datetime(frame["prediction_date"])
                    ).all():
                        reasons.append("label maturity leakage")
                    prediction_keys = set(
                        map(
                            tuple,
                            frame[
                                [
                                    "interval_id",
                                    "universe_semantics",
                                    "security_id",
                                    "prediction_date",
                                ]
                            ].itertuples(index=False, name=None),
                        )
                    )
                    if not prediction_keys <= feature_keys:
                        reasons.append("prediction row missing valid feature chronology")
                if not execution_failures.empty:
                    failures = execution_failures[
                        execution_failures["interval_id"].astype(str).eq(
                            interval_id
                        )
                        & execution_failures["universe_semantics"]
                        .astype(str)
                        .eq(semantics)
                        & execution_failures["model_id"].astype(str).eq(model_id)
                    ]
                    if not failures.empty:
                        reasons.append(
                            "model execution failure: "
                            + ";".join(
                                sorted(set(failures["reason"].astype(str)))
                            )
                        )
                valid = not reasons
                all_rows.append(
                    {
                        "cell_id": stable_id(
                            "CELL", interval_id, semantics, model_id
                        ),
                        "interval_id": interval_id,
                        "market": interval["market"],
                        "index_id": interval["index_id"],
                        "universe_semantics": semantics,
                        "model_id": model_id,
                        "scheduled_valid_folds": scheduled_folds,
                        "prediction_folds": folds,
                        "oos_trading_dates": dates,
                        "oos_security_date_predictions": rows,
                        "target_class_count": classes,
                        "valid": valid,
                        "blocking_reasons": " | ".join(reasons) or None,
                        "result_classification": (
                            None if valid else "INSUFFICIENT_INFORMATION"
                        ),
                    }
                )
    cells = pd.DataFrame(all_rows)
    invalid = cells[~cells["valid"].astype(bool)].copy()
    path = (
        PIT_ROOT
        / "reports"
        / "pit_multimarket_invalid_cells_v6.csv"
    )
    write_csv(path, invalid.to_dict("records"), list(invalid.columns))
    write_parquet(
        PIT_ROOT / "qa" / "pit_multimarket_all_cells_v6.parquet",
        cells,
    )
    return cells, invalid


def write_prediction_artifacts(
    seed_level: pd.DataFrame,
    collapsed: pd.DataFrame,
) -> None:
    write_parquet(
        PIT_ROOT
        / "processed"
        / "pit_multimarket_predictions_seed_level_v6.parquet",
        seed_level,
        PREDICTION_COLUMNS,
    )
    write_parquet(
        PIT_ROOT
        / "processed"
        / "pit_multimarket_predictions_seed_collapsed_v6.parquet",
        collapsed,
        PREDICTION_COLUMNS + ["seed_count"],
    )


def run_model_phase() -> dict[str, int]:
    """Run only the frozen prediction phase from already-written V6 panels."""
    if not CONFIG_PATH.is_file():
        raise RuntimeError("freeze the V6 configuration before model fitting")
    inventory = _ready_inventory()
    features = pd.read_parquet(
        PIT_ROOT / "processed" / "pit_multimarket_feature_panel_v6.parquet"
    )
    folds = pd.read_parquet(
        PIT_ROOT / "processed" / "pit_multimarket_fold_calendar_v6.parquet"
    )
    seed_level, collapsed, failures = run_oos_models(features, folds)
    write_prediction_artifacts(seed_level, collapsed)
    cells, invalid = validate_prediction_cells(
        inventory, folds, features, collapsed, failures
    )
    return {
        "seed_level_predictions": len(seed_level),
        "collapsed_predictions": len(collapsed),
        "valid_cells": int(cells["valid"].astype(bool).sum()),
        "invalid_cells": len(invalid),
        "execution_failures": len(failures),
    }


def run_data_phase() -> dict[str, int]:
    """Construct all interval-bounded V6 observation, feature, and fold data."""
    if not CONFIG_PATH.is_file():
        raise RuntimeError("freeze the V6 configuration before data execution")
    inventory = _ready_inventory()
    members = build_arm_membership_dates(inventory)
    observations, rejections = construct_observations(inventory, members)
    observations, features, rejections = construct_features(
        observations, rejections, members
    )
    write_observation_feature_artifacts(
        observations, features, rejections
    )
    folds = build_fold_calendar(inventory, members)
    return {
        "observation_rows": len(observations),
        "feature_rows": len(features),
        "rejection_rows": len(rejections),
        "fold_rows": len(folds),
        "valid_folds": int(folds["valid_fold"].astype(bool).sum()),
    }


def run_full_benchmark() -> None:
    """Execute the frozen V6 data, model, analytics, and validation workflow."""
    run_preflight(freeze=True)
    run_data_phase()
    run_model_phase()
    from scripts.point_in_time.pit_multimarket_analytics_v6 import (
        final_console_text,
        run_analytics,
    )

    run_analytics()
    print(final_console_text())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=(
            "preflight",
            "freeze",
            "data",
            "models",
            "analytics",
            "console",
            "full",
        ),
        default="full",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "preflight":
        result = run_preflight(freeze=False)
    elif args.phase == "freeze":
        result = run_preflight(freeze=True)
    elif args.phase == "models":
        result = run_model_phase()
        print(json.dumps(result, sort_keys=True))
        return 0
    elif args.phase == "data":
        result = run_data_phase()
        print(json.dumps(result, sort_keys=True))
        return 0
    elif args.phase == "analytics":
        from scripts.point_in_time.pit_multimarket_analytics_v6 import (
            run_analytics,
        )

        result = run_analytics()
        print(json.dumps(result, sort_keys=True))
        return 0
    elif args.phase == "console":
        from scripts.point_in_time.pit_multimarket_analytics_v6 import (
            final_console_text,
        )

        print(final_console_text())
        return 0
    else:
        run_full_benchmark()
        return 0
    print(
        f"V6 preflight: PASS; eligible intervals={result['eligible_rows']}; "
        f"by index={result['eligible_by_index']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
