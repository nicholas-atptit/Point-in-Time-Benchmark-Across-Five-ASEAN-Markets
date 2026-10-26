#!/usr/bin/env python3
"""Five-market ASEAN PIT continuous rolling-origin benchmark V7.

V7 is additive.  It never rewrites V6.  It reads the frozen V5 membership and
price surfaces, recovers repository-local PSEI raw OHLC where validation
supports that use, preserves unresolved membership windows as explicit gaps,
and evaluates data-ready intervals with continuous prior market history.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
REPORTS = PIT_ROOT / "reports"
PROCESSED = PIT_ROOT / "processed"
QA = PIT_ROOT / "qa"
MANIFESTS = PIT_ROOT / "manifests"
CONFIG_ROOT = PIT_ROOT / "config"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_portability_v6 import validate_portable_record  # noqa: E402
from scripts.point_in_time import pit_multimarket_analytics_v6 as analytics  # noqa: E402
from scripts.point_in_time import pit_multimarket_forecasting_v6 as v6  # noqa: E402

EXPERIMENT_ID = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V7"
STUDY_LABEL = EXPERIMENT_ID
EXPERIMENT_VERSION = "v7"
PIPELINE_VERSION = "7.0.0"
RUN_DATE = "2026-08-03"
DATA_VERSION = "pit-data-v7-repository-recovery"
MEMBERSHIP_VERSION = "pit-membership-v7-evidence-reset"
MANIFEST_VERSION = "manifest-portability-v7.0.0"
V6_EXPERIMENT_ID = "SELECTED_ASEAN_PIT_PARTIAL_INTERVAL_BENCHMARK_V6"
V6_DATA_VERSION = "pit-data-v5"
V6_MEMBERSHIP_VERSION = "pit-membership-v5"
V6_MANIFEST_VERSION = "manifest-portability-v6.0.0"

MARKETS = {
    "IDX30": "Indonesia",
    "SET50": "Thailand",
    "VN30": "Vietnam",
    "PSEI": "Philippines",
    "FBM_KLCI": "Malaysia",
}
INDEX_BY_MARKET = {market: index_id for index_id, market in MARKETS.items()}
PIT = v6.PIT_SEMANTICS
CURRENT = v6.CURRENT_SEMANTICS
INTERSECTION = v6.INTERSECTION_SEMANTICS
ARM_SEMANTICS = {
    "pit": PIT,
    "current": CURRENT,
    "intersection": INTERSECTION,
}
FEATURE_COLUMNS = list(v6.FEATURE_COLUMNS)
FEATURE_LOOKBACK = v6.FEATURE_LOOKBACK
MODEL_SPECS = v6.MODEL_SPECS
SEEDS = list(v6.SEEDS)
INITIAL_TRAINING_DATES = v6.INITIAL_TRAINING_DATES
CALIBRATION_DATES = v6.CALIBRATION_DATES
TEST_BLOCK_DATES = v6.TEST_BLOCK_DATES
STEP_DATES = v6.STEP_DATES
MIN_VALID_FOLDS = v6.MIN_VALID_FOLDS
MIN_OOS_DATES = v6.MIN_OOS_DATES
MIN_OOS_ROWS = v6.MIN_OOS_ROWS
MAX_TARGET_GAP_DAYS = v6.MAX_TARGET_GAP_DAYS
PRICE_COVERAGE_THRESHOLD = v6.PRICE_COVERAGE_THRESHOLD
PREDICTION_THRESHOLD = v6.PREDICTION_THRESHOLD
BOOTSTRAP_SAMPLES = v6.BOOTSTRAP_SAMPLES
BOOTSTRAP_SEED = v6.BOOTSTRAP_SEED
PRIMARY_BLOCK_LENGTH = v6.PRIMARY_BLOCK_LENGTH
ROBUSTNESS_BLOCK_LENGTHS = list(v6.ROBUSTNESS_BLOCK_LENGTHS)

CONFIG_PATH = CONFIG_ROOT / "pit_multimarket_forecasting_v7.yaml"
OBSERVATION_PATH = PROCESSED / "pit_multimarket_observation_panel_v7.parquet"
FEATURE_PATH = PROCESSED / "pit_multimarket_feature_panel_v7.parquet"
FOLD_PATH = PROCESSED / "pit_multimarket_fold_calendar_v7.parquet"
SEED_PATH = PROCESSED / "pit_multimarket_predictions_seed_level_v7.parquet"
COLLAPSED_PATH = PROCESSED / "pit_multimarket_predictions_seed_collapsed_v7.parquet"
RECOVERED_RAW_PATH = PROCESSED / "pit_historical_raw_ohlc_recovered_v7.parquet"
MEMBERSHIP_PATH = PROCESSED / "pit_membership_intervals_v7.parquet"
MEMBERSHIP_DAILY_PATH = PROCESSED / "pit_membership_daily_all_arms_v7.parquet"
REJECTION_PATH = QA / "pit_multimarket_observation_rejections_v7.parquet"
ALL_CELLS_PATH = QA / "pit_multimarket_all_cells_v7.parquet"
MODEL_CHECKPOINT_DIR = QA / "pit_v7_model_checkpoints"
PREFLIGHT_PATH = QA / "pit_v7_preflight_validation.json"
TEST_SUMMARY_PATH = QA / "pit_v7_test_summary.json"
ARTIFACT_VALIDATION_PATH = QA / "pit_v7_artifact_validation.json"
INVENTORY_PATH = REPORTS / "forecasting_ready_market_inventory_v7.csv"
INTERVAL_INVENTORY_PATH = REPORTS / "forecasting_ready_interval_inventory_v7.csv"
BLOCKER_CSV_PATH = REPORTS / "five_market_data_blocker_audit_v7.csv"
BLOCKER_MD_PATH = REPORTS / "five_market_data_blocker_audit_v7.md"
ACQUISITION_CSV_PATH = MANIFESTS / "missing_price_acquisition_manifest_v7.csv"
ACQUISITION_JSON_PATH = MANIFESTS / "missing_price_acquisition_manifest_v7.json"
MANIFEST_PATH = MANIFESTS / "pit_multimarket_forecasting_v7_manifest.json"
STATUS_JSON_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V7.json"
STATUS_MD_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V7.md"

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
FOLD_COLUMNS = [
    "market",
    "index_id",
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
    "calibration_dates",
    "test_dates",
    "continuous_market_history",
    "valid_fold",
    "invalid_reason",
]
PREDICTION_COLUMNS = list(v6.PREDICTION_COLUMNS)
ACQUISITION_COLUMNS = [
    "market",
    "index_id",
    "interval_id",
    "security",
    "required_symbol",
    "provider_symbol_candidates",
    "first_required_date",
    "last_required_date",
    "missing_date_range",
    "missing_field",
    "reason_required",
    "recommended_source",
    "blocks_forecasting_eligibility",
]


def as_date(value: Any) -> date | None:
    return v6.as_date(value)


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    return v6.stable_id(prefix, *parts, length=length)


def sha256_file(path: Path) -> str:
    return v6.sha256_file(path)


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
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_frame(
    path: Path,
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.reindex(columns=list(columns)) if columns is not None else frame
    if path.suffix.lower() == ".parquet":
        output.to_parquet(path, index=False, engine="pyarrow")
    else:
        output.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.lower().eq("true")


def validate_v6_immutability() -> dict[str, Any]:
    manifest_path = (
        PIT_ROOT / "manifests" / "pit_multimarket_forecasting_v6_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest["records"]:
        try:
            validate_portable_record(record, REPO_ROOT)
        except Exception as exc:
            failures.append(str(exc))
    if manifest.get("experiment_id") != V6_EXPERIMENT_ID:
        failures.append("V6 experiment identifier drift")
    if manifest.get("manifest_version") != V6_MANIFEST_VERSION:
        failures.append("V6 manifest version drift")
    locked_v4 = analytics.verify_locked_v4()
    if not locked_v4["all_checksums_valid"]:
        failures.extend(locked_v4["failures"])
    return {
        "v6_manifest_path": manifest_path.relative_to(REPO_ROOT).as_posix(),
        "v6_manifest_records": int(manifest["artifact_count"]),
        "v6_manifest_checksums_valid": not failures,
        "v4_locked_records": int(locked_v4["artifact_count"]),
        "v4_locked_checksums_valid": bool(locked_v4["all_checksums_valid"]),
        "failures": failures,
    }


def current_universe_frames() -> dict[str, pd.DataFrame]:
    frames = v6.v3.current_tickers(REPO_ROOT)
    result: dict[str, pd.DataFrame] = {}
    for index_id in MARKETS:
        frame = frames[index_id].copy()
        frame["local_symbol"] = frame["local_symbol"].astype(str).str.strip()
        result[index_id] = frame
    return result


def current_universe_tickers() -> dict[str, set[str]]:
    return {
        index_id: set(frame["local_symbol"].astype(str))
        for index_id, frame in current_universe_frames().items()
    }


def valid_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        numeric.notna().all(axis=1)
        & (numeric > 0).all(axis=1)
        & (numeric["high"] >= numeric[["open", "low", "close"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "high", "close"]].min(axis=1))
    )


def psei_evidence_tickers() -> list[str]:
    anchors = pd.read_csv(
        PIT_ROOT / "interim" / "philippines_membership_anchors_v2_draft.csv"
    )
    events = pd.read_csv(
        PIT_ROOT / "interim" / "philippines_membership_events_v2_draft.csv"
    )
    return sorted(
        set(anchors["ticker_as_published"].dropna().astype(str))
        | set(events["ticker_as_published"].dropna().astype(str))
    )


def recover_repository_raw_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append structurally valid PSEI archive rows without altering V5."""
    base = pd.read_parquet(
        PIT_ROOT / "processed" / "historical_raw_ohlc_v5.parquet"
    ).copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.date
    base["_source_priority"] = 0
    base["_recovery_status"] = "V5_FROZEN_INPUT"
    archive_root = (
        REPO_ROOT
        / "external"
        / "psei_open_sources"
        / "carlo01092_pse"
        / "json"
    )
    recovery_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for ticker in psei_evidence_tickers():
        path = archive_root / f"{ticker}.json"
        if not path.is_file():
            audit_rows.append(
                {
                    "market": "Philippines",
                    "index_id": "PSEI",
                    "ticker": ticker,
                    "source_path": path.relative_to(REPO_ROOT).as_posix(),
                    "source_rows": 0,
                    "accepted_rows": 0,
                    "duplicate_rows": 0,
                    "conflicting_duplicate_dates": 0,
                    "invalid_ohlc_rows": 0,
                    "status": "SOURCE_NOT_PRESENT",
                }
            )
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        lengths = [len(payload.get(key, [])) for key in ("t", "o", "h", "l", "c")]
        row_count = min(lengths) if lengths else 0
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    payload["t"][:row_count], unit="s", utc=True, errors="coerce"
                )
                .tz_convert(None)
                .date,
                "open": payload["o"][:row_count],
                "high": payload["h"][:row_count],
                "low": payload["l"][:row_count],
                "close": payload["c"][:row_count],
            }
        )
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
        duplicate_rows = int(frame.duplicated(["date"], keep=False).sum())
        conflicting_dates: set[date] = set()
        for on_date, group in frame[frame.duplicated("date", keep=False)].groupby(
            "date", sort=True
        ):
            if len(group[["open", "high", "low", "close"]].drop_duplicates()) > 1:
                conflicting_dates.add(as_date(on_date))
        frame = frame[~frame["date"].isin(conflicting_dates)].drop_duplicates(
            ["date", "open", "high", "low", "close"], keep="first"
        )
        valid = valid_ohlc_mask(frame)
        invalid_rows = int((~valid).sum())
        frame = frame[valid].copy()
        frame["market"] = "Philippines"
        frame["security_id"] = v6.security_id("PSEI", ticker)
        frame["ticker_used"] = ticker
        frame["exchange_suffix"] = ".PS"
        frame["provider"] = "repository_carlo01092_investagrams_archive"
        frame["price_semantics"] = "RAW_OHLC"
        frame["raw_file_path"] = path.relative_to(REPO_ROOT).as_posix()
        frame["retrieval_timestamp"] = "repository_archive_pre_v7"
        frame["checksum"] = sha256_file(path)
        frame["_source_priority"] = 1
        frame["_recovery_status"] = "RECOVERED_REPOSITORY_ARCHIVE"
        recovery_rows.append(frame[base.columns])
        audit_rows.append(
            {
                "market": "Philippines",
                "index_id": "PSEI",
                "ticker": ticker,
                "source_path": path.relative_to(REPO_ROOT).as_posix(),
                "source_rows": row_count,
                "accepted_rows": len(frame),
                "duplicate_rows": duplicate_rows,
                "conflicting_duplicate_dates": len(conflicting_dates),
                "invalid_ohlc_rows": invalid_rows,
                "status": (
                    "RECOVERED_VALID_ROWS"
                    if len(frame)
                    else "NO_STRUCTURALLY_VALID_ROWS"
                ),
            }
        )
    combined = pd.concat([base, *recovery_rows], ignore_index=True)
    combined = combined.sort_values(
        ["market", "security_id", "date", "_source_priority"], kind="stable"
    ).drop_duplicates(["market", "security_id", "date"], keep="first")
    valid = valid_ohlc_mask(combined)
    rejected = combined[~valid]
    if not rejected.empty:
        for market, group in rejected.groupby("market", sort=True):
            audit_rows.append(
                {
                    "market": market,
                    "index_id": INDEX_BY_MARKET.get(str(market)),
                    "ticker": "*",
                    "source_path": "research_data/point_in_time/processed/historical_raw_ohlc_v5.parquet",
                    "source_rows": len(group),
                    "accepted_rows": 0,
                    "duplicate_rows": 0,
                    "conflicting_duplicate_dates": 0,
                    "invalid_ohlc_rows": len(group),
                    "status": "INVALID_V5_ROWS_EXCLUDED_FROM_V7_MODEL_INPUT",
                }
            )
    combined = combined[valid].drop(
        columns=["_source_priority", "_recovery_status"], errors="ignore"
    )
    combined = combined.sort_values(
        ["market", "security_id", "date"], kind="stable"
    ).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    write_frame(RECOVERED_RAW_PATH, combined)
    write_frame(REPORTS / "psei_repository_raw_recovery_v7.csv", audit)
    return combined, audit


def _existing_membership_intervals() -> pd.DataFrame:
    membership = pd.read_parquet(
        PIT_ROOT / "processed" / "index_membership_v5.parquet"
    )
    eligibility = pd.read_parquet(
        PIT_ROOT / "qa" / "pit_interval_eligibility_v5.parquet"
    )
    keys = [
        "index_id",
        "market",
        "interval_id",
        "interval_start",
        "interval_end_exclusive",
        "eligible_for_forecasting",
        "price_coverage",
        "feature_lookback_available",
        "target_coverage_available",
        "high_severity_conflicts",
        "blocking_reasons",
    ]
    gates = eligibility[keys].copy()
    gates["interval_start"] = pd.to_datetime(gates["interval_start"]).dt.date
    gates["interval_end_exclusive"] = pd.to_datetime(
        gates["interval_end_exclusive"]
    ).dt.date
    membership["effective_from"] = pd.to_datetime(
        membership["effective_from"]
    ).dt.date
    membership["effective_to"] = pd.to_datetime(
        membership["effective_to"]
    ).dt.date
    merged = membership.merge(
        gates,
        left_on=["index_id", "market", "effective_from", "effective_to"],
        right_on=[
            "index_id",
            "market",
            "interval_start",
            "interval_end_exclusive",
        ],
        how="inner",
        validate="many_to_one",
    )
    return merged[
        [
            "market",
            "index_id",
            "interval_id",
            "interval_start",
            "interval_end_exclusive",
            "interval_chain_id",
            "security_id",
            "ticker_on_effective_date",
            "source_chain",
            "eligible_for_forecasting",
            "price_coverage",
            "feature_lookback_available",
            "target_coverage_available",
            "high_severity_conflicts",
            "blocking_reasons",
        ]
    ].rename(
        columns={
            "interval_chain_id": "membership_chain_id",
            "ticker_on_effective_date": "ticker_on_date",
        }
    )


def _apply_events(
    state: set[str],
    events: pd.DataFrame,
    *,
    effective_date: date | None = None,
    review_period: str | None = None,
) -> set[str]:
    selected = events.copy()
    if effective_date is not None:
        selected = selected[
            pd.to_datetime(selected["effective_date"], errors="coerce")
            .dt.date.eq(effective_date)
        ]
    if review_period is not None:
        selected = selected[selected["review_period"].astype(str).eq(review_period)]
    result = set(state)
    for row in selected.to_dict("records"):
        ticker = str(row["ticker_as_published"])
        event_type = str(row["event_type"]).lower()
        if event_type == "addition":
            result.add(ticker)
        elif event_type == "removal":
            result.discard(ticker)
        else:
            raise ValueError(f"unknown PSEI membership event {event_type}")
    return result


def build_psei_membership_intervals() -> pd.DataFrame:
    """Build two evidence-backed PSEI chains and preserve both conflict gaps."""
    anchors = pd.read_csv(
        PIT_ROOT / "interim" / "philippines_membership_anchors_v2_draft.csv"
    )
    events = pd.read_csv(
        PIT_ROOT / "interim" / "philippines_membership_events_v2_draft.csv"
    )
    sources = pd.read_csv(
        PIT_ROOT / "interim" / "philippines_source_inventory_v2_draft.csv"
    )
    state = set(anchors["ticker_as_published"].dropna().astype(str))
    if len(state) != 30:
        raise RuntimeError(f"PSEI anchor must contain 30 securities, found {len(state)}")
    # The unresolved 2016 date is never emitted.  The composition changes are
    # applied only to establish the later, independently retained 2017-H1 full
    # list reset.
    state = _apply_events(state, events, review_period="2016-H2")
    state = _apply_events(state, events, effective_date=date(2017, 3, 13))
    if len(state) != 30:
        raise RuntimeError("2017-H1 PSEI full-list reset does not contain 30 securities")
    accepted = sources[
        sources["source_status"].astype(str).eq("accepted")
        & sources["stated_effective_date"].notna()
        & sources["evidence_scope"].astype(str).str.contains(
            "constituent_list|replacement|direct_list", regex=True
        )
    ].copy()
    accepted["effective_date"] = pd.to_datetime(
        accepted["stated_effective_date"]
    ).dt.date
    first_boundaries = sorted(
        set(
            accepted.loc[
                accepted["effective_date"].between(
                    date(2017, 3, 13), date(2023, 8, 7)
                ),
                "effective_date",
            ]
        )
    )
    rows: list[dict[str, Any]] = []

    def emit_chain(
        boundaries: list[date],
        state_at_first: set[str],
        chain_label: str,
        final_end: date,
        state_already_applied_first: bool,
    ) -> set[str]:
        current_state = set(state_at_first)
        for position, start in enumerate(boundaries):
            if position > 0 or not state_already_applied_first:
                current_state = _apply_events(
                    current_state, events, effective_date=start
                )
            end = boundaries[position + 1] if position + 1 < len(boundaries) else final_end
            interval_id = stable_id("ELG7", "PSEI", start, end)
            source_ids = accepted.loc[
                accepted["effective_date"].eq(start), "source_id"
            ].astype(str)
            source_chain = ";".join(sorted(set(source_ids)))
            for ticker in sorted(current_state):
                rows.append(
                    {
                        "market": "Philippines",
                        "index_id": "PSEI",
                        "interval_id": interval_id,
                        "interval_start": start,
                        "interval_end_exclusive": end,
                        "membership_chain_id": chain_label,
                        "security_id": v6.security_id("PSEI", ticker),
                        "ticker_on_date": ticker,
                        "source_chain": source_chain,
                        "eligible_for_forecasting": False,
                        "price_coverage": np.nan,
                        "feature_lookback_available": False,
                        "target_coverage_available": False,
                        "high_severity_conflicts": 0,
                        "blocking_reasons": None,
                    }
                )
        return current_state

    state = emit_chain(
        first_boundaries,
        state,
        "PSEI-2017H1-TO-2023H2-OFFICIAL-FULL-LIST-RESET",
        date(2023, 9, 26),
        True,
    )
    # The unresolved split dates remain a gap.  The 2023-10-04 official direct
    # list confirms the post-event composition and permits a new chain.
    state = _apply_events(state, events, review_period="2023-EXTRA-01")
    state = _apply_events(state, events, effective_date=date(2023, 10, 4))
    if len(state) != 30:
        raise RuntimeError("2023-10-04 PSEI full-list reset does not contain 30 securities")
    second_boundaries = sorted(
        set(
            accepted.loc[
                accepted["effective_date"].between(
                    date(2023, 10, 4), date(2026, 2, 2)
                ),
                "effective_date",
            ]
        )
    )
    emit_chain(
        second_boundaries,
        state,
        "PSEI-2023EXTRA2-TO-2026H1-OFFICIAL-FULL-LIST-RESET",
        date(2026, 7, 1),
        True,
    )
    result = pd.DataFrame(rows)
    if result.empty or result.groupby("interval_id")["security_id"].nunique().ne(30).any():
        raise RuntimeError("every reconstructed PSEI interval must contain 30 securities")
    return result


def build_membership_intervals() -> pd.DataFrame:
    existing = _existing_membership_intervals()
    psei = build_psei_membership_intervals()
    result = pd.concat([existing, psei], ignore_index=True)
    result = result.sort_values(
        ["index_id", "interval_start", "security_id"], kind="stable"
    ).reset_index(drop=True)
    write_frame(MEMBERSHIP_PATH, result)
    return result


def build_pit_membership_daily(
    intervals: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    snapshots = pd.read_parquet(
        PIT_ROOT / "snapshots" / "pit_universe_daily_v5.parquet"
    )
    snapshots["date"] = pd.to_datetime(snapshots["date"]).dt.date
    existing = snapshots[
        [
            "market",
            "index_id",
            "interval_id",
            "security_id",
            "ticker_on_date",
            "date",
            "interval_chain_id",
        ]
    ].rename(
        columns={
            "date": "membership_date",
            "interval_chain_id": "membership_chain_id",
        }
    )
    existing["universe_semantics"] = PIT
    existing["membership_version"] = V6_MEMBERSHIP_VERSION
    psei_intervals = intervals[intervals["index_id"].eq("PSEI")]
    psei_dates = sorted(
        set(
            raw.loc[raw["market"].eq("Philippines"), "date"]
            .dropna()
            .map(as_date)
        )
    )
    psei_rows: list[dict[str, Any]] = []
    for (
        interval_id,
        start_value,
        end_value,
        chain_id,
    ), group in psei_intervals.groupby(
        [
            "interval_id",
            "interval_start",
            "interval_end_exclusive",
            "membership_chain_id",
        ],
        sort=True,
    ):
        start, end = as_date(start_value), as_date(end_value)
        dates = [value for value in psei_dates if start <= value < end]
        for on_date in dates:
            for row in group.to_dict("records"):
                psei_rows.append(
                    {
                        "market": "Philippines",
                        "index_id": "PSEI",
                        "interval_id": interval_id,
                        "security_id": row["security_id"],
                        "ticker_on_date": row["ticker_on_date"],
                        "membership_date": on_date,
                        "membership_chain_id": chain_id,
                        "universe_semantics": PIT,
                        "membership_version": MEMBERSHIP_VERSION,
                    }
                )
    psei = pd.DataFrame(psei_rows, columns=existing.columns)
    pit = pd.concat([existing, psei], ignore_index=True)
    pit = pit.drop_duplicates(
        ["interval_id", "security_id", "membership_date"], keep="first"
    )
    return pit.sort_values(
        ["index_id", "membership_date", "interval_id", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_all_arm_membership(pit: pd.DataFrame) -> pd.DataFrame:
    current = current_universe_tickers()
    current_rows: list[dict[str, Any]] = []
    intersection_rows: list[dict[str, Any]] = []
    for (interval_id, market, index_id), group in pit.groupby(
        ["interval_id", "market", "index_id"], sort=True
    ):
        current_tickers = current[str(index_id)]
        current_ids = {
            v6.security_id(str(index_id), ticker): ticker
            for ticker in current_tickers
        }
        pit_by_date = {
            as_date(on_date): date_group
            for on_date, date_group in group.groupby("membership_date", sort=True)
        }
        chain_id = str(group.iloc[0]["membership_chain_id"])
        for on_date, date_group in pit_by_date.items():
            pit_ids = {
                str(row["security_id"]): str(row["ticker_on_date"])
                for row in date_group.to_dict("records")
            }
            for security, ticker in sorted(current_ids.items()):
                current_rows.append(
                    {
                        "market": market,
                        "index_id": index_id,
                        "interval_id": interval_id,
                        "security_id": security,
                        "ticker_on_date": ticker,
                        "membership_date": on_date,
                        "membership_chain_id": (
                            f"{chain_id}+CURRENT_2026_CONFIG"
                        ),
                        "universe_semantics": CURRENT,
                        "membership_version": "current-universe-2026-config-v1",
                    }
                )
            for security in sorted(set(pit_ids).intersection(current_ids)):
                intersection_rows.append(
                    {
                        "market": market,
                        "index_id": index_id,
                        "interval_id": interval_id,
                        "security_id": security,
                        "ticker_on_date": pit_ids[security],
                        "membership_date": on_date,
                        "membership_chain_id": (
                            f"{chain_id}+CURRENT_2026_CONFIG+INTERSECTION"
                        ),
                        "universe_semantics": INTERSECTION,
                        "membership_version": (
                            f"{MEMBERSHIP_VERSION}+current-universe-2026-config-v1"
                        ),
                    }
                )
    columns = list(pit.columns)
    all_members = pd.concat(
        [
            pit,
            pd.DataFrame(current_rows, columns=columns),
            pd.DataFrame(intersection_rows, columns=columns),
        ],
        ignore_index=True,
    )
    if all_members.duplicated(
        ["interval_id", "universe_semantics", "security_id", "membership_date"]
    ).any():
        raise RuntimeError("duplicate V7 membership date key")
    write_frame(MEMBERSHIP_DAILY_PATH, all_members)
    return all_members


def _next_active_target_coverage(
    members: pd.DataFrame, raw: pd.DataFrame
) -> tuple[int, int]:
    required = len(members)
    raw_keys = set(
        map(
            tuple,
            raw[["market", "security_id", "date"]].itertuples(
                index=False, name=None
            ),
        )
    )
    covered = 0
    for (market, security), group in members.groupby(
        ["market", "security_id"], sort=False
    ):
        dates = sorted(set(group["membership_date"].map(as_date)))
        available = [
            on_date for on_date in dates if (market, security, on_date) in raw_keys
        ]
        for prediction, target in zip(available, available[1:]):
            if (target - prediction).days <= MAX_TARGET_GAP_DAYS:
                covered += 1
    return required, covered


def build_interval_inventory(
    intervals: pd.DataFrame, pit_daily: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_keys = set(
        map(
            tuple,
            raw[["market", "security_id", "date"]].itertuples(
                index=False, name=None
            ),
        )
    )
    raw_by_security = {
        (str(market), str(security)): sorted(set(group["date"].map(as_date)))
        for (market, security), group in raw.groupby(
            ["market", "security_id"], sort=False
        )
    }
    for (
        market,
        index_id,
        interval_id,
        start_value,
        end_value,
    ), group in intervals.groupby(
        [
            "market",
            "index_id",
            "interval_id",
            "interval_start",
            "interval_end_exclusive",
        ],
        sort=True,
    ):
        members = pit_daily[pit_daily["interval_id"].eq(interval_id)]
        required = len(members)
        covered = int(
            sum(
                (
                    str(row["market"]),
                    str(row["security_id"]),
                    as_date(row["membership_date"]),
                )
                in raw_keys
                for row in members.to_dict("records")
            )
        )
        feature_covered = 0
        for row in members.to_dict("records"):
            history = raw_by_security.get(
                (str(row["market"]), str(row["security_id"])), []
            )
            position = int(np.searchsorted(history, as_date(row["membership_date"])))
            if position >= FEATURE_LOOKBACK:
                feature_covered += 1
        _, target_covered = _next_active_target_coverage(members, raw)
        price_coverage = covered / required if required else 0.0
        feature_coverage = feature_covered / required if required else 0.0
        target_coverage = target_covered / required if required else 0.0
        source = group.iloc[0]
        if index_id == "PSEI":
            eligible = bool(
                required
                and price_coverage >= PRICE_COVERAGE_THRESHOLD
                and feature_coverage >= PRICE_COVERAGE_THRESHOLD
                and target_coverage >= PRICE_COVERAGE_THRESHOLD
            )
            reasons = []
            if price_coverage < PRICE_COVERAGE_THRESHOLD:
                reasons.append(
                    f"active-member-day price coverage {price_coverage:.6f} < 0.95"
                )
            if feature_coverage < PRICE_COVERAGE_THRESHOLD:
                reasons.append(
                    f"five-session feature coverage {feature_coverage:.6f} < 0.95"
                )
            if target_coverage < PRICE_COVERAGE_THRESHOLD:
                reasons.append(
                    f"next-session target coverage {target_coverage:.6f} < 0.95"
                )
            blocking = " | ".join(reasons) or None
            gate_source = "V7_RECOMPUTED_PSEI_EVIDENCE_RESET"
        else:
            eligible = bool(source["eligible_for_forecasting"])
            blocking = source["blocking_reasons"]
            gate_source = "V5_FROZEN_INTERVAL_GATE"
        rows.append(
            {
                "market": market,
                "index_id": index_id,
                "interval_id": interval_id,
                "interval_start": as_date(start_value),
                "interval_end_exclusive": as_date(end_value),
                "calendar_days": (
                    as_date(end_value) - as_date(start_value)
                ).days,
                "trading_dates": int(members["membership_date"].nunique()),
                "pit_security_count": int(group["security_id"].nunique()),
                "required_active_member_days": required,
                "covered_active_member_days": covered,
                "price_coverage": price_coverage,
                "feature_lookback_coverage": feature_coverage,
                "target_pair_coverage": target_coverage,
                "high_severity_conflicts": int(
                    source.get("high_severity_conflicts") or 0
                ),
                "eligible_for_forecasting": eligible,
                "gate_source": gate_source,
                "blocking_reasons": blocking,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["index_id", "interval_start"], kind="stable"
    )
    write_frame(INTERVAL_INVENTORY_PATH, result)
    return result


def construct_observations(
    members: pd.DataFrame, raw: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
    raw = raw.sort_values(
        ["market", "security_id", "date"], kind="stable"
    ).drop_duplicates(["market", "security_id", "date"], keep="first")
    target = raw[
        ["market", "security_id", "date", "open", "close", "ticker_used"]
    ].copy()
    target[["target_date", "target_open", "target_close", "target_ticker"]] = (
        target.groupby(["market", "security_id"], sort=False)[
            ["date", "open", "close", "ticker_used"]
        ].shift(-1)
    )
    prediction = raw[
        [
            "market",
            "security_id",
            "date",
            "ticker_used",
            "open",
            "high",
            "low",
            "close",
        ]
    ].merge(
        target[
            [
                "market",
                "security_id",
                "date",
                "target_date",
                "target_open",
                "target_close",
                "target_ticker",
            ]
        ],
        on=["market", "security_id", "date"],
        how="left",
        validate="one_to_one",
    )
    merged = members.merge(
        prediction,
        left_on=["market", "security_id", "membership_date"],
        right_on=["market", "security_id", "date"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["date"], errors="ignore")
    member_ticker = {
        (
            str(row["market"]),
            str(row["universe_semantics"]),
            str(row["security_id"]),
            as_date(row["membership_date"]),
        ): str(row["ticker_on_date"])
        for row in members.to_dict("records")
    }
    rows: list[dict[str, Any]] = []
    for row in merged.to_dict("records"):
        prediction_date = as_date(row["membership_date"])
        target_date = as_date(row.get("target_date"))
        target_membership_ticker = member_ticker.get(
            (
                str(row["market"]),
                str(row["universe_semantics"]),
                str(row["security_id"]),
                target_date,
            )
        )
        reason = None
        if pd.isna(row.get("open")):
            reason = "MISSING_PREDICTION_RAW_OHLC"
        elif target_date is None:
            reason = "NO_NEXT_VALID_SAME_SECURITY_OBSERVATION"
        elif (target_date - prediction_date).days > MAX_TARGET_GAP_DAYS:
            reason = "LONG_TARGET_GAP"
        elif target_membership_ticker is None:
            reason = "TARGET_OUTSIDE_ACTIVE_MEMBERSHIP"
        elif str(row.get("ticker_used")) != str(row["ticker_on_date"]):
            reason = "PREDICTION_TICKER_VALIDITY_BOUNDARY"
        elif str(row.get("target_ticker")) != target_membership_ticker:
            reason = "TARGET_TICKER_VALIDITY_BOUNDARY"
        target_return = np.nan
        target_direction = np.nan
        if reason is None:
            target_return = float(row["target_close"]) / float(row["target_open"]) - 1.0
            target_direction = int(target_return > 0.0)
        rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                "interval_id": row["interval_id"],
                "market": row["market"],
                "index_id": row["index_id"],
                "universe_semantics": row["universe_semantics"],
                "security_id": row["security_id"],
                "ticker_on_date": row["ticker_on_date"],
                "prediction_date": prediction_date,
                "target_date": target_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "target_open": row.get("target_open"),
                "target_close": row.get("target_close"),
                "target_return": target_return,
                "target_direction": target_direction,
                "membership_chain_id": row["membership_chain_id"],
                "membership_version": row["membership_version"],
                "price_version": DATA_VERSION,
                "valid_for_model": reason is None,
                "rejection_reason": reason,
            }
        )
    panel = pd.DataFrame(rows, columns=OBSERVATION_COLUMNS)
    valid = panel[panel["valid_for_model"].astype(bool)].copy()
    rejected = panel[~panel["valid_for_model"].astype(bool)].copy()
    key = [
        "interval_id",
        "universe_semantics",
        "security_id",
        "prediction_date",
    ]
    if valid.duplicated(key).any():
        raise RuntimeError("duplicate V7 observation keys")
    if not valid.empty and not (
        pd.to_datetime(valid["target_date"]) > pd.to_datetime(valid["prediction_date"])
    ).all():
        raise RuntimeError("V7 target chronology failure")
    return valid, rejected


def construct_features(
    observations: pd.DataFrame,
    rejections: pd.DataFrame,
    members: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from papers.paper_03_raw_ohlc_forecasting.src.features.raw_ohlc import (
        build_lagged_raw_features,
    )

    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.date
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
    for (market, semantics), member_group in members.groupby(
        ["market", "universe_semantics"], sort=True
    ):
        obs = observations[
            observations["market"].eq(market)
            & observations["universe_semantics"].eq(semantics)
        ].copy()
        if obs.empty:
            continue
        securities = set(member_group["security_id"].astype(str))
        full = raw[
            raw["market"].eq(market)
            & raw["security_id"].astype(str).isin(securities)
        ][
            ["market", "security_id", "date", "open", "high", "low", "close"]
        ].copy()
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
            full_featured, built_columns = build_lagged_raw_features(full_input)
        if list(built_columns) != FEATURE_COLUMNS:
            raise RuntimeError("locked raw-OHLC feature registry changed")
        full_featured["date"] = pd.to_datetime(full_featured["date"]).dt.date
        full_featured = full_featured.rename(
            columns={"symbol_canonical": "security_id"}
        )
        eligible_raw = member_group.merge(
            raw[
                ["market", "security_id", "date", "open", "high", "low", "close"]
            ],
            left_on=["market", "security_id", "membership_date"],
            right_on=["market", "security_id", "date"],
            how="inner",
            validate="many_to_one",
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
            eligible_featured, _ = build_lagged_raw_features(eligible_input)
        eligible_featured["date"] = pd.to_datetime(
            eligible_featured["date"]
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
        merged = obs.merge(
            feature_lookup,
            left_on=["security_id", "prediction_date"],
            right_on=["security_id", "date"],
            how="left",
            validate="one_to_one",
        ).drop(columns=["date"], errors="ignore")
        values = merged[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        finite = pd.Series(
            np.isfinite(values.to_numpy()).all(axis=1), index=merged.index
        )
        merged["feature_version"] = v6.FEATURE_VERSION
        merged["feature_registry_hash"] = registry_hash
        merged["feature_lookback"] = FEATURE_LOOKBACK
        merged["feature_available_date"] = merged["prediction_date"]
        merged["null_reason"] = np.where(finite, None, "MISSING_FEATURES")
        merged.loc[~finite, "valid_for_model"] = False
        merged.loc[~finite, "rejection_reason"] = "MISSING_FEATURES"
        observation_updates.append(merged[OBSERVATION_COLUMNS].copy())
        feature_frames.append(merged.loc[finite].copy())
        feature_rejections.append(merged.loc[~finite, OBSERVATION_COLUMNS].copy())
    features = (
        pd.concat(feature_frames, ignore_index=True)
        if feature_frames
        else pd.DataFrame(columns=OBSERVATION_COLUMNS + FEATURE_COLUMNS)
    )
    observations_updated = (
        pd.concat(observation_updates, ignore_index=True)
        if observation_updates
        else observations
    )
    all_rejections = pd.concat(
        [rejections, *feature_rejections], ignore_index=True
    )
    return observations_updated, features, all_rejections


def build_fold_calendar(
    inventory: pd.DataFrame, pit_members: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    market_dates = {
        index_id: sorted(set(group["membership_date"].map(as_date)))
        for index_id, group in pit_members.groupby("index_id", sort=True)
    }
    for interval in inventory.sort_values(
        ["index_id", "interval_start"], kind="stable"
    ).to_dict("records"):
        index_id = str(interval["index_id"])
        interval_id = str(interval["interval_id"])
        dates = sorted(
            set(
                pit_members.loc[
                    pit_members["interval_id"].astype(str).eq(interval_id),
                    "membership_date",
                ].map(as_date)
            )
        )
        if not bool(interval["eligible_for_forecasting"]):
            rows.append(
                {
                    "market": interval["market"],
                    "index_id": index_id,
                    "interval_id": interval_id,
                    "fold_id": stable_id("FOLD7", interval_id, "DATA_BLOCKED", length=16),
                    "training_start": None,
                    "training_end": None,
                    "calibration_start": None,
                    "calibration_end": None,
                    "test_start": None,
                    "test_end": None,
                    "label_maturity_cutoff": None,
                    "training_dates": 0,
                    "calibration_dates": 0,
                    "test_dates": 0,
                    "continuous_market_history": True,
                    "valid_fold": False,
                    "invalid_reason": (
                        "INTERVAL_DATA_GATE_FAILED: "
                        + str(interval.get("blocking_reasons") or "unspecified")
                    ),
                }
            )
            continue
        valid_count = 0
        for offset in range(0, len(dates), STEP_DATES):
            test = dates[offset : offset + TEST_BLOCK_DATES]
            if len(test) < TEST_BLOCK_DATES:
                continue
            history = [value for value in market_dates[index_id] if value < test[0]]
            calibration = history[-CALIBRATION_DATES:]
            training = history[:-CALIBRATION_DATES]
            valid = (
                len(calibration) == CALIBRATION_DATES
                and len(training) >= INITIAL_TRAINING_DATES
            )
            rows.append(
                {
                    "market": interval["market"],
                    "index_id": index_id,
                    "interval_id": interval_id,
                    "fold_id": stable_id(
                        "FOLD7", index_id, interval_id, offset, length=16
                    ),
                    "training_start": training[0] if training else None,
                    "training_end": training[-1] if training else None,
                    "calibration_start": calibration[0] if calibration else None,
                    "calibration_end": calibration[-1] if calibration else None,
                    "test_start": test[0],
                    "test_end": test[-1],
                    "label_maturity_cutoff": training[-1] if training else None,
                    "training_dates": len(training),
                    "calibration_dates": len(calibration),
                    "test_dates": len(test),
                    "continuous_market_history": True,
                    "valid_fold": valid,
                    "invalid_reason": (
                        None
                        if valid
                        else (
                            f"continuous history has {len(training)} training "
                            f"and {len(calibration)} calibration dates"
                        )
                    ),
                }
            )
            valid_count += int(valid)
        if valid_count == 0 and not any(
            row["interval_id"] == interval_id for row in rows
        ):
            rows.append(
                {
                    "market": interval["market"],
                    "index_id": index_id,
                    "interval_id": interval_id,
                    "fold_id": stable_id("FOLD7", interval_id, "SHORT", length=16),
                    "training_start": None,
                    "training_end": None,
                    "calibration_start": None,
                    "calibration_end": None,
                    "test_start": None,
                    "test_end": None,
                    "label_maturity_cutoff": None,
                    "training_dates": 0,
                    "calibration_dates": 0,
                    "test_dates": 0,
                    "continuous_market_history": True,
                    "valid_fold": False,
                    "invalid_reason": "NO_COMPLETE_20_DATE_TEST_BLOCK",
                }
            )
    # FBM_KLCI has no validated membership interval and therefore no test dates.
    rows.append(
        {
            "market": "Malaysia",
            "index_id": "FBM_KLCI",
            "interval_id": "BLOCKED-FBM_KLCI-NO-PIT-INTERVAL",
            "fold_id": "FOLD7-FBM-KLCI-DATA-BLOCKED",
            "training_start": None,
            "training_end": None,
            "calibration_start": None,
            "calibration_end": None,
            "test_start": None,
            "test_end": None,
            "label_maturity_cutoff": None,
            "training_dates": 0,
            "calibration_dates": 0,
            "test_dates": 0,
            "continuous_market_history": True,
            "valid_fold": False,
            "invalid_reason": "NO_VALIDATED_FBM_KLCI_PIT_MEMBERSHIP_INTERVAL",
        }
    )
    result = pd.DataFrame(rows, columns=FOLD_COLUMNS)
    valid = result[result["valid_fold"].astype(bool)]
    for index_id, group in valid.groupby("index_id", sort=True):
        ordered = group.sort_values("test_start")
        starts = pd.to_datetime(ordered["test_start"]).to_numpy()
        ends = pd.to_datetime(ordered["test_end"]).to_numpy()
        if len(group) > 1 and not (starts[1:] > ends[:-1]).all():
            raise RuntimeError(f"overlapping V7 test blocks for {index_id}")
    write_frame(FOLD_PATH, result, FOLD_COLUMNS)
    return result


def run_oos_models(
    features: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = features[features["valid_for_model"].astype(bool)].copy()
    data["prediction_date"] = pd.to_datetime(data["prediction_date"]).dt.date
    data["target_date"] = pd.to_datetime(data["target_date"]).dt.date
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    checkpoint_signature = hashlib.sha256(
        json.dumps(
            {
                "pipeline_version": PIPELINE_VERSION,
                "model_specs": MODEL_SPECS,
                "seeds": SEEDS,
                "feature_columns": FEATURE_COLUMNS,
                "prediction_threshold": PREDICTION_THRESHOLD,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    MODEL_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds[folds["valid_fold"].astype(bool)].sort_values(
        ["index_id", "test_start"], kind="stable"
    ).to_dict("records"):
        index_id = str(fold["index_id"])
        interval_id = str(fold["interval_id"])
        fold_id = str(fold["fold_id"])
        checkpoint_path = MODEL_CHECKPOINT_DIR / f"{fold_id}.parquet"
        marker_path = MODEL_CHECKPOINT_DIR / f"{fold_id}.complete.json"
        if checkpoint_path.is_file() and marker_path.is_file():
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("checkpoint_signature") == checkpoint_signature:
                checkpoint = pd.read_parquet(checkpoint_path)
                if len(checkpoint) == int(marker.get("prediction_rows", -1)):
                    predictions.extend(checkpoint.to_dict("records"))
                    failures.extend(marker.get("failures", []))
                    continue
        fold_predictions: list[dict[str, Any]] = []
        fold_failures: list[dict[str, Any]] = []
        calibration_start = as_date(fold["calibration_start"])
        test_start, test_end = as_date(fold["test_start"]), as_date(fold["test_end"])
        for semantics in ARM_SEMANTICS.values():
            market_data = data[
                data["index_id"].astype(str).eq(index_id)
                & data["universe_semantics"].astype(str).eq(semantics)
            ]
            training = market_data[
                (market_data["prediction_date"] < calibration_start)
                & (market_data["target_date"] < calibration_start)
            ].copy()
            test = market_data[
                market_data["interval_id"].astype(str).eq(interval_id)
                & (market_data["prediction_date"] >= test_start)
                & (market_data["prediction_date"] <= test_end)
            ].copy()
            y_train = training["target_direction"].astype(int)
            y_test = test["target_direction"].astype(int)
            base_failure = None
            if training.empty:
                base_failure = "NO_MATURED_CONTINUOUS_TRAINING_ROWS"
            elif test.empty:
                base_failure = "NO_TEST_ROWS"
            elif y_train.nunique() < 2:
                base_failure = "TRAINING_ONE_CLASS"
            elif y_test.nunique() < 2:
                base_failure = "TEST_ONE_CLASS"
            if base_failure:
                for model_id in MODEL_SPECS:
                    fold_failures.append(
                        {
                            "market": fold["market"],
                            "index_id": index_id,
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
            test_records = test.to_dict("records")
            training_start = min(training["prediction_date"])
            training_end = max(training["prediction_date"])
            label_maturity_cutoff = max(training["target_date"])
            for model_id, spec in MODEL_SPECS.items():
                def fit_seed(seed: int) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
                    try:
                        if model_id == "historical_class_probability":
                            probability = np.full(
                                len(test), float(y_train.mean()), dtype=float
                            )
                        else:
                            estimator = v6._build_model(model_id, seed)
                            if "n_jobs" in estimator.get_params(deep=False):
                                # Seeds are parallelized independently below;
                                # keep each estimator single-threaded to avoid
                                # nested oversubscription. Statistical
                                # parameters in the V6 registry are unchanged.
                                estimator.set_params(n_jobs=1)
                            estimator.fit(X_train, y_train)
                            probability = v6._predict_probability(
                                model_id, estimator, X_test
                            )
                        probability = np.asarray(probability, dtype=float)
                        if (
                            not np.isfinite(probability).all()
                            or (probability < 0).any()
                            or (probability > 1).any()
                        ):
                            raise ValueError("PROBABILITY_OUT_OF_BOUNDS")
                        rows = []
                        for base, value in zip(test_records, probability):
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
                                    "model_version": f"v7-{model_id}-locked-v6-registry",
                                    "seed": seed,
                                    "training_start": training_start,
                                    "training_end": training_end,
                                    "training_row_count": len(training),
                                    "training_class_0": int(train_counts.get(0, 0)),
                                    "training_class_1": int(train_counts.get(1, 0)),
                                    "label_maturity_cutoff": label_maturity_cutoff,
                                    "prediction_probability": float(value),
                                    "predicted_class": int(
                                        float(value) >= PREDICTION_THRESHOLD
                                    ),
                                    "actual_class": int(base["target_direction"]),
                                    "feature_version": base["feature_version"],
                                    "membership_version": base["membership_version"],
                                    "price_version": base["price_version"],
                                    "is_valid_prediction": True,
                                }
                            )
                        return rows, None
                    except Exception as exc:
                        return [], (
                            {
                                "market": fold["market"],
                                "index_id": index_id,
                                "interval_id": interval_id,
                                "universe_semantics": semantics,
                                "fold_id": fold["fold_id"],
                                "model_id": model_id,
                                "seed": seed,
                                "reason": f"{type(exc).__name__}:{exc}",
                            }
                        )
                with ThreadPoolExecutor(max_workers=len(SEEDS)) as executor:
                    for seed_rows, seed_failure in executor.map(fit_seed, SEEDS):
                        fold_predictions.extend(seed_rows)
                        if seed_failure is not None:
                            fold_failures.append(seed_failure)
        checkpoint = pd.DataFrame(fold_predictions, columns=PREDICTION_COLUMNS)
        write_frame(checkpoint_path, checkpoint, PREDICTION_COLUMNS)
        write_json(
            marker_path,
            {
                "checkpoint_signature": checkpoint_signature,
                "experiment_id": EXPERIMENT_ID,
                "fold_id": fold_id,
                "prediction_rows": len(checkpoint),
                "failures": fold_failures,
            },
        )
        predictions.extend(fold_predictions)
        failures.extend(fold_failures)
    seed_level = pd.DataFrame(predictions, columns=PREDICTION_COLUMNS)
    collapse_keys = [
        column
        for column in PREDICTION_COLUMNS
        if column not in {"seed", "prediction_probability", "predicted_class"}
    ]
    collapsed_rows: list[dict[str, Any]] = []
    if not seed_level.empty:
        for values, group in seed_level.groupby(
            collapse_keys, dropna=False, sort=True
        ):
            record = dict(zip(collapse_keys, values))
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
    return seed_level, collapsed, pd.DataFrame(failures)


def validate_market_cells(
    folds: pd.DataFrame,
    collapsed: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index_id, market in MARKETS.items():
        valid_folds = int(
            folds[
                folds["index_id"].eq(index_id) & folds["valid_fold"].astype(bool)
            ]["fold_id"].nunique()
        )
        eligible_intervals = int(
            folds[
                folds["index_id"].eq(index_id) & folds["valid_fold"].astype(bool)
            ]["interval_id"].nunique()
        )
        for semantics in ARM_SEMANTICS.values():
            for model_id in MODEL_SPECS:
                frame = collapsed[
                    collapsed["index_id"].astype(str).eq(index_id)
                    & collapsed["universe_semantics"].astype(str).eq(semantics)
                    & collapsed["model_id"].astype(str).eq(model_id)
                ].copy()
                prediction_folds = int(frame["fold_id"].nunique()) if len(frame) else 0
                dates = int(frame["prediction_date"].nunique()) if len(frame) else 0
                prediction_rows = len(frame)
                classes = int(frame["actual_class"].nunique()) if len(frame) else 0
                reasons: list[str] = []
                if index_id == "FBM_KLCI":
                    reasons.append("no validated FBM_KLCI PIT membership interval")
                if valid_folds < MIN_VALID_FOLDS:
                    reasons.append(
                        f"valid market folds {valid_folds} < {MIN_VALID_FOLDS}"
                    )
                if prediction_folds < MIN_VALID_FOLDS:
                    reasons.append(
                        f"prediction folds {prediction_folds} < {MIN_VALID_FOLDS}"
                    )
                if dates < MIN_OOS_DATES:
                    reasons.append(f"OOS dates {dates} < {MIN_OOS_DATES}")
                if prediction_rows < MIN_OOS_ROWS:
                    reasons.append(
                        f"OOS security-date predictions {prediction_rows} < {MIN_OOS_ROWS}"
                    )
                if classes < 2:
                    reasons.append("both target classes not represented in test")
                if len(frame) and (
                    (frame["training_class_0"] <= 0).any()
                    or (frame["training_class_1"] <= 0).any()
                ):
                    reasons.append("both target classes not represented in train")
                if len(frame) and frame.duplicated(
                    [
                        "interval_id",
                        "universe_semantics",
                        "fold_id",
                        "security_id",
                        "prediction_date",
                        "model_id",
                    ]
                ).any():
                    reasons.append("duplicate collapsed prediction key")
                if len(frame) and not (
                    pd.to_datetime(frame["label_maturity_cutoff"])
                    < pd.to_datetime(frame["prediction_date"])
                ).all():
                    reasons.append("label maturity leakage")
                if not failures.empty:
                    selected = failures[
                        failures["index_id"].astype(str).eq(index_id)
                        & failures["universe_semantics"].astype(str).eq(semantics)
                        & failures["model_id"].astype(str).eq(model_id)
                    ]
                    if not selected.empty:
                        reasons.append(
                            "model execution failure: "
                            + ";".join(sorted(set(selected["reason"].astype(str))))
                        )
                rows.append(
                    {
                        "cell_id": stable_id("CELL7", index_id, semantics, model_id),
                        "market": market,
                        "index_id": index_id,
                        "universe_semantics": semantics,
                        "model_id": model_id,
                        "eligible_intervals": eligible_intervals,
                        "scheduled_valid_folds": valid_folds,
                        "prediction_folds": prediction_folds,
                        "oos_trading_dates": dates,
                        "oos_security_date_predictions": prediction_rows,
                        "target_class_count": classes,
                        "valid": not reasons,
                        "blocking_reasons": " | ".join(reasons) or None,
                        "result_classification": (
                            None if not reasons else "INSUFFICIENT_INFORMATION"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def repository_candidate_file_counts() -> tuple[dict[str, int], list[str]]:
    allowed = {
        ".csv",
        ".gz",
        ".parquet",
        ".feather",
        ".json",
        ".xlsx",
        ".xls",
        ".pkl",
        ".pickle",
    }
    tokens = {
        "IDX30": ("indonesia", "idx30"),
        "SET50": ("thailand", "set50"),
        "VN30": ("vietnam", "vn30"),
        "PSEI": ("philippines", "psei"),
        "FBM_KLCI": ("malaysia", "klci", "fbm"),
    }
    counts = {index_id: 0 for index_id in MARKETS}
    inaccessible: list[str] = []
    for path in REPO_ROOT.rglob("*"):
        try:
            is_file = path.is_file()
        except OSError as exc:
            inaccessible.append(
                f"{path.relative_to(REPO_ROOT).as_posix()}: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if not is_file:
            continue
        suffixes = "".join(path.suffixes).lower()
        if path.suffix.lower() not in allowed and suffixes != ".csv.gz":
            continue
        text = path.relative_to(REPO_ROOT).as_posix().lower()
        for index_id, market_tokens in tokens.items():
            if any(token in text for token in market_tokens):
                counts[index_id] += 1
    return counts, inaccessible


def build_acquisition_manifest(
    intervals: pd.DataFrame,
    pit_daily: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    raw_keys = set(
        map(
            tuple,
            raw[["market", "security_id", "date"]].itertuples(
                index=False, name=None
            ),
        )
    )
    rows: list[dict[str, Any]] = []
    for (
        market,
        index_id,
        interval_id,
        security,
        ticker,
    ), group in pit_daily.groupby(
        [
            "market",
            "index_id",
            "interval_id",
            "security_id",
            "ticker_on_date",
        ],
        sort=True,
    ):
        required_dates = sorted(set(group["membership_date"].map(as_date)))
        missing = [
            value
            for value in required_dates
            if (market, security, value) not in raw_keys
        ]
        if not missing:
            continue
        interval_gate = intervals[intervals["interval_id"].eq(interval_id)]
        blocks = bool(
            len(missing) / max(len(required_dates), 1)
            > (1.0 - PRICE_COVERAGE_THRESHOLD)
            or (
                len(interval_gate)
                and not bool(interval_gate.iloc[0]["eligible_for_forecasting"])
            )
        )
        suffix = {
            "PSEI": ".PS",
            "FBM_KLCI": ".KL",
            "IDX30": ".JK",
            "SET50": ".BK",
            "VN30": "",
        }[str(index_id)]
        rows.append(
            {
                "market": market,
                "index_id": index_id,
                "interval_id": interval_id,
                "security": security,
                "required_symbol": ticker,
                "provider_symbol_candidates": f"{ticker};{ticker}{suffix}",
                "first_required_date": required_dates[0],
                "last_required_date": required_dates[-1],
                "missing_date_range": f"{missing[0]}/{missing[-1]} ({len(missing)} dates)",
                "missing_field": "Open;High;Low;Close",
                "reason_required": (
                    "active point-in-time member lacks one or more raw OHLC rows"
                ),
                "recommended_source": (
                    "official exchange or evidence-audited unadjusted daily OHLC provider"
                ),
                "blocks_forecasting_eligibility": blocks,
            }
        )
    # FBM cannot yet enumerate historical securities because the membership
    # anchor itself is unresolved.  Preserve that limitation explicitly.
    rows.append(
        {
            "market": "Malaysia",
            "index_id": "FBM_KLCI",
            "interval_id": "BLOCKED-FBM_KLCI-NO-PIT-INTERVAL",
            "security": "UNENUMERATED_HISTORICAL_CONSTITUENTS",
            "required_symbol": "PENDING_OFFICIAL_FULL_CONSTITUENT_ANCHOR",
            "provider_symbol_candidates": "",
            "first_required_date": None,
            "last_required_date": None,
            "missing_date_range": "cannot enumerate until PIT membership is validated",
            "missing_field": "Open;High;Low;Close after membership enumeration",
            "reason_required": (
                "current-universe prices exist, but historical price requirements "
                "cannot be truthfully enumerated without a PIT anchor"
            ),
            "recommended_source": (
                "retained official FTSE Russell or Bursa full constituent list, "
                "then unadjusted security-level OHLC"
            ),
            "blocks_forecasting_eligibility": True,
        }
    )
    result = pd.DataFrame(rows, columns=ACQUISITION_COLUMNS)
    write_frame(ACQUISITION_CSV_PATH, result, ACQUISITION_COLUMNS)
    write_json(
        ACQUISITION_JSON_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "record_count": len(result),
            "records": result.to_dict("records"),
            "limitations": [
                "missing dates are not filled or interpolated",
                "FBM_KLCI securities cannot be enumerated before membership validation",
            ],
        },
    )
    return result


def build_blocker_audit(
    inventory: pd.DataFrame,
    raw: pd.DataFrame,
    recovery_audit: pd.DataFrame,
    folds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    candidate_counts, scan_issues = repository_candidate_file_counts()
    current = current_universe_frames()
    raw_summary = (
        raw.groupby("market", sort=True)
        .agg(
            raw_ohlc_rows=("date", "size"),
            raw_ohlc_securities=("security_id", "nunique"),
            raw_ohlc_first_date=("date", "min"),
            raw_ohlc_last_date=("date", "max"),
            duplicate_security_dates=(
                "date",
                lambda _: 0,
            ),
        )
        .reset_index()
        .set_index("market")
    )
    rows: list[dict[str, Any]] = []
    for interval in inventory.to_dict("records"):
        index_id = str(interval["index_id"])
        market = str(interval["market"])
        market_raw = raw[raw["market"].eq(market)]
        suffix_issues = int(
            market_raw["exchange_suffix"].fillna("").astype(str).eq("").sum()
            if index_id != "VN30"
            else 0
        )
        valid_folds = 0
        if folds is not None:
            valid_folds = int(
                folds[
                    folds["interval_id"].astype(str).eq(str(interval["interval_id"]))
                    & folds["valid_fold"].astype(bool)
                ]["fold_id"].nunique()
            )
        reasons = []
        if not bool(interval["eligible_for_forecasting"]):
            reasons.append(str(interval.get("blocking_reasons") or "data gate failed"))
        if bool(interval["eligible_for_forecasting"]) and valid_folds == 0 and folds is not None:
            reasons.append("no continuous rolling-origin fold")
        rows.append(
            {
                "audit_grain": "INTERVAL",
                "market": market,
                "index_id": index_id,
                "interval_id": interval["interval_id"],
                "interval_start": interval["interval_start"],
                "interval_end_exclusive": interval["interval_end_exclusive"],
                "available_pit_membership_dates": interval["trading_dates"],
                "current_universe_members": len(current[index_id]),
                "repository_ohlc_candidate_files": candidate_counts[index_id],
                "raw_ohlc_rows": int(raw_summary.loc[market, "raw_ohlc_rows"]),
                "raw_ohlc_securities": int(
                    raw_summary.loc[market, "raw_ohlc_securities"]
                ),
                "raw_ohlc_first_date": raw_summary.loc[market, "raw_ohlc_first_date"],
                "raw_ohlc_last_date": raw_summary.loc[market, "raw_ohlc_last_date"],
                "missing_ticker_mappings": 0,
                "delisted_or_renamed_securities": (
                    "EMP/EMI unresolved documented mapping" if index_id == "PSEI" else ""
                ),
                "duplicate_symbols": int(
                    current[index_id]["local_symbol"].duplicated().sum()
                ),
                "exchange_suffix_issues": suffix_issues,
                "missing_historical_price_joins": int(
                    interval["required_active_member_days"]
                    - interval["covered_active_member_days"]
                ),
                "invalid_ohlc_rows": int(
                    recovery_audit.loc[
                        recovery_audit["market"].eq(market), "invalid_ohlc_rows"
                    ].sum()
                ),
                "feature_lookback_coverage": interval["feature_lookback_coverage"],
                "next_session_target_pair_coverage": interval[
                    "target_pair_coverage"
                ],
                "market_calendar_alignment": (
                    "active-member dates intersected with observed market raw dates"
                ),
                "corporate_action_or_adjusted_price_ambiguity": (
                    "RAW_OHLC fields only; no adjusted-close field consumed; "
                    "archive has no corporate-action adjustment metadata"
                    if index_id == "PSEI"
                    else "RAW_OHLC only; adjusted close prohibited"
                ),
                "price_coverage": interval["price_coverage"],
                "continuous_valid_folds": valid_folds,
                "passes_data_gate": bool(interval["eligible_for_forecasting"]),
                "passes_continuous_protocol_gate": valid_folds >= MIN_VALID_FOLDS,
                "exact_reasons": " | ".join(reasons) or "PASS",
            }
        )
    for market, index_id, reason in [
        (
            "Philippines",
            "PSEI",
            (
                "2016-H2 effective date unresolved; 2023 extraordinary split dates "
                "unresolved; both windows preserved as non-membership gaps"
            ),
        ),
        (
            "Malaysia",
            "FBM_KLCI",
            (
                "no retained official full constituent anchor; 2016-12, 2018-06, "
                "and 2018-12 source gaps remain; current constituents not backcast"
            ),
        ),
    ]:
        summary = raw_summary.loc[market]
        rows.append(
            {
                "audit_grain": "MARKET_BLOCKER",
                "market": market,
                "index_id": index_id,
                "interval_id": None,
                "interval_start": None,
                "interval_end_exclusive": None,
                "available_pit_membership_dates": int(
                    inventory.loc[
                        inventory["index_id"].eq(index_id), "trading_dates"
                    ].sum()
                ),
                "current_universe_members": len(current[index_id]),
                "repository_ohlc_candidate_files": candidate_counts[index_id],
                "raw_ohlc_rows": int(summary["raw_ohlc_rows"]),
                "raw_ohlc_securities": int(summary["raw_ohlc_securities"]),
                "raw_ohlc_first_date": summary["raw_ohlc_first_date"],
                "raw_ohlc_last_date": summary["raw_ohlc_last_date"],
                "missing_ticker_mappings": (
                    1 if index_id == "PSEI" else 0
                ),
                "delisted_or_renamed_securities": (
                    "EMP/EMI unresolved documented mapping" if index_id == "PSEI" else ""
                ),
                "duplicate_symbols": int(
                    current[index_id]["local_symbol"].duplicated().sum()
                ),
                "exchange_suffix_issues": 0,
                "missing_historical_price_joins": None,
                "invalid_ohlc_rows": int(
                    recovery_audit.loc[
                        recovery_audit["market"].eq(market), "invalid_ohlc_rows"
                    ].sum()
                ),
                "feature_lookback_coverage": None,
                "next_session_target_pair_coverage": None,
                "market_calendar_alignment": "see interval rows",
                "corporate_action_or_adjusted_price_ambiguity": (
                    "RAW_OHLC only; archive adjustment metadata unavailable"
                ),
                "price_coverage": None,
                "continuous_valid_folds": int(
                    folds[
                        folds["index_id"].eq(index_id)
                        & folds["valid_fold"].astype(bool)
                    ]["fold_id"].nunique()
                )
                if folds is not None
                else 0,
                "passes_data_gate": False,
                "passes_continuous_protocol_gate": False,
                "exact_reasons": reason,
            }
        )
    if scan_issues:
        rows.append(
            {
                "audit_grain": "REPOSITORY_SCAN",
                "market": "ALL",
                "index_id": "ALL",
                "interval_id": None,
                "interval_start": None,
                "interval_end_exclusive": None,
                "available_pit_membership_dates": None,
                "current_universe_members": None,
                "repository_ohlc_candidate_files": sum(candidate_counts.values()),
                "raw_ohlc_rows": len(raw),
                "raw_ohlc_securities": raw["security_id"].nunique(),
                "raw_ohlc_first_date": raw["date"].min(),
                "raw_ohlc_last_date": raw["date"].max(),
                "missing_ticker_mappings": None,
                "delisted_or_renamed_securities": "",
                "duplicate_symbols": None,
                "exchange_suffix_issues": None,
                "missing_historical_price_joins": None,
                "invalid_ohlc_rows": None,
                "feature_lookback_coverage": None,
                "next_session_target_pair_coverage": None,
                "market_calendar_alignment": "repository scan only",
                "corporate_action_or_adjusted_price_ambiguity": "",
                "price_coverage": None,
                "continuous_valid_folds": None,
                "passes_data_gate": None,
                "passes_continuous_protocol_gate": None,
                "exact_reasons": (
                    f"{len(scan_issues)} inaccessible path(s) skipped: "
                    + " | ".join(scan_issues[:5])
                ),
            }
        )
    result = pd.DataFrame(rows)
    write_frame(BLOCKER_CSV_PATH, result)
    summary_lines = [
        "# Five-Market Data Blocker Audit V7",
        "",
        f"Experiment: `{EXPERIMENT_ID}`.",
        "",
        "This audit distinguishes raw price availability from validated point-in-time membership. "
        "Unavailable observations are never interpreted as zero, inactive, or non-member.",
        "",
        "| Index | PIT intervals | Data-ready intervals | Continuous valid folds | Main blocker |",
        "|---|---:|---:|---:|---|",
    ]
    for index_id in MARKETS:
        market_rows = result[
            result["index_id"].eq(index_id)
            & result["audit_grain"].eq("INTERVAL")
        ]
        blocker = result[
            result["index_id"].eq(index_id)
            & result["audit_grain"].eq("MARKET_BLOCKER")
        ]
        main = (
            str(blocker.iloc[0]["exact_reasons"])
            if len(blocker)
            else (
                "none beyond interval rows"
                if market_rows["passes_data_gate"].any()
                else "all interval data gates fail"
            )
        )
        summary_lines.append(
            f"| {index_id} | {len(market_rows)} | "
            f"{int(market_rows['passes_data_gate'].sum())} | "
            f"{int(market_rows['continuous_valid_folds'].sum())} | {main} |"
        )
    summary_lines.extend(
        [
            "",
            "Detailed interval-level coverage, mapping, suffix, invalid-row, lookback, "
            "target-pair, calendar, and corporate-action fields are in the companion CSV.",
        ]
    )
    BLOCKER_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BLOCKER_MD_PATH.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return result


def expected_config(inventory: pd.DataFrame) -> dict[str, Any]:
    feature_source = (
        REPO_ROOT
        / "papers"
        / "paper_03_raw_ohlc_forecasting"
        / "src"
        / "features"
        / "raw_ohlc.py"
    )
    eligible = inventory[inventory["eligible_for_forecasting"].astype(bool)]
    return {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_date": RUN_DATE,
        "v6_reference": {
            "experiment_id": V6_EXPERIMENT_ID,
            "data_version": V6_DATA_VERSION,
            "membership_version": V6_MEMBERSHIP_VERSION,
            "manifest_version": V6_MANIFEST_VERSION,
            "correction_policy": "V6 immutable; changes require V6.1 or later",
        },
        "data_version": DATA_VERSION,
        "membership_version": MEMBERSHIP_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "predefined_market_scope": list(MARKETS),
        "eligible_interval_ids": sorted(eligible["interval_id"].astype(str)),
        "eligible_intervals_by_market": {
            index_id: int(
                eligible[eligible["index_id"].eq(index_id)]["interval_id"].nunique()
            )
            for index_id in MARKETS
        },
        "universe_arms": {
            "pit": PIT,
            "current": CURRENT,
            "intersection": INTERSECTION,
        },
        "target_definition": {
            "name": "next-session open-to-close direction",
            "target_return": "target_close / target_open - 1",
            "direction": "1 if target_return > 0 else 0",
            "next_observation": "next valid same-security observation",
            "maximum_calendar_gap_days": MAX_TARGET_GAP_DAYS,
            "adjusted_close_allowed": False,
            "volume_allowed": False,
            "imputation_allowed": False,
        },
        "feature_registry_hash": sha256_file(feature_source),
        "feature_version": v6.FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "feature_lookback": FEATURE_LOOKBACK,
        "continuous_market_level_rolling_origin": {
            "primary_market_pooling": False,
            "training_history": (
                "all earlier valid observations in the same market and arm, "
                "including prior membership intervals"
            ),
            "evaluation": "only intervals passing the unchanged data gate",
            "initial_training_dates": INITIAL_TRAINING_DATES,
            "calibration_dates": CALIBRATION_DATES,
            "test_block_dates": TEST_BLOCK_DATES,
            "step_dates": STEP_DATES,
            "training_window": "expanding",
            "labels_must_mature_before_fit": True,
            "test_blocks_strictly_out_of_sample": True,
        },
        "minimum_validity_gates": {
            "valid_folds_per_market_arm_model": MIN_VALID_FOLDS,
            "oos_trading_dates_per_market_arm_model": MIN_OOS_DATES,
            "oos_security_date_predictions_per_market_arm_model": MIN_OOS_ROWS,
            "both_target_classes_train": True,
            "both_target_classes_test": True,
        },
        "model_registry": MODEL_SPECS,
        "seeds": SEEDS,
        "primary_metric": "brier_score",
        "secondary_metrics": [
            "log_loss",
            "balanced_accuracy",
            "calibration_intercept",
            "calibration_slope",
            "prediction_coverage",
            "model_rank",
        ],
        "aggregation_hierarchy": [
            "equal-security mean within date",
            "equal-date mean within evaluation interval",
            "equal-interval mean within market",
            "equal-market mean across model-valid markets",
        ],
        "bootstrap": {
            "method": "paired moving-block bootstrap",
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "primary_block_length": PRIMARY_BLOCK_LENGTH,
            "robustness_block_lengths": ROBUSTNESS_BLOCK_LENGTHS,
        },
        "multiple_testing": (
            "Benjamini-Hochberg across all valid current-minus-PIT comparisons "
            "in each declared block-length family"
        ),
        "package_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "prohibited_claims": [
            "manuscript",
            "profitability",
            "causal",
            "trading",
        ],
    }


def freeze_config(inventory: pd.DataFrame) -> dict[str, Any]:
    config = expected_config(inventory)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config


def build_composition(members: pd.DataFrame) -> pd.DataFrame:
    counts = (
        members.groupby(
            ["interval_id", "market", "index_id", "membership_date", "universe_semantics"],
            sort=True,
        )["security_id"]
        .nunique()
        .unstack("universe_semantics", fill_value=0)
        .reset_index()
    )
    for semantics in (PIT, CURRENT, INTERSECTION):
        if semantics not in counts:
            counts[semantics] = 0
    return counts.rename(
        columns={
            "membership_date": "prediction_date",
            PIT: "PIT_constituent_count",
            CURRENT: "current_constituent_count",
            INTERSECTION: "intersection_count",
        }
    )


def _empty_report(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def run_analytics(
    inventory: pd.DataFrame,
    members: pd.DataFrame,
    seed_level: pd.DataFrame,
    collapsed: pd.DataFrame,
    cells: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    valid_cells = cells[cells["valid"].astype(bool)][
        ["index_id", "universe_semantics", "model_id"]
    ]
    valid_predictions = collapsed.merge(
        valid_cells,
        on=["index_id", "universe_semantics", "model_id"],
        how="inner",
        validate="many_to_one",
    )
    composition = build_composition(members)
    if valid_predictions.empty:
        by_date = _empty_report(
            [
                "interval_id",
                "market",
                "index_id",
                "universe_semantics",
                "model_id",
                "prediction_date",
                "brier_score",
            ]
        )
        by_interval = by_market = macro = by_date.copy()
        current_minus_pit = _empty_report(["aggregation_level", "model_id"])
        inference = _empty_report(
            [
                "inference_level",
                "model_id",
                "metric",
                "block_length",
                "valid",
                "bh_adjusted_p_value",
            ]
        )
        seed_robustness = _empty_report(["index_id", "model_id"])
        rank = _empty_report(["scope_level", "model_id", "mean_rank"])
        heterogeneity = _empty_report(["model_id", "metric"])
    else:
        by_date = analytics.build_performance_by_date(
            valid_predictions, composition
        )
        by_interval, by_market, macro = analytics.build_performance_hierarchy(
            by_date
        )
        current_minus_pit, date_diff = analytics.build_current_minus_pit(
            by_date, by_interval, by_market, macro
        )
        inference = analytics.build_bootstrap_inference(date_diff)
        valid_seed = seed_level.merge(
            valid_cells,
            on=["index_id", "universe_semantics", "model_id"],
            how="inner",
            validate="many_to_one",
        )
        seed_robustness = analytics.build_seed_robustness(valid_seed)
        rank = analytics.build_model_rank(
            by_interval,
            by_market,
            valid_predictions,
            seed_robustness,
        )
        heterogeneity = analytics.build_heterogeneity(
            current_minus_pit, inference
        )
    outputs = {
        "pit_multimarket_performance_by_date_v7": by_date,
        "pit_multimarket_performance_by_interval_v7": by_interval,
        "pit_multimarket_performance_by_market_v7": by_market,
        "pit_multimarket_equal_market_macro_v7": macro,
        "pit_multimarket_current_minus_pit_v7": current_minus_pit,
        "pit_multimarket_bootstrap_inference_v7": inference,
        "pit_multimarket_heterogeneity_v7": heterogeneity,
        "pit_multimarket_model_rank_v7": rank,
        "pit_multimarket_seed_robustness_v7": seed_robustness,
    }
    for base, frame in outputs.items():
        write_frame(REPORTS / f"{base}.csv", frame)
    return outputs


def build_market_inventory(
    interval_inventory: pd.DataFrame,
    folds: pd.DataFrame,
    cells: pd.DataFrame,
    collapsed: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index_id, market in MARKETS.items():
        intervals = interval_inventory[interval_inventory["index_id"].eq(index_id)]
        market_folds = folds[
            folds["index_id"].eq(index_id) & folds["valid_fold"].astype(bool)
        ]
        market_cells = cells[cells["index_id"].eq(index_id)]
        predictions = collapsed[collapsed["index_id"].eq(index_id)]
        pit_cells = market_cells[
            market_cells["universe_semantics"].eq(PIT)
            & market_cells["valid"].astype(bool)
        ]
        rows.append(
            {
                "market": market,
                "index_id": index_id,
                "validated_pit_membership": bool(len(intervals)),
                "validated_raw_ohlc": bool(len(raw[raw["market"].eq(market)])),
                "pit_intervals": int(intervals["interval_id"].nunique()),
                "data_ready_intervals": int(
                    intervals["eligible_for_forecasting"].astype(bool).sum()
                ),
                "valid_continuous_folds": int(market_folds["fold_id"].nunique()),
                "oos_trading_dates": int(predictions["prediction_date"].nunique()),
                "seed_collapsed_predictions": len(predictions),
                "valid_pit_model_cells": len(pit_cells),
                "valid_all_arm_model_cells": int(
                    market_cells["valid"].astype(bool).sum()
                ),
                "invalid_cells": int((~market_cells["valid"].astype(bool)).sum()),
                "forecasting_inference_valid": len(pit_cells) == len(MODEL_SPECS),
                "status": (
                    "MODEL_VALID"
                    if len(pit_cells) == len(MODEL_SPECS)
                    else "DATA_OR_PROTOCOL_BLOCKED"
                ),
            }
        )
    result = pd.DataFrame(rows)
    write_frame(INVENTORY_PATH, result)
    return result


def load_test_summary() -> dict[str, Any]:
    if not TEST_SUMMARY_PATH.is_file():
        return {
            "status": "PENDING",
            "passed": 0,
            "failed": 0,
            "commands": [],
        }
    return json.loads(TEST_SUMMARY_PATH.read_text(encoding="utf-8"))


def completion_gates(
    market_inventory: pd.DataFrame,
    cells: pd.DataFrame,
    v6_integrity: Mapping[str, Any],
    artifact_status: str,
    test_summary: Mapping[str, Any],
) -> dict[str, bool]:
    all_markets = len(market_inventory) == 5
    return {
        "all_five_validated_pit_membership": bool(
            all_markets
            and market_inventory["validated_pit_membership"].astype(bool).all()
        ),
        "all_five_validated_raw_ohlc": bool(
            all_markets
            and market_inventory["validated_raw_ohlc"].astype(bool).all()
        ),
        "each_market_at_least_two_valid_folds": bool(
            all_markets
            and (market_inventory["valid_continuous_folds"] >= MIN_VALID_FOLDS).all()
        ),
        "each_market_at_least_40_oos_dates": bool(
            all_markets
            and (market_inventory["oos_trading_dates"] >= MIN_OOS_DATES).all()
        ),
        "each_market_at_least_800_predictions": bool(
            all_markets
            and (
                market_inventory["seed_collapsed_predictions"] >= MIN_OOS_ROWS
            ).all()
        ),
        "both_classes_train_and_test": bool(
            len(cells) and cells["valid"].astype(bool).all()
        ),
        "no_unresolved_high_severity_leakage_or_mapping_conflict": False,
        "all_generated_artifact_checksums_valid": artifact_status == "PASS",
        "all_new_tests_pass": test_summary.get("status") == "PASS",
        "all_three_arms_have_results": bool(
            len(cells)
            and cells.groupby("index_id")["valid"].all().all()
        ),
        "v6_and_locked_v4_immutable": bool(
            v6_integrity["v6_manifest_checksums_valid"]
            and v6_integrity["v4_locked_checksums_valid"]
        ),
    }


def classify_status(gates: Mapping[str, bool], market_inventory: pd.DataFrame) -> str:
    required = [
        key
        for key in gates
        if key != "v6_and_locked_v4_immutable"
    ]
    if all(gates[key] for key in required) and gates["v6_and_locked_v4_immutable"]:
        return "COMPLETE_FIVE_MARKET"
    if not gates["v6_and_locked_v4_immutable"]:
        return "FAILED_INTEGRITY"
    data_blocked = not (
        gates["all_five_validated_pit_membership"]
        and gates["all_five_validated_raw_ohlc"]
        and gates["no_unresolved_high_severity_leakage_or_mapping_conflict"]
    )
    model_markets = market_inventory[
        market_inventory["validated_pit_membership"].astype(bool)
    ]
    protocol_blocked = bool(
        len(model_markets)
        and (
            (model_markets["valid_continuous_folds"] < MIN_VALID_FOLDS)
            | (model_markets["oos_trading_dates"] < MIN_OOS_DATES)
        ).any()
    )
    if data_blocked and protocol_blocked:
        return "PARTIAL_DATA_AND_PROTOCOL_BLOCKED"
    if data_blocked:
        return "PARTIAL_DATA_BLOCKED"
    return "PARTIAL_PROTOCOL_BLOCKED"


def required_artifact_paths() -> list[Path]:
    return [
        SCRIPT_PATH,
        REPO_ROOT / "tests" / "point_in_time" / "test_pit_multimarket_forecasting_v7.py",
        CONFIG_PATH,
        OBSERVATION_PATH,
        FEATURE_PATH,
        FOLD_PATH,
        SEED_PATH,
        COLLAPSED_PATH,
        RECOVERED_RAW_PATH,
        MEMBERSHIP_PATH,
        MEMBERSHIP_DAILY_PATH,
        REJECTION_PATH,
        ALL_CELLS_PATH,
        INVENTORY_PATH,
        INTERVAL_INVENTORY_PATH,
        BLOCKER_CSV_PATH,
        BLOCKER_MD_PATH,
        REPORTS / "pit_multimarket_invalid_cells_v7.csv",
        REPORTS / "pit_multimarket_performance_by_interval_v7.csv",
        REPORTS / "pit_multimarket_performance_by_market_v7.csv",
        REPORTS / "pit_multimarket_equal_market_macro_v7.csv",
        REPORTS / "pit_multimarket_current_minus_pit_v7.csv",
        REPORTS / "pit_multimarket_bootstrap_inference_v7.csv",
        REPORTS / "pit_multimarket_heterogeneity_v7.csv",
        REPORTS / "pit_multimarket_model_rank_v7.csv",
        REPORTS / "pit_multimarket_seed_robustness_v7.csv",
        REPORTS / "psei_repository_raw_recovery_v7.csv",
        ACQUISITION_CSV_PATH,
        ACQUISITION_JSON_PATH,
        PREFLIGHT_PATH,
        TEST_SUMMARY_PATH,
        ARTIFACT_VALIDATION_PATH,
        STATUS_JSON_PATH,
        STATUS_MD_PATH,
    ]


def build_manifest(paths: Sequence[Path]) -> dict[str, Any]:
    records = []
    for path in paths:
        if not path.is_file():
            continue
        records.append(
            {
                "artifact_relative_path": path.relative_to(REPO_ROOT).as_posix(),
                "checksum": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "manifest_version": MANIFEST_VERSION,
                "repository_root_hint": ".",
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "manifest_version": MANIFEST_VERSION,
        "validation_key": "artifact_relative_path",
        "artifact_count": len(records),
        "records": records,
    }
    write_json(MANIFEST_PATH, payload)
    return payload


def validate_artifacts(
    manifest: Mapping[str, Any], *, write_result: bool = True
) -> dict[str, Any]:
    failures: list[str] = []
    required = required_artifact_paths()
    for path in required:
        if not path.is_file():
            failures.append(f"missing required artifact: {path.relative_to(REPO_ROOT)}")
    for record in manifest["records"]:
        relative = Path(record["artifact_relative_path"])
        if relative.is_absolute():
            failures.append(f"absolute manifest key: {relative}")
            continue
        path = REPO_ROOT / relative
        if not path.is_file():
            failures.append(f"manifest file missing: {relative}")
        elif sha256_file(path) != record["checksum"]:
            failures.append(f"manifest checksum mismatch: {relative}")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if not failures else "FAIL",
        "manifest_records": len(manifest["records"]),
        "repository_relative_keys": not any(
            Path(record["artifact_relative_path"]).is_absolute()
            for record in manifest["records"]
        ),
        "checksum_failures": failures,
    }
    if write_result:
        write_json(ARTIFACT_VALIDATION_PATH, payload)
    return payload


def write_status(
    *,
    market_inventory: pd.DataFrame,
    interval_inventory: pd.DataFrame,
    cells: pd.DataFrame,
    folds: pd.DataFrame,
    seed_level: pd.DataFrame,
    collapsed: pd.DataFrame,
    recovery_audit: pd.DataFrame,
    acquisition: pd.DataFrame,
    v6_integrity: Mapping[str, Any],
    manifest_records: int,
    artifact_status: str,
) -> dict[str, Any]:
    test_summary = load_test_summary()
    gates = completion_gates(
        market_inventory,
        cells,
        v6_integrity,
        artifact_status,
        test_summary,
    )
    status = classify_status(gates, market_inventory)
    valid_cells = cells[cells["valid"].astype(bool)]
    invalid_cells = cells[~cells["valid"].astype(bool)]
    dates_by_market = {
        index_id: int(
            collapsed.loc[
                collapsed["index_id"].astype(str).eq(index_id),
                "prediction_date",
            ].nunique()
        )
        for index_id in MARKETS
    }
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "status": status,
        "complete_five_market": status == "COMPLETE_FIVE_MARKET",
        "completion_gates": gates,
        "markets_with_valid_forecasting_inference": sorted(
            market_inventory.loc[
                market_inventory["forecasting_inference_valid"].astype(bool),
                "index_id",
            ].astype(str)
        ),
        "valid_cells_by_market": {
            index_id: int(
                valid_cells["index_id"].astype(str).eq(index_id).sum()
            )
            for index_id in MARKETS
        },
        "invalid_cells_by_market": {
            index_id: int(
                invalid_cells["index_id"].astype(str).eq(index_id).sum()
            )
            for index_id in MARKETS
        },
        "oos_trading_dates_by_market": dates_by_market,
        "seed_level_predictions": len(seed_level),
        "seed_collapsed_predictions": len(collapsed),
        "data_recovery": {
            "psei_archive_files_with_valid_rows": int(
                recovery_audit["status"].eq("RECOVERED_VALID_ROWS").sum()
            ),
            "psei_recovered_rows_before_v5_precedence": int(
                recovery_audit["accepted_rows"].sum()
            ),
            "psei_invalid_rows_excluded": int(
                recovery_audit.loc[
                    recovery_audit["market"].eq("Philippines"),
                    "invalid_ohlc_rows",
                ].sum()
            ),
            "fbm_klci_current_raw_rows_present": int(
                pd.read_parquet(RECOVERED_RAW_PATH)["market"].eq("Malaysia").sum()
            ),
            "fbm_klci_pit_membership_recovered": False,
        },
        "idx30_continuous_history_valid": bool(
            market_inventory.loc[
                market_inventory["index_id"].eq("IDX30"),
                "forecasting_inference_valid",
            ].iloc[0]
        ),
        "unresolved_missing_data_records": len(acquisition),
        "test_summary": test_summary,
        "manifest_records": manifest_records,
        "artifact_checksum_status": artifact_status,
        "v6_integrity": dict(v6_integrity),
        "market_inventory": market_inventory.to_dict("records"),
        "eligible_intervals_by_market": {
            index_id: int(
                interval_inventory[
                    interval_inventory["index_id"].eq(index_id)
                    & interval_inventory["eligible_for_forecasting"].astype(bool)
                ]["interval_id"].nunique()
            )
            for index_id in MARKETS
        },
        "protocol_differences_from_v6": [
            "V6 fits within isolated eligible intervals; V7 uses expanding prior history from earlier intervals in the same market and arm.",
            "V6 applies minimum fold/date/prediction gates to interval cells; V7 applies the declared gates to market-arm-model cells while retaining interval results.",
            "V7 preserves dynamic membership on every date and permits interval boundaries without resetting training history.",
            "V7 adds evidence-backed PSEI chain resets and repository-local raw OHLC recovery; unresolved windows remain gaps.",
            "V7 retains the V6 feature, target, model, seed, metric, aggregation, bootstrap, and BH registries.",
        ],
        "bounded_conclusion": (
            "Forecasting and current-minus-PIT results are predictive-loss "
            "comparisons only. Non-significance is not equivalence. No "
            "profitability, causal, manuscript, or trading claim is made."
        ),
    }
    write_json(STATUS_JSON_PATH, payload)
    lines = [
        "# PIT Multimarket Forecasting Status V7",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Five-market status: `{status}`",
        f"- Complete five-market: `{str(status == 'COMPLETE_FIVE_MARKET').lower()}`",
        "- Valid forecasting markets: "
        + (
            ", ".join(payload["markets_with_valid_forecasting_inference"])
            or "none"
        ),
        f"- Seed-level predictions: {len(seed_level):,}",
        f"- Seed-collapsed predictions: {len(collapsed):,}",
        f"- Manifest records: {manifest_records}",
        f"- Artifact checksum status: `{artifact_status}`",
        "",
        "## Market status",
        "",
        "| Index | PIT intervals | Ready intervals | Folds | OOS dates | Valid cells | Invalid cells |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in market_inventory.to_dict("records"):
        lines.append(
            f"| {row['index_id']} | {row['pit_intervals']} | "
            f"{row['data_ready_intervals']} | {row['valid_continuous_folds']} | "
            f"{row['oos_trading_dates']} | {row['valid_all_arm_model_cells']} | "
            f"{row['invalid_cells']} |"
        )
    lines.extend(
        [
            "",
            "## Protocol difference",
            "",
            "V7 trains continuously with expanding earlier history from the same market "
            "and universe arm. Test observations remain inside their actual PIT interval, "
            "labels mature before fitting, and interval boundaries do not reset history.",
            "",
            "## Boundaries",
            "",
            payload["bounded_conclusion"],
        ]
    )
    STATUS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_data_phase() -> dict[str, Any]:
    v6_integrity = validate_v6_immutability()
    if not v6_integrity["v6_manifest_checksums_valid"]:
        raise RuntimeError("V6 integrity failed before V7 data execution")
    raw, recovery_audit = recover_repository_raw_ohlc()
    membership_intervals = build_membership_intervals()
    pit_daily = build_pit_membership_daily(membership_intervals, raw)
    all_members = build_all_arm_membership(pit_daily)
    interval_inventory = build_interval_inventory(
        membership_intervals, pit_daily, raw
    )
    freeze_config(interval_inventory)
    observations, rejections = construct_observations(all_members, raw)
    observations, features, rejections = construct_features(
        observations, rejections, all_members, raw
    )
    write_frame(OBSERVATION_PATH, observations, OBSERVATION_COLUMNS)
    write_frame(FEATURE_PATH, features)
    write_frame(REJECTION_PATH, rejections, OBSERVATION_COLUMNS)
    folds = build_fold_calendar(interval_inventory, pit_daily)
    build_acquisition_manifest(interval_inventory, pit_daily, raw)
    build_blocker_audit(interval_inventory, raw, recovery_audit, folds)
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "predefined_markets": list(MARKETS),
        "membership_intervals": len(interval_inventory),
        "eligible_intervals": int(
            interval_inventory["eligible_for_forecasting"].astype(bool).sum()
        ),
        "valid_continuous_folds": int(folds["valid_fold"].astype(bool).sum()),
        "observation_rows": len(observations),
        "feature_rows": len(features),
        "rejection_rows": len(rejections),
        "recovered_raw_rows": len(raw),
        "v6_integrity": v6_integrity,
    }
    write_json(PREFLIGHT_PATH, preflight)
    if not TEST_SUMMARY_PATH.is_file():
        write_json(
            TEST_SUMMARY_PATH,
            {
                "status": "PENDING",
                "passed": 0,
                "failed": 0,
                "commands": [],
            },
        )
    return preflight


def run_model_phase() -> dict[str, Any]:
    features = pd.read_parquet(FEATURE_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    seed_level, collapsed, failures = run_oos_models(features, folds)
    write_frame(SEED_PATH, seed_level, PREDICTION_COLUMNS)
    write_frame(
        COLLAPSED_PATH,
        collapsed,
        PREDICTION_COLUMNS + ["seed_count"],
    )
    cells = validate_market_cells(folds, collapsed, failures)
    write_frame(ALL_CELLS_PATH, cells)
    write_frame(
        REPORTS / "pit_multimarket_invalid_cells_v7.csv",
        cells[~cells["valid"].astype(bool)],
    )
    return {
        "seed_level_predictions": len(seed_level),
        "collapsed_predictions": len(collapsed),
        "valid_cells": int(cells["valid"].astype(bool).sum()),
        "invalid_cells": int((~cells["valid"].astype(bool)).sum()),
        "execution_failures": len(failures),
    }


def run_reporting_phase() -> dict[str, Any]:
    interval_inventory = pd.read_csv(INTERVAL_INVENTORY_PATH)
    interval_inventory["eligible_for_forecasting"] = _bool_series(
        interval_inventory["eligible_for_forecasting"]
    )
    folds = pd.read_parquet(FOLD_PATH)
    members = pd.read_parquet(MEMBERSHIP_DAILY_PATH)
    raw = pd.read_parquet(RECOVERED_RAW_PATH)
    recovery_audit = pd.read_csv(REPORTS / "psei_repository_raw_recovery_v7.csv")
    seed_level = pd.read_parquet(SEED_PATH)
    collapsed = pd.read_parquet(COLLAPSED_PATH)
    cells = pd.read_parquet(ALL_CELLS_PATH)
    acquisition = pd.read_csv(ACQUISITION_CSV_PATH)
    outputs = run_analytics(
        interval_inventory, members, seed_level, collapsed, cells
    )
    market_inventory = build_market_inventory(
        interval_inventory, folds, cells, collapsed, raw
    )
    build_blocker_audit(interval_inventory, raw, recovery_audit, folds)
    v6_integrity = validate_v6_immutability()
    write_status(
        market_inventory=market_inventory,
        interval_inventory=interval_inventory,
        cells=cells,
        folds=folds,
        seed_level=seed_level,
        collapsed=collapsed,
        recovery_audit=recovery_audit,
        acquisition=acquisition,
        v6_integrity=v6_integrity,
        manifest_records=0,
        artifact_status="PENDING",
    )
    # Ensure the validation file exists before it becomes a manifested artifact.
    write_json(
        ARTIFACT_VALIDATION_PATH,
        {
            "experiment_id": EXPERIMENT_ID,
            "status": "PENDING",
            "manifest_records": 0,
            "repository_relative_keys": True,
            "checksum_failures": [],
        },
    )
    manifest = build_manifest(required_artifact_paths())
    validation = validate_artifacts(manifest)
    write_status(
        market_inventory=market_inventory,
        interval_inventory=interval_inventory,
        cells=cells,
        folds=folds,
        seed_level=seed_level,
        collapsed=collapsed,
        recovery_audit=recovery_audit,
        acquisition=acquisition,
        v6_integrity=v6_integrity,
        manifest_records=len(manifest["records"]),
        artifact_status=validation["status"],
    )
    # Finalize the manifest only after status and validation are stable. The
    # final read-only validation must not rewrite a manifested artifact.
    manifest = build_manifest(required_artifact_paths())
    validation = validate_artifacts(manifest, write_result=False)
    if validation["status"] != "PASS":
        raise RuntimeError(
            "final V7 artifact manifest validation failed: "
            + " | ".join(validation["checksum_failures"])
        )
    return {
        "status": json.loads(STATUS_JSON_PATH.read_text())["status"],
        "manifest_records": len(manifest["records"]),
        "artifact_validation": validation["status"],
        "valid_cells": int(cells["valid"].astype(bool).sum()),
        "invalid_cells": int((~cells["valid"].astype(bool)).sum()),
        "analytics_outputs": len(outputs),
    }


def record_test_summary(
    *, passed: int, failed: int, command: str, scope: str
) -> dict[str, Any]:
    existing = load_test_summary()
    commands = list(existing.get("commands", []))
    commands.append(
        {
            "scope": scope,
            "command": command,
            "passed": int(passed),
            "failed": int(failed),
        }
    )
    payload = {
        "status": "PASS" if failed == 0 else "FAIL",
        "passed": int(passed),
        "failed": int(failed),
        "commands": commands,
    }
    write_json(TEST_SUMMARY_PATH, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=["data", "models", "report", "full", "record-tests"],
        default="full",
    )
    parser.add_argument("--passed", type=int, default=0)
    parser.add_argument("--failed", type=int, default=0)
    parser.add_argument("--command", default="")
    parser.add_argument("--scope", default="V7 tests")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "data":
        result = run_data_phase()
    elif args.phase == "models":
        result = run_model_phase()
    elif args.phase == "report":
        result = run_reporting_phase()
    elif args.phase == "record-tests":
        result = record_test_summary(
            passed=args.passed,
            failed=args.failed,
            command=args.command,
            scope=args.scope,
        )
    else:
        run_data_phase()
        run_model_phase()
        result = run_reporting_phase()
    print(json.dumps(json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
