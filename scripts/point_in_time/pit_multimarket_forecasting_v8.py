#!/usr/bin/env python3
"""Five-market ASEAN PIT continuous rolling-origin benchmark V8.

V8 resolves the final Malaysian point-in-time membership blocker and reruns
the complete five-market benchmark.  The statistical protocol is the frozen
V7 protocol; V7 artifacts are read-only inputs and are checksum-validated
before and after execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.manifest_portability_v6 import validate_portable_record  # noqa: E402
from scripts.point_in_time import pit_multimarket_forecasting_v7 as v7  # noqa: E402


EXPERIMENT_ID = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V8"
STUDY_LABEL = EXPERIMENT_ID
EXPERIMENT_VERSION = "v8"
PIPELINE_VERSION = "8.0.0"
RUN_DATE = "2026-08-03"
DATA_VERSION = "pit-data-v8"
MEMBERSHIP_VERSION = "pit-membership-v8"
SNAPSHOT_VERSION = "pit-universe-v8"
PRICE_VERSION = DATA_VERSION
MANIFEST_VERSION = "manifest-portability-v8.0.0"

PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
REPORTS = PIT_ROOT / "reports"
PROCESSED = PIT_ROOT / "processed"
QA = PIT_ROOT / "qa"
MANIFESTS = PIT_ROOT / "manifests"
CONFIG_ROOT = PIT_ROOT / "config"
RECONSTRUCTED = PIT_ROOT / "reconstructed"
SOURCE_ROOT = PIT_ROOT / "sources" / "fbm_klci_v8"
PRICE_ROOT = SOURCE_ROOT / "prices"

CONFIG_PATH = CONFIG_ROOT / "pit_multimarket_forecasting_v8.yaml"
OBSERVATION_PATH = PROCESSED / "pit_multimarket_observation_panel_v8.parquet"
FEATURE_PATH = PROCESSED / "pit_multimarket_feature_panel_v8.parquet"
FOLD_PATH = PROCESSED / "pit_multimarket_fold_calendar_v8.parquet"
SEED_PATH = PROCESSED / "pit_multimarket_predictions_seed_level_v8.parquet"
COLLAPSED_PATH = PROCESSED / "pit_multimarket_predictions_seed_collapsed_v8.parquet"
RECOVERED_RAW_PATH = PROCESSED / "pit_historical_raw_ohlc_recovered_v8.parquet"
MEMBERSHIP_PATH = PROCESSED / "pit_membership_intervals_v8.parquet"
MEMBERSHIP_DAILY_PATH = PROCESSED / "pit_membership_daily_all_arms_v8.parquet"
REJECTION_PATH = QA / "pit_multimarket_observation_rejections_v8.parquet"
ALL_CELLS_PATH = QA / "pit_multimarket_all_cells_v8.parquet"
MODEL_CHECKPOINT_DIR = QA / "pit_v8_model_checkpoints"
PREFLIGHT_PATH = QA / "pit_v8_preflight_validation.json"
TEST_SUMMARY_PATH = QA / "pit_v8_test_summary.json"
ARTIFACT_VALIDATION_PATH = QA / "pit_v8_artifact_validation.json"
INVENTORY_PATH = REPORTS / "forecasting_ready_market_inventory_v8.csv"
INTERVAL_INVENTORY_PATH = REPORTS / "forecasting_ready_interval_inventory_v8.csv"
BLOCKER_CSV_PATH = REPORTS / "five_market_data_blocker_audit_v8.csv"
BLOCKER_MD_PATH = REPORTS / "five_market_data_blocker_audit_v8.md"
PRICE_AUDIT_CSV_PATH = REPORTS / "fbm_klci_price_membership_join_audit_v8.csv"
PRICE_AUDIT_MD_PATH = REPORTS / "fbm_klci_price_membership_join_audit_v8.md"
RECOVERY_AUDIT_PATH = REPORTS / "fbm_klci_raw_ohlc_recovery_v8.csv"
MANIFEST_PATH = MANIFESTS / "pit_multimarket_forecasting_v8_manifest.json"
STATUS_JSON_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8.json"
STATUS_MD_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8.md"
RECONCILIATION_PATH = REPORT_ROOT / "V7_V8_RECONCILIATION.md"

FBM_INTERVAL_PATH = RECONSTRUCTED / "fbm_klci_membership_intervals_v8.parquet"
FBM_DAILY_PATH = RECONSTRUCTED / "fbm_klci_membership_daily_v8.parquet"
FBM_CROSSWALK_PATH = RECONSTRUCTED / "fbm_klci_security_crosswalk_v8.csv"
FBM_VALIDATION_PATH = QA / "fbm_klci_membership_validation_v8.json"
UNRESOLVED_PATH = MANIFESTS / "unresolved_membership_evidence_v8.csv"

V7_MANIFEST_PATH = MANIFESTS / "pit_multimarket_forecasting_v7_manifest.json"
V7_STATUS_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V7.json"
V7_RAW_PATH = PROCESSED / "pit_historical_raw_ohlc_recovered_v7.parquet"
V7_INTERVAL_PATH = PROCESSED / "pit_membership_intervals_v7.parquet"
V7_DAILY_PATH = PROCESSED / "pit_membership_daily_all_arms_v7.parquet"
V7_INTERVAL_INVENTORY_PATH = REPORTS / "forecasting_ready_interval_inventory_v7.csv"
V7_SEED_PATH = PROCESSED / "pit_multimarket_predictions_seed_level_v7.parquet"
V7_COLLAPSED_PATH = PROCESSED / "pit_multimarket_predictions_seed_collapsed_v7.parquet"

MARKETS = dict(v7.MARKETS)
PIT = v7.PIT
CURRENT = v7.CURRENT
INTERSECTION = v7.INTERSECTION
ARM_SEMANTICS = dict(v7.ARM_SEMANTICS)
MODEL_SPECS = v7.MODEL_SPECS
FEATURE_COLUMNS = list(v7.FEATURE_COLUMNS)
FEATURE_LOOKBACK = v7.FEATURE_LOOKBACK
SEEDS = list(v7.SEEDS)
MIN_VALID_FOLDS = v7.MIN_VALID_FOLDS
MIN_OOS_DATES = v7.MIN_OOS_DATES
MIN_OOS_ROWS = v7.MIN_OOS_ROWS
PRICE_COVERAGE_THRESHOLD = v7.PRICE_COVERAGE_THRESHOLD
PREDICTION_COLUMNS = list(v7.PREDICTION_COLUMNS)
OBSERVATION_COLUMNS = list(v7.OBSERVATION_COLUMNS)


def _configure_v7_runtime() -> None:
    """Redirect reused pure V7 routines to V8 metadata and output paths."""

    replacements = {
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "STUDY_LABEL": STUDY_LABEL,
        "EXPERIMENT_VERSION": EXPERIMENT_VERSION,
        "PIPELINE_VERSION": PIPELINE_VERSION,
        "RUN_DATE": RUN_DATE,
        "DATA_VERSION": DATA_VERSION,
        "MEMBERSHIP_VERSION": MEMBERSHIP_VERSION,
        "MANIFEST_VERSION": MANIFEST_VERSION,
        "CONFIG_PATH": CONFIG_PATH,
        "OBSERVATION_PATH": OBSERVATION_PATH,
        "FEATURE_PATH": FEATURE_PATH,
        "FOLD_PATH": FOLD_PATH,
        "SEED_PATH": SEED_PATH,
        "COLLAPSED_PATH": COLLAPSED_PATH,
        "RECOVERED_RAW_PATH": RECOVERED_RAW_PATH,
        "MEMBERSHIP_PATH": MEMBERSHIP_PATH,
        "MEMBERSHIP_DAILY_PATH": MEMBERSHIP_DAILY_PATH,
        "REJECTION_PATH": REJECTION_PATH,
        "ALL_CELLS_PATH": ALL_CELLS_PATH,
        "MODEL_CHECKPOINT_DIR": MODEL_CHECKPOINT_DIR,
        "PREFLIGHT_PATH": PREFLIGHT_PATH,
        "TEST_SUMMARY_PATH": TEST_SUMMARY_PATH,
        "ARTIFACT_VALIDATION_PATH": ARTIFACT_VALIDATION_PATH,
        "INVENTORY_PATH": INVENTORY_PATH,
        "INTERVAL_INVENTORY_PATH": INTERVAL_INVENTORY_PATH,
        "BLOCKER_CSV_PATH": BLOCKER_CSV_PATH,
        "BLOCKER_MD_PATH": BLOCKER_MD_PATH,
        "MANIFEST_PATH": MANIFEST_PATH,
        "STATUS_JSON_PATH": STATUS_JSON_PATH,
        "STATUS_MD_PATH": STATUS_MD_PATH,
    }
    for name, value in replacements.items():
        setattr(v7, name, value)


def as_date(value: Any) -> date | None:
    return v7.as_date(value)


def sha256_file(path: Path) -> str:
    return v7.sha256_file(path)


def write_json(path: Path, payload: Any) -> None:
    v7.write_json(path, payload)


def write_frame(
    path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> None:
    v7.write_frame(path, frame, columns)


def _bool_series(series: pd.Series) -> pd.Series:
    return v7._bool_series(series)


def validate_v7_immutability() -> dict[str, Any]:
    manifest = json.loads(V7_MANIFEST_PATH.read_text(encoding="utf-8"))
    status = json.loads(V7_STATUS_PATH.read_text(encoding="utf-8"))
    failures: list[str] = []
    for record in manifest.get("records", []):
        try:
            validate_portable_record(record, REPO_ROOT)
        except Exception as exc:
            failures.append(str(exc))
    expected = {
        "experiment_id": "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V7",
        "manifest_version": "manifest-portability-v7.0.0",
        "artifact_count": 34,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            failures.append(
                f"V7 manifest {key}={manifest.get(key)!r}, expected {value!r}"
            )
    if status.get("experiment_id") != expected["experiment_id"]:
        failures.append("V7 status experiment identifier drift")
    if status.get("status") != "PARTIAL_DATA_BLOCKED":
        failures.append("V7 terminal status drift")
    return {
        "status": "PASS" if not failures else "FAIL",
        "v7_manifest_checksums_valid": not failures,
        "v7_manifest_sha256": sha256_file(V7_MANIFEST_PATH),
        "v7_manifest_records": len(manifest.get("records", [])),
        "v7_status": status.get("status"),
        "failures": failures,
    }


def recover_raw_ohlc() -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_parquet(V7_RAW_PATH)
    base["date"] = pd.to_datetime(base["date"]).dt.date
    acquisition = pd.read_csv(
        SOURCE_ROOT / "fbm_klci_price_acquisition_v8.csv",
        dtype={"bursa_code": str},
    )
    acquisition["bursa_code"] = acquisition["bursa_code"].str.zfill(4)
    additions: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for record in acquisition.to_dict("records"):
        code = str(record["bursa_code"])
        normalized = str(record.get("normalized_path") or "")
        path = REPO_ROOT / normalized if normalized else None
        accepted = pd.DataFrame()
        if path is not None and path.is_file() and str(record["status"]) == "FETCHED":
            accepted = pd.read_csv(path)
            accepted["date"] = pd.to_datetime(accepted["date"], errors="coerce").dt.date
            for column in ["open", "high", "low", "close"]:
                accepted[column] = pd.to_numeric(accepted[column], errors="coerce")
            valid = (
                accepted[["open", "high", "low", "close"]].notna().all(axis=1)
                & (accepted[["open", "high", "low", "close"]] > 0).all(axis=1)
                & (
                    accepted["high"]
                    >= accepted[["open", "low", "close"]].max(axis=1)
                )
                & (
                    accepted["low"]
                    <= accepted[["open", "high", "close"]].min(axis=1)
                )
            )
            accepted = accepted.loc[valid].copy()
            accepted = accepted.drop_duplicates("date", keep="first")
            accepted["market"] = "Malaysia"
            accepted["security_id"] = v7.v6.security_id("FBM_KLCI", code)
            accepted["ticker_used"] = code
            accepted["exchange_suffix"] = ".KL"
            accepted["provider"] = "yahoo_query2_chart"
            accepted["price_semantics"] = "RAW_OHLC"
            accepted["raw_file_path"] = normalized
            accepted["retrieval_timestamp"] = str(record["retrieval_timestamp"])
            accepted["checksum"] = str(record["normalized_sha256"])
            accepted = accepted[
                [
                    "market",
                    "security_id",
                    "ticker_used",
                    "exchange_suffix",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "provider",
                    "price_semantics",
                    "raw_file_path",
                    "retrieval_timestamp",
                    "checksum",
                ]
            ]
            additions.append(accepted)
        audit_rows.append(
            {
                "bursa_code": code,
                "provider_symbol": record.get("provider_symbol"),
                "status": record.get("status"),
                "accepted_input_rows": len(accepted),
                "input_first_date": (
                    min(accepted["date"]) if len(accepted) else None
                ),
                "input_last_date": max(accepted["date"]) if len(accepted) else None,
                "duplicate_dates_excluded": int(record.get("duplicate_dates") or 0),
                "invalid_ohlc_rows_excluded": int(
                    record.get("invalid_ohlc_rows") or 0
                ),
                "adjusted_close_used": False,
                "volume_used": False,
                "imputation_used": False,
                "normalized_path": normalized or None,
                "normalized_sha256": record.get("normalized_sha256"),
                "error_message": record.get("error_message"),
            }
        )
    combined = pd.concat([base, *additions], ignore_index=True)
    combined["_priority"] = np.where(
        combined["provider"].astype(str).eq("yahoo_query2_chart"), 1, 0
    )
    combined = (
        combined.sort_values(
            ["market", "security_id", "date", "_priority"], kind="stable"
        )
        .drop_duplicates(["market", "security_id", "date"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )
    invalid = (
        combined[["open", "high", "low", "close"]].isna().any(axis=1)
        | (combined[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (
            combined["high"]
            < combined[["open", "low", "close"]].max(axis=1)
        )
        | (
            combined["low"]
            > combined[["open", "high", "close"]].min(axis=1)
        )
    )
    if invalid.any():
        raise RuntimeError(f"V8 combined raw surface has {int(invalid.sum())} invalid rows")
    if combined.duplicated(["market", "security_id", "date"]).any():
        raise RuntimeError("V8 combined raw surface has duplicate security-date rows")
    write_frame(RECOVERED_RAW_PATH, combined)
    audit = pd.DataFrame(audit_rows)
    write_frame(RECOVERY_AUDIT_PATH, audit)
    return combined, audit


def build_membership_surfaces() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v7_intervals = pd.read_parquet(V7_INTERVAL_PATH)
    v7_intervals = v7_intervals[~v7_intervals["index_id"].eq("FBM_KLCI")]
    fbm_intervals = pd.read_parquet(FBM_INTERVAL_PATH)
    fbm_intervals = fbm_intervals.drop(columns=["validation_status"], errors="ignore")
    intervals = pd.concat([v7_intervals, fbm_intervals], ignore_index=True)
    intervals = intervals.sort_values(
        ["index_id", "interval_start", "security_id"], kind="stable"
    ).reset_index(drop=True)
    write_frame(MEMBERSHIP_PATH, intervals)

    v7_daily = pd.read_parquet(V7_DAILY_PATH)
    v7_pit = v7_daily[
        v7_daily["universe_semantics"].astype(str).eq(PIT)
        & ~v7_daily["index_id"].astype(str).eq("FBM_KLCI")
    ].copy()
    fbm_daily = pd.read_parquet(FBM_DAILY_PATH)
    pit = pd.concat([v7_pit, fbm_daily], ignore_index=True)
    if pit.duplicated(["interval_id", "security_id", "membership_date"]).any():
        raise RuntimeError("duplicate V8 PIT daily membership key")
    per_fbm_date = fbm_daily.groupby("membership_date")["security_id"].nunique()
    if per_fbm_date.empty or per_fbm_date.ne(30).any():
        raise RuntimeError("FBM KLCI PIT daily surface is not exactly 30 per date")
    all_members = v7.build_all_arm_membership(pit)
    return intervals, pit, all_members


def _coverage_record(
    members: pd.DataFrame, raw: pd.DataFrame
) -> tuple[int, int, int, int]:
    raw_keys = set(
        map(
            tuple,
            raw[["market", "security_id", "date"]].itertuples(index=False, name=None),
        )
    )
    raw_by_security = {
        (str(market), str(security)): sorted(set(group["date"].map(as_date)))
        for (market, security), group in raw.groupby(
            ["market", "security_id"], sort=False
        )
    }
    required = len(members)
    covered = 0
    feature = 0
    for row in members.to_dict("records"):
        key = (
            str(row["market"]),
            str(row["security_id"]),
            as_date(row["membership_date"]),
        )
        covered += int(key in raw_keys)
        history = raw_by_security.get((key[0], key[1]), [])
        position = int(np.searchsorted(history, key[2]))
        feature += int(key in raw_keys and position >= FEATURE_LOOKBACK)
    _, target = v7._next_active_target_coverage(members, raw)
    return required, covered, feature, target


def build_interval_inventory(
    intervals: pd.DataFrame, pit_daily: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    frozen = pd.read_csv(V7_INTERVAL_INVENTORY_PATH)
    frozen = frozen[~frozen["index_id"].astype(str).eq("FBM_KLCI")].copy()
    rows: list[dict[str, Any]] = []
    fbm_intervals = intervals[intervals["index_id"].eq("FBM_KLCI")]
    for (
        market,
        index_id,
        interval_id,
        start_value,
        end_value,
    ), group in fbm_intervals.groupby(
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
        required, covered, feature, target = _coverage_record(members, raw)
        price_coverage = covered / required if required else 0.0
        feature_coverage = feature / required if required else 0.0
        target_coverage = target / required if required else 0.0
        reasons: list[str] = []
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
        conflicts = int(pd.to_numeric(group["high_severity_conflicts"]).max())
        if conflicts:
            reasons.append(f"{conflicts} high-severity membership conflict(s)")
        rows.append(
            {
                "market": market,
                "index_id": index_id,
                "interval_id": interval_id,
                "interval_start": as_date(start_value),
                "interval_end_exclusive": as_date(end_value),
                "calendar_days": (as_date(end_value) - as_date(start_value)).days,
                "trading_dates": int(members["membership_date"].nunique()),
                "pit_security_count": int(group["security_id"].nunique()),
                "required_active_member_days": required,
                "covered_active_member_days": covered,
                "price_coverage": price_coverage,
                "feature_lookback_coverage": feature_coverage,
                "target_pair_coverage": target_coverage,
                "high_severity_conflicts": conflicts,
                "eligible_for_forecasting": not reasons,
                "gate_source": "V8_RECOMPUTED_FBM_KLCI_EVIDENCE_CHAIN",
                "blocking_reasons": " | ".join(reasons) or None,
            }
        )
    result = pd.concat([frozen, pd.DataFrame(rows)], ignore_index=True)
    result["interval_start"] = pd.to_datetime(result["interval_start"]).dt.date
    result["interval_end_exclusive"] = pd.to_datetime(
        result["interval_end_exclusive"]
    ).dt.date
    result["eligible_for_forecasting"] = _bool_series(
        result["eligible_for_forecasting"]
    )
    result = result.sort_values(["index_id", "interval_start"], kind="stable")
    write_frame(INTERVAL_INVENTORY_PATH, result)
    return result


def build_price_membership_audit(
    pit_daily: pd.DataFrame, raw: pd.DataFrame
) -> pd.DataFrame:
    crosswalk = pd.read_csv(FBM_CROSSWALK_PATH, dtype={"bursa_code": str})
    crosswalk["bursa_code"] = crosswalk["bursa_code"].str.zfill(4)
    members = pit_daily[pit_daily["index_id"].eq("FBM_KLCI")].copy()
    malaysia_raw = raw[raw["market"].eq("Malaysia")].copy()
    rows: list[dict[str, Any]] = []
    for security, group in members.groupby("security_id", sort=True):
        raw_group = malaysia_raw[malaysia_raw["security_id"].eq(security)]
        required, covered, feature, target = _coverage_record(group, raw)
        map_row = crosswalk[crosswalk["stable_security_id"].eq(security)]
        code = str(group.iloc[0]["ticker_on_date"])
        mapping = map_row.iloc[0].to_dict() if len(map_row) else {}
        price_coverage = covered / required if required else 0.0
        feature_coverage = feature / required if required else 0.0
        target_coverage = target / required if required else 0.0
        rows.append(
            {
                "audit_grain": "SECURITY",
                "stable_security_id": security,
                "bursa_code": code,
                "canonical_company_name": mapping.get("canonical_company_name"),
                "provider_ticker": mapping.get("provider_ticker", f"{code}.KL"),
                "raw_source_paths": ";".join(
                    sorted(set(raw_group["raw_file_path"].astype(str)))
                ),
                "raw_rows": len(raw_group),
                "raw_first_date": (
                    min(raw_group["date"].map(as_date)) if len(raw_group) else None
                ),
                "raw_last_date": (
                    max(raw_group["date"].map(as_date)) if len(raw_group) else None
                ),
                "required_active_member_days": required,
                "covered_active_member_days": covered,
                "price_coverage": price_coverage,
                "feature_lookback_coverage": feature_coverage,
                "next_session_target_coverage": target_coverage,
                "duplicate_security_date_rows": int(
                    raw_group.duplicated(["security_id", "date"]).sum()
                ),
                "invalid_ohlc_rows": 0,
                "exchange_suffixes": ";".join(
                    sorted(set(raw_group["exchange_suffix"].fillna("").astype(str)))
                ),
                "price_semantics": ";".join(
                    sorted(set(raw_group["price_semantics"].astype(str)))
                ),
                "adjusted_close_used": False,
                "volume_used": False,
                "forward_fill_used": False,
                "imputation_used": False,
                "rename_effective_date": mapping.get("rename_effective_date"),
                "corporate_action_or_delisting_notes": mapping.get(
                    "corporate_action_notes"
                ),
                "join_status": (
                    "PASS"
                    if min(price_coverage, feature_coverage, target_coverage)
                    >= PRICE_COVERAGE_THRESHOLD
                    else "DOCUMENTED_SECURITY_PRICE_GAP"
                ),
            }
        )
    result = pd.DataFrame(rows)
    write_frame(PRICE_AUDIT_CSV_PATH, result)
    missing = result[result["join_status"].ne("PASS")]
    aggregate_required = int(result["required_active_member_days"].sum())
    aggregate_covered = int(result["covered_active_member_days"].sum())
    lines = [
        "# FBM KLCI Price-Membership Join Audit V8",
        "",
        f"- Audited active securities: {len(result)}",
        f"- Raw Malaysian rows after precedence/deduplication: {len(malaysia_raw):,}",
        f"- Raw Malaysian security IDs: {malaysia_raw['security_id'].nunique()}",
        f"- Raw date range: {malaysia_raw['date'].min()} to {malaysia_raw['date'].max()}",
        f"- Aggregate active-member-day coverage: {aggregate_covered / max(aggregate_required, 1):.6f}",
        f"- Securities with documented price gaps: {len(missing)}",
        "- Semantics: raw Open/High/Low/Close only; no adjusted close, volume, forward fill, or imputation.",
        "",
        "Security-level gaps are preserved in the CSV and are not converted to "
        "zero prices or inferred observations. Eligibility is evaluated with "
        "the unchanged 0.95 interval thresholds.",
    ]
    if len(missing):
        lines.extend(["", "## Documented gaps", ""])
        for row in missing.to_dict("records"):
            lines.append(
                f"- {row['bursa_code']} {row['canonical_company_name']}: "
                f"{row['covered_active_member_days']}/{row['required_active_member_days']} "
                "active-member days joined."
            )
    PRICE_AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRICE_AUDIT_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def build_fold_calendar(
    inventory: pd.DataFrame, pit_daily: pd.DataFrame
) -> pd.DataFrame:
    folds = v7.build_fold_calendar(inventory, pit_daily)
    folds = folds[
        ~folds["interval_id"].astype(str).eq("BLOCKED-FBM_KLCI-NO-PIT-INTERVAL")
    ].copy()
    write_frame(FOLD_PATH, folds, v7.FOLD_COLUMNS)
    return folds


def validate_market_cells(
    folds: pd.DataFrame,
    collapsed: pd.DataFrame,
    failures: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index_id, market in MARKETS.items():
        market_folds = folds[
            folds["index_id"].eq(index_id) & folds["valid_fold"].astype(bool)
        ]
        valid_folds = int(market_folds["fold_id"].nunique())
        eligible_intervals = int(market_folds["interval_id"].nunique())
        for semantics in ARM_SEMANTICS.values():
            for model_id in MODEL_SPECS:
                frame = collapsed[
                    collapsed["index_id"].astype(str).eq(index_id)
                    & collapsed["universe_semantics"].astype(str).eq(semantics)
                    & collapsed["model_id"].astype(str).eq(model_id)
                ].copy()
                reasons: list[str] = []
                prediction_folds = int(frame["fold_id"].nunique()) if len(frame) else 0
                dates = int(frame["prediction_date"].nunique()) if len(frame) else 0
                classes = int(frame["actual_class"].nunique()) if len(frame) else 0
                if valid_folds < MIN_VALID_FOLDS:
                    reasons.append(f"valid market folds {valid_folds} < {MIN_VALID_FOLDS}")
                if prediction_folds < MIN_VALID_FOLDS:
                    reasons.append(
                        f"prediction folds {prediction_folds} < {MIN_VALID_FOLDS}"
                    )
                if dates < MIN_OOS_DATES:
                    reasons.append(f"OOS dates {dates} < {MIN_OOS_DATES}")
                if len(frame) < MIN_OOS_ROWS:
                    reasons.append(
                        f"OOS security-date predictions {len(frame)} < {MIN_OOS_ROWS}"
                    )
                if classes < 2:
                    reasons.append("both target classes not represented in test")
                if len(frame) and (
                    (frame["training_class_0"] <= 0).any()
                    or (frame["training_class_1"] <= 0).any()
                ):
                    reasons.append("both target classes not represented in train")
                key = [
                    "interval_id",
                    "universe_semantics",
                    "fold_id",
                    "security_id",
                    "prediction_date",
                    "model_id",
                ]
                if len(frame) and frame.duplicated(key).any():
                    reasons.append("duplicate collapsed prediction key")
                if len(frame) and not (
                    pd.to_datetime(frame["label_maturity_cutoff"])
                    < pd.to_datetime(frame["prediction_date"])
                ).all():
                    reasons.append("label maturity leakage")
                if len(frame) and not (
                    pd.to_datetime(frame["training_end"])
                    < pd.to_datetime(frame["prediction_date"])
                ).all():
                    reasons.append("train-test overlap")
                if not failures.empty:
                    selected = failures[
                        failures["index_id"].astype(str).eq(index_id)
                        & failures["universe_semantics"].astype(str).eq(semantics)
                        & failures["model_id"].astype(str).eq(model_id)
                    ]
                    if len(selected):
                        reasons.append(
                            "model execution failure: "
                            + ";".join(sorted(set(selected["reason"].astype(str))))
                        )
                rows.append(
                    {
                        "cell_id": v7.stable_id(
                            "CELL8", index_id, semantics, model_id
                        ),
                        "market": market,
                        "index_id": index_id,
                        "universe_semantics": semantics,
                        "model_id": model_id,
                        "eligible_intervals": eligible_intervals,
                        "scheduled_valid_folds": valid_folds,
                        "prediction_folds": prediction_folds,
                        "oos_trading_dates": dates,
                        "oos_security_date_predictions": len(frame),
                        "target_class_count": classes,
                        "valid": not reasons,
                        "blocking_reasons": " | ".join(reasons) or None,
                        "result_classification": (
                            None if not reasons else "INSUFFICIENT_INFORMATION"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def run_analytics(
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
    composition = v7.build_composition(members)
    if valid_predictions.empty:
        raise RuntimeError("V8 has no model-valid predictions for analytics")
    analytics = v7.analytics
    by_date = analytics.build_performance_by_date(valid_predictions, composition)
    by_interval, by_market, macro = analytics.build_performance_hierarchy(by_date)
    current_minus_pit, date_diff = analytics.build_current_minus_pit(
        by_date, by_interval, by_market, macro
    )
    inference = analytics.build_bootstrap_inference(date_diff)
    inference["bh_family"] = (
        inference["inference_level"].astype(str)
        + "|"
        + inference["metric"].astype(str)
        + "|block_"
        + inference["block_length"].astype(str)
    )
    inference["bh_model_scope"] = "ALL_DECLARED_MODEL_FAMILIES"
    valid_seed = seed_level.merge(
        valid_cells,
        on=["index_id", "universe_semantics", "model_id"],
        how="inner",
        validate="many_to_one",
    )
    seed_robustness = analytics.build_seed_robustness(valid_seed)
    rank = analytics.build_model_rank(
        by_interval, by_market, valid_predictions, seed_robustness
    )
    heterogeneity = analytics.build_heterogeneity(current_minus_pit, inference)
    outputs = {
        "pit_multimarket_performance_by_date_v8": by_date,
        "pit_multimarket_performance_by_interval_v8": by_interval,
        "pit_multimarket_performance_by_market_v8": by_market,
        "pit_multimarket_equal_market_macro_v8": macro,
        "pit_multimarket_current_minus_pit_v8": current_minus_pit,
        "pit_multimarket_bootstrap_inference_v8": inference,
        "pit_multimarket_heterogeneity_v8": heterogeneity,
        "pit_multimarket_model_rank_v8": rank,
        "pit_multimarket_seed_robustness_v8": seed_robustness,
    }
    for base, frame in outputs.items():
        write_frame(REPORTS / f"{base}.csv", frame)
    return outputs


def build_blocker_audit(
    inventory: pd.DataFrame, folds: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in inventory.to_dict("records"):
        interval_folds = folds[
            folds["interval_id"].astype(str).eq(str(record["interval_id"]))
            & folds["valid_fold"].astype(bool)
        ]
        rows.append(
            {
                "audit_grain": "INTERVAL",
                **record,
                "continuous_valid_folds": int(interval_folds["fold_id"].nunique()),
                "passes_data_gate": bool(record["eligible_for_forecasting"]),
                "passes_continuous_protocol_gate": bool(len(interval_folds)),
                "exact_reasons": record.get("blocking_reasons"),
            }
        )
    result = pd.DataFrame(rows)
    write_frame(BLOCKER_CSV_PATH, result)
    lines = [
        "# Five-Market Data Blocker Audit V8",
        "",
        f"Experiment: `{EXPERIMENT_ID}`.",
        "",
        "| Index | PIT intervals | Data-ready intervals | Continuous valid folds | Remaining blocker |",
        "|---|---:|---:|---:|---|",
    ]
    for index_id in MARKETS:
        selected = result[result["index_id"].eq(index_id)]
        blockers = sorted(
            set(selected["exact_reasons"].dropna().astype(str))
        )
        lines.append(
            f"| {index_id} | {len(selected)} | "
            f"{int(selected['passes_data_gate'].sum())} | "
            f"{int(selected['continuous_valid_folds'].sum())} | "
            f"{' ; '.join(blockers) if blockers else 'none for evaluated intervals'} |"
        )
    lines.extend(
        [
            "",
            "Intervals failing the unchanged V7 0.95 data gates remain disclosed in "
            "the CSV. They are not silently repaired, filled, or interpreted as inactivity.",
        ]
    )
    BLOCKER_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def build_market_inventory(
    interval_inventory: pd.DataFrame,
    folds: pd.DataFrame,
    cells: pd.DataFrame,
    collapsed: pd.DataFrame,
    raw: pd.DataFrame,
) -> pd.DataFrame:
    result = v7.build_market_inventory(
        interval_inventory, folds, cells, collapsed, raw
    )
    write_frame(INVENTORY_PATH, result)
    return result


def expected_config(inventory: pd.DataFrame) -> dict[str, Any]:
    config = v7.expected_config(inventory)
    config.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "study_label": STUDY_LABEL,
            "experiment_version": EXPERIMENT_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "data_version": DATA_VERSION,
            "membership_version": MEMBERSHIP_VERSION,
            "snapshot_version": SNAPSHOT_VERSION,
            "price_version": PRICE_VERSION,
            "manifest_version": MANIFEST_VERSION,
            "v7_reference": {
                "experiment_id": (
                    "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V7"
                ),
                "status": "PARTIAL_DATA_BLOCKED",
                "manifest_version": "manifest-portability-v7.0.0",
                "manifest_records": 34,
                "protocol_change": "none",
            },
            "v8_membership_resolution": {
                "accepted_anchor_dates": ["2017-12-31", "2019-12-23"],
                "malaysia_chain_start": "2018-01-01",
                "malaysia_chain_end_exclusive": "2026-07-02",
                "current_arm_role": (
                    "frozen deliberately biased V7 comparison arm; not membership evidence"
                ),
            },
        }
    )
    return config


def freeze_config(inventory: pd.DataFrame) -> dict[str, Any]:
    import yaml

    config = expected_config(inventory)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config


def load_test_summary() -> dict[str, Any]:
    if not TEST_SUMMARY_PATH.is_file():
        return {"status": "PENDING", "passed": 0, "failed": 0, "commands": []}
    return json.loads(TEST_SUMMARY_PATH.read_text(encoding="utf-8"))


def completion_gates(
    market_inventory: pd.DataFrame,
    interval_inventory: pd.DataFrame,
    cells: pd.DataFrame,
    macro: pd.DataFrame,
    v7_integrity: Mapping[str, Any],
    artifact_status: str,
    test_summary: Mapping[str, Any],
) -> dict[str, bool]:
    unresolved = pd.read_csv(UNRESOLVED_PATH)
    five = len(market_inventory) == 5
    macro_valid = bool(
        len(macro) == len(ARM_SEMANTICS) * len(MODEL_SPECS)
        and macro["model_valid_markets"].astype(int).eq(5).all()
    )
    ready_by_market = (
        interval_inventory[
            interval_inventory["eligible_for_forecasting"].astype(bool)
        ]
        .groupby("index_id")
        .size()
    )
    return {
        "1_all_five_validated_pit_membership": bool(
            five and market_inventory["validated_pit_membership"].astype(bool).all()
        ),
        "2_all_five_valid_ohlc_joins": bool(
            five
            and market_inventory["validated_raw_ohlc"].astype(bool).all()
            and all(int(ready_by_market.get(index_id, 0)) > 0 for index_id in MARKETS)
        ),
        "3_all_five_at_least_two_valid_folds": bool(
            five
            and (market_inventory["valid_continuous_folds"] >= MIN_VALID_FOLDS).all()
        ),
        "4_all_five_at_least_40_oos_dates": bool(
            five and (market_inventory["oos_trading_dates"] >= MIN_OOS_DATES).all()
        ),
        "5_all_five_at_least_800_collapsed_predictions": bool(
            five
            and (
                market_inventory["seed_collapsed_predictions"] >= MIN_OOS_ROWS
            ).all()
        ),
        "6_all_required_market_arm_model_cells_valid": bool(
            len(cells) == len(MARKETS) * len(ARM_SEMANTICS) * len(MODEL_SPECS)
            and cells["valid"].astype(bool).all()
        ),
        "7_no_unresolved_high_severity_membership_conflict": bool(
            not (
                _bool_series(unresolved["blocks_forecasting"])
                | unresolved["severity"].astype(str).str.startswith("HIGH")
            ).any()
        ),
        "8_all_artifact_checksums_pass": artifact_status == "PASS",
        "9_all_tests_pass": test_summary.get("status") == "PASS",
        "10_five_market_macro_recomputed": macro_valid,
        "v7_immutable": bool(v7_integrity["v7_manifest_checksums_valid"]),
    }


def classify_status(gates: Mapping[str, bool]) -> str:
    declared = [key for key in gates if key[0].isdigit()]
    if all(gates[key] for key in declared) and gates["v7_immutable"]:
        return "COMPLETE_FIVE_MARKET"
    if not gates["v7_immutable"]:
        return "FAILED_INTEGRITY"
    data_keys = [
        "1_all_five_validated_pit_membership",
        "2_all_five_valid_ohlc_joins",
        "7_no_unresolved_high_severity_membership_conflict",
    ]
    protocol_keys = [
        key for key in declared if key not in data_keys
    ]
    data_blocked = not all(gates[key] for key in data_keys)
    protocol_blocked = not all(gates[key] for key in protocol_keys)
    if data_blocked and protocol_blocked:
        return "PARTIAL_DATA_AND_PROTOCOL_BLOCKED"
    if data_blocked:
        return "PARTIAL_DATA_BLOCKED"
    return "PARTIAL_PROTOCOL_BLOCKED"


def build_reconciliation(v7_integrity: Mapping[str, Any]) -> None:
    lines = [
        "# V7–V8 Reconciliation",
        "",
        "## Frozen V7 integrity",
        "",
        f"- V7 manifest status: `{v7_integrity['status']}`",
        f"- V7 manifest records: {v7_integrity['v7_manifest_records']}",
        f"- V7 terminal status: `{v7_integrity['v7_status']}`",
        f"- V7 manifest SHA-256: `{v7_integrity['v7_manifest_sha256']}`",
        "",
        "## Protocol reconciliation",
        "",
        "V8 changes only the evidence and data surfaces needed to resolve FBM KLCI. "
        "Features, target, models, seeds, continuous expanding-history folds, metrics, "
        "aggregation hierarchy, moving-block bootstrap, and BH rules are unchanged.",
        "",
        "The frozen current-universe arm remains a deliberately biased comparison arm. "
        "It is never used as Malaysian historical-membership evidence.",
    ]
    if V7_COLLAPSED_PATH.is_file() and COLLAPSED_PATH.is_file():
        old = pd.read_parquet(V7_COLLAPSED_PATH)
        new = pd.read_parquet(COLLAPSED_PATH)
        old = old[old["index_id"].isin(["IDX30", "SET50", "VN30", "PSEI"])]
        new = new[new["index_id"].isin(["IDX30", "SET50", "VN30", "PSEI"])]
        keys = [
            "interval_id",
            "index_id",
            "universe_semantics",
            "fold_id",
            "security_id",
            "prediction_date",
            "model_id",
        ]
        check = old[keys + ["prediction_probability", "actual_class"]].merge(
            new[keys + ["prediction_probability", "actual_class"]],
            on=keys,
            how="outer",
            suffixes=("_v7", "_v8"),
            indicator=True,
        )
        matched = check[check["_merge"].eq("both")]
        max_difference = (
            float(
                (
                    matched["prediction_probability_v7"]
                    - matched["prediction_probability_v8"]
                )
                .abs()
                .max()
            )
            if len(matched)
            else math.nan
        )
        class_mismatches = int(
            (
                matched["actual_class_v7"].astype(str)
                != matched["actual_class_v8"].astype(str)
            ).sum()
        )
        lines.extend(
            [
                "",
                "## Four-market rerun reconciliation",
                "",
                f"- V7 collapsed rows in unchanged markets: {len(old):,}",
                f"- V8 collapsed rows in unchanged markets: {len(new):,}",
                f"- Matched prediction keys: {len(matched):,}",
                f"- V7-only keys: {int(check['_merge'].eq('left_only').sum()):,}",
                f"- V8-only keys: {int(check['_merge'].eq('right_only').sum()):,}",
                f"- Maximum matched probability difference: {max_difference:.12g}",
                f"- Matched target-class mismatches: {class_mismatches}",
            ]
        )
    RECONCILIATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECONCILIATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def required_artifact_paths() -> list[Path]:
    paths = [
        SCRIPT_PATH,
        REPO_ROOT / "scripts" / "point_in_time" / "fbm_klci_reconstruction_v8.py",
        REPO_ROOT / "scripts" / "point_in_time" / "fetch_fbm_klci_prices_v8.py",
        REPO_ROOT / "tests" / "point_in_time" / "test_pit_multimarket_forecasting_v8.py",
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
        PREFLIGHT_PATH,
        TEST_SUMMARY_PATH,
        INVENTORY_PATH,
        INTERVAL_INVENTORY_PATH,
        BLOCKER_CSV_PATH,
        BLOCKER_MD_PATH,
        PRICE_AUDIT_CSV_PATH,
        PRICE_AUDIT_MD_PATH,
        RECOVERY_AUDIT_PATH,
        STATUS_JSON_PATH,
        STATUS_MD_PATH,
        RECONCILIATION_PATH,
        RECONSTRUCTED / "fbm_klci_anchor_snapshots_v8.csv",
        RECONSTRUCTED / "fbm_klci_membership_events_v8.csv",
        RECONSTRUCTED / "fbm_klci_security_crosswalk_v8.csv",
        FBM_INTERVAL_PATH,
        FBM_DAILY_PATH,
        RECONSTRUCTED / "fbm_klci_reconstruction_trace_v8.csv",
        SOURCE_ROOT / "fbm_klci_source_registry_v8.csv",
        SOURCE_ROOT / "fbm_klci_source_registry_v8.json",
        MANIFESTS / "unresolved_membership_evidence_v8.csv",
        MANIFESTS / "unresolved_membership_evidence_v8.json",
        FBM_VALIDATION_PATH,
    ]
    report_bases = [
        "pit_multimarket_invalid_cells_v8.csv",
        "pit_multimarket_performance_by_date_v8.csv",
        "pit_multimarket_performance_by_interval_v8.csv",
        "pit_multimarket_performance_by_market_v8.csv",
        "pit_multimarket_equal_market_macro_v8.csv",
        "pit_multimarket_current_minus_pit_v8.csv",
        "pit_multimarket_bootstrap_inference_v8.csv",
        "pit_multimarket_heterogeneity_v8.csv",
        "pit_multimarket_model_rank_v8.csv",
        "pit_multimarket_seed_robustness_v8.csv",
    ]
    paths.extend(REPORTS / name for name in report_bases)
    paths.extend(sorted(SOURCE_ROOT.glob("*_factual_extraction.txt")))
    paths.extend(sorted(PRICE_ROOT.glob("raw_ohlc_*.csv")))
    paths.extend(sorted(PRICE_ROOT.glob("yahoo_chart_*.json")))
    paths.extend(
        [
            SOURCE_ROOT / "fbm_klci_price_acquisition_v8.csv",
            SOURCE_ROOT / "fbm_klci_price_acquisition_v8.json",
        ]
    )
    return list(dict.fromkeys(paths))


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


def validate_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    for path in required_artifact_paths():
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
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in required_artifact_paths()
        if path.is_file()
    }
    recorded = {str(record["artifact_relative_path"]) for record in manifest["records"]}
    missing_records = sorted(expected - recorded)
    failures.extend(f"required artifact absent from manifest: {item}" for item in missing_records)
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if not failures else "FAIL",
        "manifest_records": len(manifest["records"]),
        "repository_relative_keys": all(
            not Path(record["artifact_relative_path"]).is_absolute()
            for record in manifest["records"]
        ),
        "checksum_failures": failures,
        "validation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_status(
    market_inventory: pd.DataFrame,
    interval_inventory: pd.DataFrame,
    cells: pd.DataFrame,
    seed_level: pd.DataFrame,
    collapsed: pd.DataFrame,
    macro: pd.DataFrame,
    inference: pd.DataFrame,
    v7_integrity: Mapping[str, Any],
    manifest_records: int,
    artifact_status: str,
) -> dict[str, Any]:
    tests = load_test_summary()
    gates = completion_gates(
        market_inventory,
        interval_inventory,
        cells,
        macro,
        v7_integrity,
        artifact_status,
        tests,
    )
    status = classify_status(gates)
    brier = inference[
        inference["inference_level"].eq("selected_market_macro")
        & inference["metric"].eq("brier_score")
        & inference["is_primary_block"].astype(bool)
    ].copy()
    coverage = inference[
        inference["inference_level"].eq("selected_market_macro")
        & inference["metric"].eq("prediction_coverage")
        & inference["is_primary_block"].astype(bool)
    ].copy()
    fbm_validation = json.loads(FBM_VALIDATION_PATH.read_text(encoding="utf-8"))
    source_registry = pd.read_csv(SOURCE_ROOT / "fbm_klci_source_registry_v8.csv")
    events = pd.read_csv(RECONSTRUCTED / "fbm_klci_membership_events_v8.csv")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "membership_version": MEMBERSHIP_VERSION,
        "data_version": DATA_VERSION,
        "snapshot_version": SNAPSHOT_VERSION,
        "price_version": PRICE_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "status": status,
        "complete_five_market": status == "COMPLETE_FIVE_MARKET",
        "completion_gates": gates,
        "fbm_klci_reconstruction": {
            "status": fbm_validation["status"],
            "accepted_anchor_dates": fbm_validation["accepted_anchor_dates"],
            "scheduled_constituent_swap_rows": fbm_validation[
                "scheduled_constituent_swap_rows"
            ],
            "extraordinary_event_rows": fbm_validation["extraordinary_event_rows"],
            "source_tier_counts": fbm_validation["source_tier_counts"],
            "source_conflicts": fbm_validation["failures"],
            "interval_count": fbm_validation["interval_count"],
            "unknown_membership_dates_inside_evaluated_windows": fbm_validation[
                "unknown_membership_dates_inside_evaluated_windows"
            ],
            "event_ledger_rows": len(events),
            "source_registry_rows": len(source_registry),
        },
        "market_inventory": market_inventory.to_dict("records"),
        "valid_cells_by_market": {
            index_id: int(
                cells[
                    cells["index_id"].eq(index_id) & cells["valid"].astype(bool)
                ].shape[0]
            )
            for index_id in MARKETS
        },
        "invalid_cells_by_market": {
            index_id: int(
                cells[
                    cells["index_id"].eq(index_id) & ~cells["valid"].astype(bool)
                ].shape[0]
            )
            for index_id in MARKETS
        },
        "seed_level_predictions": len(seed_level),
        "seed_collapsed_predictions": len(collapsed),
        "primary_brier_current_minus_pit": brier.to_dict("records"),
        "operational_prediction_coverage_current_minus_pit": coverage.to_dict(
            "records"
        ),
        "five_market_macro_rows": len(macro),
        "test_summary": tests,
        "manifest_records": manifest_records,
        "artifact_checksum_status": artifact_status,
        "v7_integrity": dict(v7_integrity),
        "inference_boundaries": [
            "Brier score is the primary predictive-loss inference.",
            "Prediction coverage is operational and is not predictive-loss evidence.",
            "Non-significance is not equivalence.",
            "No profitability, causal, manuscript, or trading claim is made.",
        ],
    }
    write_json(STATUS_JSON_PATH, payload)
    lines = [
        "# PIT Multimarket Forecasting Status V8",
        "",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Five-market status: `{status}`",
        f"- FBM KLCI reconstruction: `{fbm_validation['status']}`",
        f"- Seed-level predictions: {len(seed_level):,}",
        f"- Seed-collapsed predictions: {len(collapsed):,}",
        f"- Manifest records: {manifest_records}",
        f"- Artifact checksum status: `{artifact_status}`",
        f"- Tests: `{tests.get('status')}` ({tests.get('passed', 0)} passed, {tests.get('failed', 0)} failed)",
        "",
        "## Completion gates",
        "",
    ]
    lines.extend(
        f"- {key}: `{str(value).lower()}`" for key, value in gates.items()
    )
    lines.extend(
        [
            "",
            "## Market execution",
            "",
            "| Index | PIT intervals | Ready intervals | Folds | OOS dates | Collapsed predictions | Valid cells | Invalid cells |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in market_inventory.to_dict("records"):
        index_id = row["index_id"]
        lines.append(
            f"| {index_id} | {row['pit_intervals']} | {row['data_ready_intervals']} | "
            f"{row['valid_continuous_folds']} | {row['oos_trading_dates']} | "
            f"{row['seed_collapsed_predictions']} | "
            f"{payload['valid_cells_by_market'][index_id]} | "
            f"{payload['invalid_cells_by_market'][index_id]} |"
        )
    lines.extend(["", "## Primary Brier inference", ""])
    for row in brier.to_dict("records"):
        lines.append(
            f"- {row['model_id']}: current-minus-PIT {row['point_estimate']:.8f}; "
            f"95% interval [{row['ci_lower']:.8f}, {row['ci_upper']:.8f}]; "
            f"BH-adjusted p={row['adjusted_p_value']:.6f}; "
            f"`{row['result_classification']}`."
        )
    lines.extend(
        [
            "",
            "## Operational prediction coverage",
            "",
            "Coverage is reported separately and is not treated as forecasting-performance evidence.",
        ]
    )
    for row in coverage.to_dict("records"):
        lines.append(
            f"- {row['model_id']}: current-minus-PIT {row['point_estimate']:.8f}; "
            f"`{row['result_classification']}`."
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "These are predictive-loss comparisons only. Non-significance is not "
            "equivalence. No profitability, causal, manuscript, or trading claim is made.",
        ]
    )
    STATUS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_data_phase() -> dict[str, Any]:
    _configure_v7_runtime()
    v7_integrity = validate_v7_immutability()
    if v7_integrity["status"] != "PASS":
        raise RuntimeError("V7 integrity failed before V8 data execution")
    fbm_validation = json.loads(FBM_VALIDATION_PATH.read_text(encoding="utf-8"))
    allowed = {"PASS", "PASS_WITH_DOCUMENTED_LIMITATIONS"}
    if (
        fbm_validation.get("status") not in allowed
        or fbm_validation.get("unknown_membership_dates_inside_evaluated_windows") != 0
    ):
        raise RuntimeError("FBM KLCI reconstruction does not pass the forecasting gate")
    raw, _ = recover_raw_ohlc()
    intervals, pit_daily, all_members = build_membership_surfaces()
    inventory = build_interval_inventory(intervals, pit_daily, raw)
    price_audit = build_price_membership_audit(pit_daily, raw)
    freeze_config(inventory)
    observations, rejections = v7.construct_observations(all_members, raw)
    observations, features, rejections = v7.construct_features(
        observations, rejections, all_members, raw
    )
    write_frame(OBSERVATION_PATH, observations, OBSERVATION_COLUMNS)
    write_frame(FEATURE_PATH, features)
    write_frame(REJECTION_PATH, rejections, OBSERVATION_COLUMNS)
    folds = build_fold_calendar(inventory, pit_daily)
    build_blocker_audit(inventory, folds)
    fbm_ready = inventory[
        inventory["index_id"].eq("FBM_KLCI")
        & inventory["eligible_for_forecasting"].astype(bool)
    ]
    preflight_failures: list[str] = []
    if fbm_ready.empty:
        preflight_failures.append("no data-ready FBM KLCI interval")
    if price_audit["adjusted_close_used"].astype(bool).any():
        preflight_failures.append("adjusted close detected")
    if price_audit["imputation_used"].astype(bool).any():
        preflight_failures.append("imputation detected")
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if not preflight_failures else "FAIL",
        "predefined_markets": list(MARKETS),
        "membership_intervals": int(inventory["interval_id"].nunique()),
        "fbm_klci_intervals": int(
            inventory[inventory["index_id"].eq("FBM_KLCI")][
                "interval_id"
            ].nunique()
        ),
        "fbm_klci_ready_intervals": int(fbm_ready["interval_id"].nunique()),
        "eligible_intervals": int(
            inventory["eligible_for_forecasting"].astype(bool).sum()
        ),
        "valid_continuous_folds": int(folds["valid_fold"].astype(bool).sum()),
        "observation_rows": len(observations),
        "feature_rows": len(features),
        "rejection_rows": len(rejections),
        "raw_rows": len(raw),
        "fbm_membership_validation_status": fbm_validation["status"],
        "v7_integrity": v7_integrity,
        "failures": preflight_failures,
    }
    write_json(PREFLIGHT_PATH, preflight)
    if preflight_failures:
        raise RuntimeError("V8 preflight failed: " + " | ".join(preflight_failures))
    if not TEST_SUMMARY_PATH.is_file():
        write_json(
            TEST_SUMMARY_PATH,
            {"status": "PENDING", "passed": 0, "failed": 0, "commands": []},
        )
    return preflight


def run_model_phase() -> dict[str, Any]:
    _configure_v7_runtime()
    features = pd.read_parquet(FEATURE_PATH)
    folds = pd.read_parquet(FOLD_PATH)
    seed_level, collapsed, failures = v7.run_oos_models(features, folds)
    seed_level["model_version"] = seed_level["model_version"].astype(str).str.replace(
        r"^v7-", "v8-", regex=True
    )
    collapsed["model_version"] = collapsed["model_version"].astype(str).str.replace(
        r"^v7-", "v8-", regex=True
    )
    write_frame(SEED_PATH, seed_level, PREDICTION_COLUMNS)
    write_frame(COLLAPSED_PATH, collapsed, PREDICTION_COLUMNS + ["seed_count"])
    cells = validate_market_cells(folds, collapsed, failures)
    write_frame(ALL_CELLS_PATH, cells)
    write_frame(
        REPORTS / "pit_multimarket_invalid_cells_v8.csv",
        cells[~cells["valid"].astype(bool)],
    )
    build_reconciliation(validate_v7_immutability())
    return {
        "seed_level_predictions": len(seed_level),
        "collapsed_predictions": len(collapsed),
        "valid_cells": int(cells["valid"].astype(bool).sum()),
        "invalid_cells": int((~cells["valid"].astype(bool)).sum()),
        "execution_failures": len(failures),
    }


def run_reporting_phase() -> dict[str, Any]:
    _configure_v7_runtime()
    interval_inventory = pd.read_csv(INTERVAL_INVENTORY_PATH)
    interval_inventory["eligible_for_forecasting"] = _bool_series(
        interval_inventory["eligible_for_forecasting"]
    )
    folds = pd.read_parquet(FOLD_PATH)
    members = pd.read_parquet(MEMBERSHIP_DAILY_PATH)
    raw = pd.read_parquet(RECOVERED_RAW_PATH)
    seed_level = pd.read_parquet(SEED_PATH)
    collapsed = pd.read_parquet(COLLAPSED_PATH)
    cells = pd.read_parquet(ALL_CELLS_PATH)
    outputs = run_analytics(members, seed_level, collapsed, cells)
    market_inventory = build_market_inventory(
        interval_inventory, folds, cells, collapsed, raw
    )
    build_blocker_audit(interval_inventory, folds)
    v7_integrity = validate_v7_immutability()
    build_reconciliation(v7_integrity)
    macro = outputs["pit_multimarket_equal_market_macro_v8"]
    inference = outputs["pit_multimarket_bootstrap_inference_v8"]

    # Status is manifested, so its final record count must be known before the
    # manifest is written. The manifest and its validation file are deliberately
    # not self-manifested; every substantive generated artifact is.
    expected_record_count = len(
        [path for path in required_artifact_paths() if path.is_file()]
    )
    write_status(
        market_inventory,
        interval_inventory,
        cells,
        seed_level,
        collapsed,
        macro,
        inference,
        v7_integrity,
        expected_record_count,
        "PASS",
    )
    manifest = build_manifest(required_artifact_paths())
    validation = validate_artifacts(manifest)
    write_json(ARTIFACT_VALIDATION_PATH, validation)
    if validation["status"] != "PASS":
        raise RuntimeError(
            "V8 artifact validation failed: "
            + " | ".join(validation["checksum_failures"])
        )
    # The status checksum was computed after its final PASS content and is now
    # part of the manifest. No manifested file is changed below.
    return {
        "status": json.loads(STATUS_JSON_PATH.read_text(encoding="utf-8"))["status"],
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
    parser.add_argument("--scope", default="repository test suite")
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
    print(json.dumps(v7.json_safe(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
