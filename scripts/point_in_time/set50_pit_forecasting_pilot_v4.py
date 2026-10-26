#!/usr/bin/env python3
"""Controlled SET50 PIT-versus-current forecasting pilot (v4).

This runner is deliberately additive.  It reads the validated v3 SET50
interval, snapshot, and raw-OHLC artifacts, writes only v4 outputs, and
fails closed when an arm or model cell does not satisfy the locked gates.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import warnings
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
SCRIPT_PATH = Path(__file__).resolve()
UTC = timezone.utc
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPERIMENT_ID = "SET50_PIT_FORECASTING_PILOT_V4_20260802"
EXPERIMENT_VERSION = "v4"
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
SEEDS = [17, 42, 101]
TARGET_START = date(2025, 7, 1)
TARGET_END_EXCLUSIVE = date(2026, 7, 1)
INITIAL_END = date(2025, 10, 31)
CALIBRATION_START = date(2025, 11, 1)
CALIBRATION_END = date(2025, 12, 31)
FINAL_START = date(2026, 1, 1)
FINAL_END_EXCLUSIVE = TARGET_END_EXCLUSIVE
MAX_TARGET_GAP_DAYS = 14
THRESHOLD = 0.5
PRIMARY_BLOCK_LENGTH = 20
ROBUSTNESS_BLOCK_LENGTHS = [10, 40]
BOOTSTRAP_SAMPLES = 500
BOOTSTRAP_SEED = 1729
MIN_FINAL_DATES = 60
MIN_FINAL_ROWS = 1000
MIN_CLASS_COUNT = 1
RAW_SEMANTICS = "RAW_OHLC"
PIT_SEMANTICS = "POINT_IN_TIME"
CURRENT_SEMANTICS = "CURRENT_2026_APPLIED_BACKWARD"
INTERSECTION_SEMANTICS = "PIT_CURRENT_INTERSECTION_CONTROL"
PIT_MEMBERSHIP_VERSION = "pit-membership-v3"
PIT_PRICE_VERSION = "pit-data-v3"
CURRENT_MEMBERSHIP_VERSION = "current-universe-2026-config-v1"
CURRENT_PRICE_VERSION = "pit-data-v3"
INTERSECTION_MEMBERSHIP_VERSION = "pit-membership-v3+current-universe-2026-config-v1"

ARM_SPECS = {
    "pit": {
        "universe_semantics": PIT_SEMANTICS,
        "membership_version": PIT_MEMBERSHIP_VERSION,
        "price_version": PIT_PRICE_VERSION,
    },
    "current": {
        "universe_semantics": CURRENT_SEMANTICS,
        "membership_version": CURRENT_MEMBERSHIP_VERSION,
        "price_version": CURRENT_PRICE_VERSION,
    },
    "intersection": {
        "universe_semantics": INTERSECTION_SEMANTICS,
        "membership_version": INTERSECTION_MEMBERSHIP_VERSION,
        "price_version": PIT_PRICE_VERSION,
    },
}

MODEL_SPECS = {
    "historical_class_probability": {
        "family": "historical_class_probability",
        "module": "information_set_baseline",
        "parameters": {"history": "matured_labels_before_prediction_date", "fallback_probability": 0.5},
        "stochastic": False,
    },
    "logistic_regression": {
        "family": "logistic_regression",
        "module": "papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.logistic_regression",
        "parameters": {"C": 1.0, "max_iter": 500},
        "stochastic": True,
    },
    "random_forest": {
        "family": "random_forest",
        "module": "papers.paper_04_regime_transfer_reliability.src.models.machine_learning.random_forest",
        "parameters": {
            "n_estimators": 60,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "max_samples": 0.8,
            "class_weight": "balanced_subsample",
            "n_jobs": 1,
        },
        "stochastic": True,
    },
    "lightgbm": {
        "family": "lightgbm",
        "module": "papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.lightgbm",
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
        "stochastic": True,
    },
}


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    payload = "|".join("" if value is None else str(value) for value in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed.date().isoformat()


def as_date(value: Any) -> date | None:
    text = iso(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return iso(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
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
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow")


def write_csv(path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    frame.to_csv(path, index=False)


def read_parquet(relative: str) -> pd.DataFrame:
    path = PIT_ROOT / relative
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def artifact_manifest(path: Path, purpose: str, record_count: int | None = None) -> Path:
    try:
        relative = path.relative_to(PIT_ROOT).as_posix()
    except ValueError:
        relative = os.path.relpath(path, PIT_ROOT).replace(os.sep, "/")
    safe = relative.replace("/", "__").replace("..", "dotdot").replace(" ", "_")
    manifest_path = PIT_ROOT / "manifests" / f"v4__{safe}.manifest.json"
    write_json(
        manifest_path,
        {
            "pipeline": "set50_pit_forecasting_pilot_v4",
            "pipeline_version": EXPERIMENT_VERSION,
            "path": relative,
            "purpose": purpose,
            "format": path.suffix.lstrip("."),
            "sha256": sha256_file(path),
            "record_count": record_count,
            "created_at": now_utc(),
        },
    )
    return manifest_path


def input_checksums() -> dict[str, str]:
    paths = {
        "membership_v3": PIT_ROOT / "processed" / "index_membership_v3.parquet",
        "pit_snapshot_v3": PIT_ROOT / "snapshots" / "pit_universe_daily_v3.parquet",
        "pit_raw_panel_v3": PIT_ROOT / "processed" / "pit_raw_ohlc_panel_v3.parquet",
        "current_panel_v3": PIT_ROOT / "processed" / "current_universe_matched_panel_v3.parquet",
        "historical_raw_ohlc_v3": PIT_ROOT / "processed" / "historical_raw_ohlc_v3.parquet",
        "eligibility_v3": PIT_ROOT / "qa" / "pit_interval_eligibility_v3.parquet",
        "current_set50_config": REPO_ROOT / "config" / "benchmark_universe" / "thailand_set50_2026_06.csv",
        "feature_source": REPO_ROOT / "papers" / "paper_03_raw_ohlc_forecasting" / "src" / "features" / "raw_ohlc.py",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "lightgbm", "PyYAML"):
        module_name = {"scikit-learn": "sklearn", "PyYAML": "yaml"}.get(name, name)
        try:
            module = importlib.import_module(module_name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - dependency diagnostics
            versions[name] = f"UNAVAILABLE:{type(exc).__name__}"
    return versions


def git_commit_hash() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else "NO_GIT_REPOSITORY"
    except OSError:
        return "NO_GIT_REPOSITORY"


def ensure_dirs() -> None:
    for relative in ("config", "processed", "reconciliation", "reports", "qa", "manifests"):
        (PIT_ROOT / relative).mkdir(parents=True, exist_ok=True)
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)


def load_v3_inputs() -> dict[str, Any]:
    eligibility = read_parquet("qa/pit_interval_eligibility_v3.parquet")
    eligible = eligibility[(eligibility["index_id"].astype(str) == "SET50") & eligibility["eligible"].astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError("No eligible SET50 v3 interval exists; forecasting is blocked.")
    if set(eligible["market"].astype(str)) != {"Thailand"}:
        raise RuntimeError("Eligible SET50 interval has an unexpected market.")
    snapshots = read_parquet("snapshots/pit_universe_daily_v3.parquet")
    snapshots = snapshots[snapshots["index_id"].astype(str) == "SET50"].copy()
    pit_panel = read_parquet("processed/pit_raw_ohlc_panel_v3.parquet")
    current_panel = read_parquet("processed/current_universe_matched_panel_v3.parquet")
    raw = read_parquet("processed/historical_raw_ohlc_v3.parquet")
    for frame_name, frame in (("snapshots", snapshots), ("pit_panel", pit_panel), ("current_panel", current_panel), ("raw", raw)):
        if frame.empty:
            raise RuntimeError(f"Required v3 {frame_name} artifact is empty.")
    for frame in (snapshots, pit_panel, current_panel, raw):
        if "date" in frame:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    raw = raw[raw["price_semantics"].astype(str).eq(RAW_SEMANTICS)].copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.date
    raw["_priority"] = raw["provider"].astype(str).map({"local_existing": 0, "local_vnstock": 1, "yahoo": 2}).fillna(99)
    raw = raw.sort_values(["security_id", "date", "_priority", "provider"], kind="stable").drop_duplicates(["security_id", "date"], keep="first")
    raw = raw.drop(columns=["_priority"])
    eligible_dates = sorted({as_date(value) for value in snapshots["date"].tolist() if as_date(value)})
    return {"eligibility": eligible, "snapshots": snapshots, "pit_panel": pit_panel, "current_panel": current_panel, "raw": raw, "eligible_dates": eligible_dates}


def reconcile_identity() -> pd.DataFrame:
    candidates = read_parquet("reconciliation/entity_resolution_candidates_v2.parquet")
    events = read_parquet("interim/membership_events_extracted_v2.parquet")
    unresolved = candidates[candidates["decision"].astype(str).eq("UNRESOLVED_NAME_ONLY_OFFICIAL_EVENT")].copy()
    if unresolved.empty:
        frame = pd.DataFrame(columns=["record_id", "market", "index_id", "published_name", "unique_security_candidate", "event_record_count", "resolved_security_id", "resolution_status", "unresolved_type", "affects_eligible_interval", "notes"])
    else:
        counts = unresolved.groupby(["market", "index_id", "security_name_as_published"], dropna=False).size().to_dict()
        event_map = {str(row.get("source_record")): row for row in events.to_dict("records")}
        rows = []
        for row in unresolved.sort_values(["market", "index_id", "security_name_as_published", "candidate_id"], kind="stable").to_dict("records"):
            name = row.get("security_name_as_published")
            count = int(counts.get((row.get("market"), row.get("index_id"), name), 1))
            event = event_map.get(str(row.get("source_record")), {})
            duplicate_note = "; repeated published name appears in multiple unresolved event records" if count > 1 else ""
            rows.append({
                "record_id": row.get("candidate_id"),
                "market": row.get("market"),
                "index_id": row.get("index_id"),
                "published_name": name,
                "unique_security_candidate": False,
                "event_record_count": count,
                "resolved_security_id": None,
                "resolution_status": "UNRESOLVED_NAME_ONLY_EVENT_RECORD",
                "unresolved_type": "NAME_ONLY_EVENT_RECORD" + (";DUPLICATED_PUBLISHED_NAME" if count > 1 else ""),
                "affects_eligible_interval": False,
                "notes": f"No official exchange code, ISIN, dated ticker, or documented corporate-action identity link was retained. This is an FBM KLCI record outside the eligible SET50 interval; it is preserved and not used for forecasting.{duplicate_note} Source record: {event.get('source_id') or 'unknown'}.",
            })
        frame = pd.DataFrame(rows)
    path = PIT_ROOT / "reconciliation" / "identity_status_reconciled_v4.csv"
    write_csv(path, frame)
    artifact_manifest(path, "Reconciled identity-status definitions for the 20 v3 name-only records", len(frame))
    return frame


def _coverage_stats(inputs: Mapping[str, Any]) -> dict[str, Any]:
    coverage = read_parquet("reports/active_member_day_price_coverage_v3.parquet")
    market = coverage[(coverage["grain"] == "market_interval") & coverage["index_id"].astype(str).eq("SET50") & (pd.to_numeric(coverage["required_active_member_days"], errors="coerce") > 0)].copy()
    full_required = int(pd.to_numeric(market["required_active_member_days"], errors="coerce").sum())
    full_covered = int(pd.to_numeric(market["covered_active_member_days"], errors="coerce").sum())
    # The old v3 blocker dictionary retained the last blocked SET50 row. It is
    # a single interval, not a full-history aggregate.
    old_blocker = market[(market["interval_start"] == date(2024, 7, 1)) & (market["interval_end"] == date(2025, 1, 1))]
    pit_snapshots = inputs["snapshots"]
    pit_panel = inputs["pit_panel"]
    join = len(pit_panel) / len(pit_snapshots) if len(pit_snapshots) else 0.0
    eligible_rows = market[market["interval_start"].isin([date(2025, 7, 1), date(2026, 1, 1)])].copy()
    eligible_rows["coverage_ratio"] = pd.to_numeric(eligible_rows["coverage_ratio"], errors="coerce")
    security = coverage[(coverage["grain"] == "security") & coverage["index_id"].astype(str).eq("SET50")].copy()
    security["coverage_ratio"] = pd.to_numeric(security["coverage_ratio"], errors="coerce")
    eligible_ids = set(inputs["eligibility"]["eligibility_interval_id"].astype(str))
    eligible_security = security[security["interval_id"].astype(str).isin(eligible_ids)]
    return {
        "full_reconstructed": {
            "interval_count": int(len(market)),
            "required_active_member_days": full_required,
            "covered_active_member_days": full_covered,
            "coverage": full_covered / full_required if full_required else 0.0,
        },
        "old_report_0_920": {
            "interval_start": iso(old_blocker.iloc[0]["interval_start"]) if not old_blocker.empty else None,
            "interval_end_exclusive": iso(old_blocker.iloc[0]["interval_end"]) if not old_blocker.empty else None,
            "required_active_member_days": int(old_blocker.iloc[0]["required_active_member_days"]) if not old_blocker.empty else None,
            "covered_active_member_days": int(old_blocker.iloc[0]["covered_active_member_days"]) if not old_blocker.empty else None,
            "coverage": float(old_blocker.iloc[0]["coverage_ratio"]) if not old_blocker.empty else None,
        },
        "eligible_intervals": [
            {
                "interval_start": iso(row["interval_start"]),
                "interval_end_exclusive": iso(row["interval_end"]),
                "required_active_member_days": int(row["required_active_member_days"]),
                "covered_active_member_days": int(row["covered_active_member_days"]),
                "coverage": float(row["coverage_ratio"]),
            }
            for _, row in eligible_rows.sort_values("interval_start").iterrows()
        ],
        "security_level": {
            "all_interval_security_rows": int(len(security)),
            "all_mean_coverage": float(security["coverage_ratio"].mean()) if not security.empty else 0.0,
            "all_minimum_coverage": float(security["coverage_ratio"].min()) if not security.empty else 0.0,
            "eligible_interval_security_rows": int(len(eligible_security)),
            "eligible_mean_coverage": float(eligible_security["coverage_ratio"].mean()) if not eligible_security.empty else 0.0,
            "eligible_minimum_coverage": float(eligible_security["coverage_ratio"].min()) if not eligible_security.empty else 0.0,
        },
        "snapshot_to_price_join": {"snapshot_rows": int(len(pit_snapshots)), "price_rows": int(len(pit_panel)), "coverage": join, "missing_rows": int(len(pit_snapshots) - len(pit_panel))},
    }


def write_status_reconciliation(inputs: Mapping[str, Any], identity: pd.DataFrame, panel_stats: Mapping[str, Any] | None = None) -> dict[str, Any]:
    coverage = _coverage_stats(inputs)
    panel_stats = panel_stats or {}
    payload = {
        "coverage": coverage,
        "identity": {
            "unresolved_event_records": int(len(identity)),
            "unique_published_names": int(identity["published_name"].nunique()) if not identity.empty else 0,
            "unique_security_candidates": int(identity["unique_security_candidate"].astype(bool).sum()) if not identity.empty else 0,
            "unresolved_index_counts": identity.groupby("index_id").size().to_dict() if not identity.empty else {},
            "affects_eligible_interval": int(identity["affects_eligible_interval"].astype(bool).sum()) if not identity.empty else 0,
        },
        "forecasting_panel": panel_stats,
    }
    path = REPORT_ROOT / "SET50_PIT_STATUS_RECONCILIATION.md"
    c = coverage
    lines = [
        "# SET50 PIT status reconciliation v4",
        "",
        "This audit reconciles v3 coverage and identity terminology before the forecasting pilot. It does not promote blocked SET50 history to eligible data.",
        "",
        "## Coverage",
        "",
        f"- Full reconstructed-history population: {c['full_reconstructed']['interval_count']} evidence-backed SET50 intervals; {c['full_reconstructed']['covered_active_member_days']:,}/{c['full_reconstructed']['required_active_member_days']:,} active-member days covered = **{c['full_reconstructed']['coverage']:.6f}**. The denominator is 50 constituents times the union of observed raw-price market dates for each interval.",
        f"- The prior v3 value **0.920** was not a full-history estimate. It was the single blocked interval {c['old_report_0_920']['interval_start']} through {c['old_report_0_920']['interval_end_exclusive']} ({c['old_report_0_920']['covered_active_member_days']:,}/{c['old_report_0_920']['required_active_member_days']:,} active-member days).",
        *[f"- Eligible interval {item['interval_start']} through {item['interval_end_exclusive']}: {item['covered_active_member_days']:,}/{item['required_active_member_days']:,} active-member days = **{item['coverage']:.6f}**." for item in c["eligible_intervals"]],
        f"- Security-level coverage: all reconstructed interval-security rows mean {c['security_level']['all_mean_coverage']:.6f}, minimum {c['security_level']['all_minimum_coverage']:.6f}; eligible interval-security rows mean {c['security_level']['eligible_mean_coverage']:.6f}, minimum {c['security_level']['eligible_minimum_coverage']:.6f}.",
        f"- PIT snapshot-to-price join: {c['snapshot_to_price_join']['price_rows']:,}/{c['snapshot_to_price_join']['snapshot_rows']:,} = **{c['snapshot_to_price_join']['coverage']:.6f}**; {c['snapshot_to_price_join']['missing_rows']:,} snapshot rows lack a raw-OHLC join.",
        f"- Forecasting-panel coverage is reported after observation/feature gates: {json.dumps(panel_stats, sort_keys=True, default=json_default)}.",
        "",
        "## Identity",
        "",
        f"- Unresolved v3 records: {len(identity):,} name-only FBM KLCI event records representing {identity['published_name'].nunique() if not identity.empty else 0} unique published names; no unique security candidate was accepted.",
        f"- Affects eligible SET50 interval: {int(identity['affects_eligible_interval'].astype(bool).sum()) if not identity.empty else 0}. These records are outside the pilot market/interval.",
        "- Repeated published names are retained as separate event records and are not silently collapsed into securities.",
        "",
        "## Eligibility interpretation",
        "",
        "Only the two v3 SET50 intervals passing all locked gates are used for forecasting. The full reconstructed SET50 history remains ineligible because earlier intervals fail the active-member-day price gate and the target period has incomplete event-chain coverage.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_manifest(path, "Coverage and identity reconciliation before SET50 forecasting", None)
    return payload


def build_config(inputs: Mapping[str, Any]) -> dict[str, Any]:
    feature_source = REPO_ROOT / "papers" / "paper_03_raw_ohlc_forecasting" / "src" / "features" / "raw_ohlc.py"
    model_paths = {
        "logistic_regression": REPO_ROOT / "papers" / "paper_03_raw_ohlc_forecasting" / "src" / "models" / "machine_learning" / "logistic_regression.py",
        "random_forest": REPO_ROOT / "papers" / "paper_04_regime_transfer_reliability" / "src" / "models" / "machine_learning" / "random_forest.py",
        "lightgbm": REPO_ROOT / "papers" / "paper_03_raw_ohlc_forecasting" / "src" / "models" / "machine_learning" / "lightgbm.py",
    }
    models = json.loads(json.dumps(MODEL_SPECS))
    for model_id, path in model_paths.items():
        models[model_id]["source_path"] = str(path.relative_to(REPO_ROOT))
        models[model_id]["source_sha256"] = sha256_file(path) if path.is_file() else None
    return {
        "experiment": {
            "experiment_id": EXPERIMENT_ID,
            "experiment_version": EXPERIMENT_VERSION,
            "status": "LOCKED_PRE_EXECUTION",
            "correction_policy": "CREATE_NEW_EXPERIMENT_VERSION_DO_NOT_EDIT_AFTER_FINAL_EVALUATION",
            "date_semantics": "exchange-local trading date stored as a naive ISO calendar date",
            "code_commit_hash": git_commit_hash(),
            "code_sha256": sha256_file(SCRIPT_PATH),
        },
        "inputs": {
            "data_versions": {"membership": PIT_MEMBERSHIP_VERSION, "snapshot": "pit-universe-v3", "price": PIT_PRICE_VERSION},
            "checksums": input_checksums(),
            "source_mode": "READ_ONLY_VALIDATED_V3_ARTIFACTS",
        },
        "scope": {"market": "Thailand", "index_id": "SET50", "eligible_interval_start": iso(TARGET_START), "eligible_interval_end_exclusive": iso(TARGET_END_EXCLUSIVE), "full_target_period_ineligible": True},
        "universe_arms": {name: {"universe_semantics": spec["universe_semantics"], "role": "primary" if name == "pit" else "matched comparator" if name == "current" else "diagnostic intersection control"} for name, spec in ARM_SPECS.items()},
        "feature_registry": {"version": FEATURE_VERSION, "source_path": str(feature_source.relative_to(REPO_ROOT)), "source_sha256": sha256_file(feature_source), "columns": FEATURE_COLUMNS, "prohibited": ["adj_close", "adjustment_factor", "volume_raw", "volume_valid", "dividend", "stock_split"]},
        "target": {"name": "next_session_open_to_close_direction", "formula": "I(close_raw[target_date] / open_raw[target_date] - 1 > 0)", "prediction_timestamp": "after close of prediction_date", "target_date_rule": "next valid same-security raw-OHLC observation within 14 calendar days", "pit_target_membership_required": True, "zero_return_class": 0, "missing_labels": "rejected_not_zero"},
        "temporal_split": {"initial_training": {"start": iso(TARGET_START), "end": iso(INITIAL_END)}, "calibration": {"start": iso(CALIBRATION_START), "end": iso(CALIBRATION_END)}, "final_test": {"start": iso(FINAL_START), "end_exclusive": iso(FINAL_END_EXCLUSIVE)}, "training_window": "EXPANDING", "threshold": THRESHOLD, "final_endpoint_adjustment": "last prediction date with a valid matured target; no split boundary moved after final-test inspection"},
        "model_registry": models,
        "execution": {"seeds": SEEDS, "minimum_final_test_dates": MIN_FINAL_DATES, "minimum_final_test_rows": MIN_FINAL_ROWS, "minimum_class_count": MIN_CLASS_COUNT, "training_label_rule": "target_date < prediction_date", "feature_selection_final_test": False, "hyperparameter_search": False, "identity_as_feature": False},
        "inference": {"primary": {"method": "moving_block_bootstrap_over_trading_dates", "block_length": PRIMARY_BLOCK_LENGTH, "samples": BOOTSTRAP_SAMPLES, "seed": BOOTSTRAP_SEED}, "robustness_block_lengths": ROBUSTNESS_BLOCK_LENGTHS, "confidence": 0.95, "aggregation": "equal_date", "comparison_family": "current_minus_pit_brier_by_model_and_comparison_arm", "multiple_testing": "Benjamini-Hochberg within block length"},
        "package_versions": package_versions(),
    }


def freeze_config(inputs: Mapping[str, Any]) -> dict[str, Any]:
    ensure_dirs()
    path = PIT_ROOT / "config" / "set50_pit_forecasting_pilot_v4.yaml"
    expected = build_config(inputs)
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if existing.get("experiment", {}).get("experiment_id") != EXPERIMENT_ID:
            raise RuntimeError("Existing v4 config belongs to a different experiment; create a new version.")
        return existing
    path.write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")
    artifact_manifest(path, "Immutable SET50 forecasting pilot configuration", None)
    return expected


def _valid_ohlc(frame: pd.DataFrame) -> pd.Series:
    values = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=frame.index)
    valid &= values.gt(0.0).all(axis=1)
    valid &= values["high"] >= values[["open", "low", "close"]].max(axis=1)
    valid &= values["low"] <= values[["open", "high", "close"]].min(axis=1)
    return valid


def _raw_by_ticker(raw: pd.DataFrame) -> pd.DataFrame:
    result = raw.copy()
    result = result.rename(columns={"ticker_used": "ticker"})
    result = result[_valid_ohlc(result)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.date
    result = result.dropna(subset=["market", "ticker", "date"])
    result["_priority"] = result["provider"].astype(str).map({"local_existing": 0, "local_vnstock": 1, "yahoo": 2}).fillna(99)
    result = result.sort_values(["market", "ticker", "date", "_priority", "security_id"], kind="stable")
    return result.drop_duplicates(["market", "ticker", "date"], keep="first").drop(columns=["_priority"])


def _current_universe_tickers() -> list[str]:
    path = REPO_ROOT / "config" / "benchmark_universe" / "thailand_set50_2026_06.csv"
    frame = pd.read_csv(path, dtype=str).fillna("")
    return sorted(set(frame["local_symbol"].astype(str)))


def _eligible_dates(inputs: Mapping[str, Any]) -> list[date]:
    return sorted({as_date(value) for value in inputs["snapshots"]["date"].tolist() if as_date(value)})


def _interval_for_date(eligibility: pd.DataFrame, on_date: date) -> dict[str, Any]:
    rows = eligibility[(pd.to_datetime(eligibility["interval_start"], errors="coerce").dt.date <= on_date) & (pd.to_datetime(eligibility["interval_end"], errors="coerce").dt.date > on_date)]
    return rows.iloc[0].to_dict() if not rows.empty else {}


def _membership_tables(inputs: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    dates = _eligible_dates(inputs)
    date_frame = pd.DataFrame({"date": dates})
    snapshots = inputs["snapshots"].copy()
    snapshots["date"] = pd.to_datetime(snapshots["date"], errors="coerce").dt.date
    pit_members = snapshots[["date", "market", "index_id", "security_id", "ticker_on_date", "membership_effective_from", "membership_effective_to", "source_chain", "eligibility_interval_id"]].copy()
    pit_members["price_security_id"] = pit_members["security_id"]
    pit_members["arm_key"] = pit_members["security_id"]

    current_tickers = _current_universe_tickers()
    current_rows = []
    for on_date in dates:
        interval = _interval_for_date(inputs["eligibility"], on_date)
        for ticker in current_tickers:
            security_id = stable_id("CURRENT", "SET50", ticker)
            current_rows.append({
                "date": on_date,
                "market": "Thailand",
                "index_id": "SET50",
                "security_id": security_id,
                "ticker_on_date": ticker,
                "membership_effective_from": interval.get("interval_start"),
                "membership_effective_to": interval.get("interval_end"),
                "source_chain": "CURRENT_2026_CONFIG_ONLY_NOT_PIT_EVIDENCE",
                "eligibility_interval_id": interval.get("eligibility_interval_id"),
                "price_security_id": None,
                "arm_key": security_id,
            })
    current_members = pd.DataFrame(current_rows)
    current_members["membership_effective_from"] = current_members["membership_effective_from"].map(as_date)
    current_members["membership_effective_to"] = current_members["membership_effective_to"].map(as_date)

    pit_key = pit_members["date"].astype(str) + "|" + pit_members["ticker_on_date"].astype(str)
    current_key = current_members["date"].astype(str) + "|" + current_members["ticker_on_date"].astype(str)
    common = sorted(set(pit_key).intersection(set(current_key)))
    common_set = set(common)
    intersection_rows = []
    for row in pit_members.to_dict("records"):
        key = f"{row['date']}|{row['ticker_on_date']}"
        if key not in common_set:
            continue
        intersection_rows.append({
            **row,
            "security_id": stable_id("INTERSECTION", "SET50", row["ticker_on_date"]),
            "price_security_id": None,
            "arm_key": stable_id("INTERSECTION", "SET50", row["ticker_on_date"]),
            "source_chain": f"{row.get('source_chain') or ''}|CURRENT_2026_CONFIG_ONLY_NOT_PIT_EVIDENCE",
        })
    intersection_members = pd.DataFrame(intersection_rows)
    return {"pit": pit_members, "current": current_members, "intersection": intersection_members, "date_frame": date_frame}


def _raw_series_for_arm(arm: str, members: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Return all valid raw rows for the arm's security/ticker identities.

    Pre-eligible rows are retained only as price lookbacks.  Membership fields
    are never carried backward from the eligible interval.
    """
    ticker_raw = _raw_by_ticker(raw)
    rows: list[pd.DataFrame] = []
    if arm == "pit":
        ids = sorted(set(members["security_id"].astype(str)))
        for security_id in ids:
            part = raw[raw["security_id"].astype(str).eq(security_id)].copy()
            if part.empty:
                continue
            part["arm_security_id"] = security_id
            part["ticker_on_date"] = part["ticker_used"].astype(str)
            rows.append(part)
    else:
        tickers = sorted(set(members["ticker_on_date"].astype(str)))
        for ticker in tickers:
            part = ticker_raw[(ticker_raw["market"].astype(str) == "Thailand") & ticker_raw["ticker"].astype(str).eq(ticker)].copy()
            if part.empty:
                continue
            prefix = "CURRENT" if arm == "current" else "INTERSECTION"
            arm_id = stable_id(prefix, "SET50", ticker)
            part["arm_security_id"] = arm_id
            part["ticker_on_date"] = ticker
            rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["arm_security_id", "market", "date", "ticker_on_date", "open", "high", "low", "close", "provider", "price_semantics"])
    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame[_valid_ohlc(frame)].copy()
    frame = frame.sort_values(["arm_security_id", "date", "provider"], kind="stable").drop_duplicates(["arm_security_id", "date"], keep="first")
    return frame


def _target_map(series: pd.DataFrame) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=["arm_security_id", "date", "target_date", "target_open", "target_close", "target_ticker", "target_gap_days"])
    data = series.sort_values(["arm_security_id", "date"], kind="stable").copy()
    grouped = data.groupby("arm_security_id", sort=False)
    data["target_date"] = grouped["date"].shift(-1)
    data["target_open"] = grouped["open"].shift(-1)
    data["target_close"] = grouped["close"].shift(-1)
    data["target_ticker"] = grouped["ticker_on_date"].shift(-1)
    data["target_gap_days"] = [(b - a).days if isinstance(a, date) and isinstance(b, date) else np.nan for a, b in zip(data["date"], data["target_date"])]
    return data[["arm_security_id", "date", "target_date", "target_open", "target_close", "target_ticker", "target_gap_days"]]


def _candidate_rows(arm: str, members: pd.DataFrame, series: pd.DataFrame, inputs: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct valid observations and explicit rejection reasons."""
    dates = set(_eligible_dates(inputs))
    target_map = _target_map(series)
    price_cols = ["open", "high", "low", "close", "provider", "price_semantics"]
    candidate = members.copy()
    candidate = candidate.rename(columns={"security_id": "arm_security_id"})
    candidate = candidate.merge(series[["arm_security_id", "date", *price_cols]], on=["arm_security_id", "date"], how="left", suffixes=("", "_price"))
    candidate = candidate.merge(target_map, on=["arm_security_id", "date"], how="left")
    candidate["prediction_date"] = candidate["date"]
    candidate["rejection_reason"] = None

    # Build active-member keys from the arm's evidence-backed membership table.
    active_keys = {(str(row["security_id"]), as_date(row["date"])) for row in members.to_dict("records") if as_date(row.get("date"))}
    for idx, row in candidate.iterrows():
        on_date = as_date(row.get("date")); target_date = as_date(row.get("target_date")); arm_id = str(row.get("arm_security_id"))
        reason = None
        if on_date not in dates:
            reason = "OUTSIDE_ELIGIBLE_INTERVAL"
        elif pd.isna(row.get("open")) or pd.isna(row.get("high")) or pd.isna(row.get("low")) or pd.isna(row.get("close")):
            reason = "MISSING_PREDICTION_OHLC"
        elif not target_date:
            reason = "NO_NEXT_VALID_TARGET_OBSERVATION"
        elif target_date not in dates:
            reason = "TARGET_OUTSIDE_ELIGIBLE_INTERVAL"
        elif float(row.get("target_gap_days") or 0) > MAX_TARGET_GAP_DAYS:
            reason = "LONG_TARGET_GAP"
        elif pd.isna(row.get("target_open")) or pd.isna(row.get("target_close")):
            reason = "MISSING_TARGET_OHLC"
        elif arm == "pit" and (arm_id, target_date) not in active_keys:
            reason = "TARGET_NOT_ACTIVE_PIT_MEMBER"
        elif arm == "intersection" and (arm_id, target_date) not in active_keys:
            reason = "TARGET_NOT_IN_INTERSECTION"
        elif str(row.get("ticker_on_date") or "") != str(row.get("target_ticker") or ""):
            reason = "TICKER_VALIDITY_BOUNDARY"
        if reason:
            candidate.at[idx, "rejection_reason"] = reason

    reject = candidate[candidate["rejection_reason"].notna()].copy()
    valid = candidate[candidate["rejection_reason"].isna()].copy()
    if not valid.empty:
        valid["target_return"] = valid["target_close"].astype(float) / valid["target_open"].astype(float) - 1.0
        valid["actual_class"] = (valid["target_return"] > 0).astype(int)
        valid["prediction_date"] = valid["prediction_date"].map(as_date)
        valid["target_date"] = valid["target_date"].map(as_date)
        valid["price_semantics"] = RAW_SEMANTICS
        valid["experiment_id"] = EXPERIMENT_ID
        valid["universe_semantics"] = ARM_SPECS[arm]["universe_semantics"]
        valid["membership_version"] = ARM_SPECS[arm]["membership_version"]
        valid["price_version"] = ARM_SPECS[arm]["price_version"]
        valid["market"] = "Thailand"
        valid["index_id"] = "SET50"
        valid["security_id"] = valid["arm_security_id"]
        valid = valid.drop(columns=["arm_security_id"])
    if not reject.empty:
        reject["experiment_id"] = EXPERIMENT_ID
        reject["arm"] = arm
        reject["universe_semantics"] = ARM_SPECS[arm]["universe_semantics"]
        reject["prediction_date"] = reject["prediction_date"].map(as_date)
        reject["target_date"] = reject["target_date"].map(as_date)
        reject["security_id"] = reject["arm_security_id"]
        reject["reason_detail"] = reject["rejection_reason"]
        reject = reject[["experiment_id", "arm", "universe_semantics", "market", "index_id", "security_id", "ticker_on_date", "prediction_date", "target_date", "rejection_reason", "reason_detail"]]
    return valid, reject


def _apply_market_feature_definition(full_featured: pd.DataFrame, eligible_series: pd.DataFrame, feature_builder: Any) -> pd.DataFrame:
    """Replace market-level feature columns with eligible-date-only observables.

    This preserves pre-eligible individual price lookbacks while preventing
    unknown pre-eligible membership from entering market aggregates.
    """
    if full_featured.empty or eligible_series.empty:
        return full_featured
    market_input = eligible_series[["market", "arm_security_id", "date", "open", "high", "low", "close"]].copy()
    market_input = market_input.rename(columns={"arm_security_id": "symbol_canonical", "open": "open_raw", "high": "high_raw", "low": "low_raw", "close": "close_raw"})
    market_featured, _ = feature_builder(market_input)
    market_featured["date"] = pd.to_datetime(market_featured["date"], errors="coerce").dt.date
    lookup = market_featured[["symbol_canonical", "date", "lagged_market_return", "lagged_market_breadth"]].rename(columns={"symbol_canonical": "arm_security_id"})
    result = full_featured.drop(columns=["lagged_market_return", "lagged_market_breadth"], errors="ignore")
    return result.merge(lookup, on=["arm_security_id", "date"], how="left")


def build_features_for_arm(arm: str, members: pd.DataFrame, valid_obs: pd.DataFrame, series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from papers.paper_03_raw_ohlc_forecasting.src.features.raw_ohlc import build_lagged_raw_features

    if valid_obs.empty or series.empty:
        return pd.DataFrame(), pd.DataFrame()
    full = series[["market", "arm_security_id", "date", "open", "high", "low", "close"]].copy()
    full = full.rename(columns={"arm_security_id": "symbol_canonical", "open": "open_raw", "high": "high_raw", "low": "low_raw", "close": "close_raw"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full_featured, _ = build_lagged_raw_features(full)
    full_featured["date"] = pd.to_datetime(full_featured["date"], errors="coerce").dt.date
    full_featured = full_featured.rename(columns={"symbol_canonical": "arm_security_id"})
    eligible_keys = {(str(row["security_id"]), as_date(row["prediction_date"])) for row in valid_obs.to_dict("records") if as_date(row.get("prediction_date"))}
    eligible_series = series[series.apply(lambda row: (str(row["arm_security_id"]), as_date(row["date"])) in eligible_keys, axis=1)].copy()
    full_featured = _apply_market_feature_definition(full_featured, eligible_series, build_lagged_raw_features)
    wanted = valid_obs[["security_id", "prediction_date", "target_date", "target_return", "actual_class", "ticker_on_date", "market", "index_id", "membership_effective_from", "membership_effective_to", "source_chain", "eligibility_interval_id", "price_version", "membership_version", "universe_semantics", "price_semantics"]].copy()
    wanted = wanted.rename(columns={"security_id": "arm_security_id", "prediction_date": "date"})
    result = wanted.merge(full_featured[["arm_security_id", "date", *FEATURE_COLUMNS]], on=["arm_security_id", "date"], how="left")
    result["feature_version"] = FEATURE_VERSION
    result["experiment_id"] = EXPERIMENT_ID
    result["target_date"] = result["target_date"].map(as_date)
    result["prediction_date"] = result["date"].map(as_date)
    result["date"] = result["prediction_date"]
    result["security_id"] = result["arm_security_id"]
    result = result.drop(columns=["arm_security_id"])
    finite = np.isfinite(result[FEATURE_COLUMNS].to_numpy(dtype=float)).all(axis=1)
    feature_reject = result.loc[~finite, ["experiment_id", "market", "index_id", "security_id", "ticker_on_date", "prediction_date", "target_date"]].copy()
    if not feature_reject.empty:
        feature_reject["arm"] = arm
        feature_reject["universe_semantics"] = ARM_SPECS[arm]["universe_semantics"]
        feature_reject["rejection_reason"] = "MISSING_FEATURES"
        feature_reject["reason_detail"] = "At least one locked raw-OHLC lagged feature is unavailable; no imputation or boundary backfill was applied."
    result = result.loc[finite].copy()
    result["is_final_test"] = result["prediction_date"].map(lambda value: FINAL_START <= value < FINAL_END_EXCLUSIVE)
    result = result.sort_values(["prediction_date", "security_id"], kind="stable").reset_index(drop=True)
    return result, feature_reject


def write_arm_data(inputs: Mapping[str, Any], identity: pd.DataFrame) -> dict[str, Any]:
    members = _membership_tables(inputs)
    outputs: dict[str, Any] = {"inputs": inputs, "members": members, "observations": {}, "features": {}, "rejections": []}
    all_rejections = []
    for arm in ("pit", "current", "intersection"):
        series = _raw_series_for_arm(arm, members[arm], inputs["raw"])
        valid_obs, reject = _candidate_rows(arm, members[arm], series, inputs)
        feature_panel, feature_reject = build_features_for_arm(arm, members[arm], valid_obs, series)
        if not feature_reject.empty:
            all_rejections.append(feature_reject)
        if not reject.empty:
            all_rejections.append(reject)
        outputs["observations"][arm] = valid_obs
        outputs["features"][arm] = feature_panel
        outputs.setdefault("series", {})[arm] = series
        obs_path = PIT_ROOT / "processed" / f"set50_{'pit' if arm == 'pit' else 'current' if arm == 'current' else 'intersection'}_observation_panel_v4.parquet"
        feat_path = PIT_ROOT / "processed" / f"set50_{'pit' if arm == 'pit' else 'current' if arm == 'current' else 'intersection'}_feature_panel_v4.parquet"
        write_parquet(obs_path, valid_obs)
        write_parquet(feat_path, feature_panel)
        artifact_manifest(obs_path, f"SET50 v4 {arm} observation panel with explicit next-session target", len(valid_obs))
        artifact_manifest(feat_path, f"SET50 v4 {arm} locked common raw-OHLC feature panel", len(feature_panel))
    rejection_frame = pd.concat(all_rejections, ignore_index=True) if all_rejections else pd.DataFrame(columns=["experiment_id", "arm", "universe_semantics", "market", "index_id", "security_id", "ticker_on_date", "prediction_date", "target_date", "rejection_reason", "reason_detail"])
    rejection_path = PIT_ROOT / "qa" / "set50_observation_rejections_v4.parquet"
    write_parquet(rejection_path, rejection_frame)
    artifact_manifest(rejection_path, "Fail-closed SET50 observation and feature rejection reasons", len(rejection_frame))
    # Rewrite the reconciliation report with post-feature coverage counts.
    stats = {
        "by_arm": {
            arm: {
                "candidate_membership_rows": int(len(members[arm])),
                "valid_observation_rows": int(len(outputs["observations"][arm])),
                "feature_rows": int(len(outputs["features"][arm])),
                "final_feature_rows": int(outputs["features"][arm]["is_final_test"].astype(bool).sum()) if not outputs["features"][arm].empty else 0,
                "coverage_from_membership_to_observation": len(outputs["observations"][arm]) / len(members[arm]) if len(members[arm]) else 0.0,
                "coverage_from_observation_to_feature": len(outputs["features"][arm]) / len(outputs["observations"][arm]) if len(outputs["observations"][arm]) else 0.0,
            }
            for arm in ("pit", "current", "intersection")
        },
        "rejection_rows": int(len(rejection_frame)),
    }
    write_status_reconciliation(inputs, identity, stats)
    return outputs


def _period_for_date(on_date: date) -> str:
    if TARGET_START <= on_date <= INITIAL_END:
        return "initial_training"
    if CALIBRATION_START <= on_date <= CALIBRATION_END:
        return "calibration"
    if FINAL_START <= on_date < FINAL_END_EXCLUSIVE:
        return "final_test"
    return "outside"


def _build_model(model_id: str, seed: int) -> Any:
    if model_id == "logistic_regression":
        module = importlib.import_module("papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.logistic_regression")
    elif model_id == "random_forest":
        module = importlib.import_module("papers.paper_04_regime_transfer_reliability.src.models.machine_learning.random_forest")
    elif model_id == "lightgbm":
        module = importlib.import_module("papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.lightgbm")
    else:
        raise ValueError(model_id)
    return module.build_model(seed=seed)


def _model_probability(model_id: str, model: Any, X: pd.DataFrame) -> np.ndarray:
    if model_id == "logistic_regression":
        module = importlib.import_module("papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.logistic_regression")
    elif model_id == "random_forest":
        module = importlib.import_module("papers.paper_04_regime_transfer_reliability.src.models.machine_learning.random_forest")
    elif model_id == "lightgbm":
        module = importlib.import_module("papers.paper_03_raw_ohlc_forecasting.src.models.machine_learning.lightgbm")
    else:
        raise ValueError(model_id)
    probability = np.asarray(module.predict_proba(model, X), dtype=float)
    return np.clip(probability, 0.0, 1.0)


def _training_rows(frame: pd.DataFrame, prediction_date: date) -> pd.DataFrame:
    dates = pd.to_datetime(frame["prediction_date"], errors="coerce").dt.date
    targets = pd.to_datetime(frame["target_date"], errors="coerce").dt.date
    selected = frame[(dates < prediction_date) & (targets < prediction_date)].copy()
    return selected


def _prediction_rows(frame: pd.DataFrame, prediction_date: date) -> pd.DataFrame:
    return frame[pd.to_datetime(frame["prediction_date"], errors="coerce").dt.date.eq(prediction_date)].copy()


def _prediction_row(base: Mapping[str, Any], *, model_id: str, model_family: str, model_version: str, training: pd.DataFrame, seed: int, probability: float, is_final: bool) -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "universe_semantics": base.get("universe_semantics"),
        "market": base.get("market"),
        "index_id": base.get("index_id"),
        "security_id": base.get("security_id"),
        "ticker_on_date": base.get("ticker_on_date"),
        "prediction_date": as_date(base.get("prediction_date")),
        "target_date": as_date(base.get("target_date")),
        "model_id": model_id,
        "model_family": model_family,
        "model_version": model_version,
        "training_start": as_date(training["prediction_date"].min()) if not training.empty else None,
        "training_end": as_date(training["prediction_date"].max()) if not training.empty else None,
        "training_row_count": int(len(training)),
        "seed": seed,
        "prediction_probability": float(probability),
        "predicted_class": int(float(probability) >= THRESHOLD),
        "actual_class": int(base.get("actual_class")),
        "feature_version": FEATURE_VERSION,
        "membership_version": base.get("membership_version"),
        "price_version": base.get("price_version"),
        "is_final_test": bool(is_final),
    }


def _prediction_columns() -> list[str]:
    return [
        "experiment_id", "universe_semantics", "market", "index_id", "security_id", "ticker_on_date",
        "prediction_date", "target_date", "model_id", "model_family", "model_version", "training_start",
        "training_end", "training_row_count", "seed", "prediction_probability", "predicted_class",
        "actual_class", "feature_version", "membership_version", "price_version", "is_final_test",
    ]


def run_models_for_arm(arm: str, features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train expanding models daily and retain calibration/final predictions."""
    columns = _prediction_columns()
    if features.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns + ["seed_count"]), pd.DataFrame()
    data = features.copy()
    data["prediction_date"] = pd.to_datetime(data["prediction_date"], errors="coerce").dt.date
    data["target_date"] = pd.to_datetime(data["target_date"], errors="coerce").dt.date
    data = data.sort_values(["prediction_date", "security_id"], kind="stable").reset_index(drop=True)
    prediction_dates = sorted({value for value in data["prediction_date"].tolist() if value and _period_for_date(value) in {"calibration", "final_test"}})
    rows: list[dict[str, Any]] = []
    invalid_training: list[dict[str, Any]] = []
    X_all = data[FEATURE_COLUMNS]
    for prediction_date in prediction_dates:
        train = _training_rows(data, prediction_date)
        test = _prediction_rows(data, prediction_date)
        if train.empty or test.empty:
            continue
        y_train = train["actual_class"].astype(int)
        if y_train.nunique() < 2:
            invalid_training.append({"arm": arm, "model_id": "ALL", "cell": "training", "prediction_date": prediction_date, "reason": "TRAINING_ONE_CLASS"})
            continue
        X_train = train[FEATURE_COLUMNS]
        X_test = test[FEATURE_COLUMNS]
        is_final = prediction_date >= FINAL_START
        for model_id in MODEL_SPECS:
            for seed in SEEDS:
                try:
                    if model_id == "historical_class_probability":
                        probability = np.full(len(test), float(y_train.mean()), dtype=float)
                    else:
                        estimator = _build_model(model_id, seed)
                        estimator.fit(X_train, y_train)
                        probability = _model_probability(model_id, estimator, X_test)
                    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
                        invalid_training.append({"arm": arm, "model_id": model_id, "cell": "prediction", "prediction_date": prediction_date, "reason": "PROBABILITY_OUT_OF_BOUNDS"})
                        continue
                    model_version = f"{EXPERIMENT_VERSION}-{model_id}-locked"
                    for base, value in zip(test.to_dict("records"), probability):
                        rows.append(_prediction_row(base, model_id=model_id, model_family=MODEL_SPECS[model_id]["family"], model_version=model_version, training=train, seed=seed, probability=float(value), is_final=is_final))
                except Exception as exc:  # dependency or one-cell fit failure is explicit, not silently substituted
                    invalid_training.append({"arm": arm, "model_id": model_id, "cell": "fit", "prediction_date": prediction_date, "reason": f"{type(exc).__name__}:{exc}"})
    seed_frame = pd.DataFrame(rows, columns=columns)
    if not seed_frame.empty:
        seed_frame = seed_frame.sort_values(["prediction_date", "model_id", "security_id", "seed"], kind="stable").reset_index(drop=True)
    collapsed_rows = []
    if not seed_frame.empty:
        group_columns = [column for column in columns if column not in {"seed", "prediction_probability", "predicted_class"}]
        for key, group in seed_frame.groupby(["experiment_id", "universe_semantics", "market", "index_id", "security_id", "ticker_on_date", "prediction_date", "target_date", "model_id", "model_family", "model_version", "training_start", "training_end", "training_row_count", "actual_class", "feature_version", "membership_version", "price_version", "is_final_test"], dropna=False, sort=True):
            record = dict(zip(["experiment_id", "universe_semantics", "market", "index_id", "security_id", "ticker_on_date", "prediction_date", "target_date", "model_id", "model_family", "model_version", "training_start", "training_end", "training_row_count", "actual_class", "feature_version", "membership_version", "price_version", "is_final_test"], key))
            record["seed"] = "COLLAPSED"
            record["prediction_probability"] = float(group["prediction_probability"].mean())
            record["predicted_class"] = int(record["prediction_probability"] >= THRESHOLD)
            record["seed_count"] = int(group["seed"].nunique())
            collapsed_rows.append(record)
    collapsed = pd.DataFrame(collapsed_rows)
    if not collapsed.empty:
        collapsed = collapsed.reindex(columns=columns + ["seed_count"]).sort_values(["prediction_date", "model_id", "security_id"], kind="stable").reset_index(drop=True)
    invalid = pd.DataFrame(invalid_training)
    return seed_frame, collapsed, invalid


def run_models(data: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    predictions: dict[str, dict[str, pd.DataFrame]] = {}
    invalid_rows = []
    for arm in ("pit", "current", "intersection"):
        seed, collapsed, invalid = run_models_for_arm(arm, data["features"][arm])
        predictions[arm] = {"seed": seed, "collapsed": collapsed}
        if not invalid.empty:
            invalid["stage"] = "model_run"
            invalid_rows.append(invalid)
        prefix = "pit" if arm == "pit" else "current" if arm == "current" else "intersection"
        seed_path = PIT_ROOT / "processed" / f"set50_{prefix}_predictions_seed_level_v4.parquet"
        collapsed_path = PIT_ROOT / "processed" / f"set50_{prefix}_predictions_seed_collapsed_v4.parquet"
        write_parquet(seed_path, seed)
        write_parquet(collapsed_path, collapsed)
        artifact_manifest(seed_path, f"SET50 v4 {arm} seed-level expanding OOS predictions", len(seed))
        artifact_manifest(collapsed_path, f"SET50 v4 {arm} arithmetic seed-collapsed predictions", len(collapsed))
    invalid_frame = pd.concat(invalid_rows, ignore_index=True) if invalid_rows else pd.DataFrame(columns=["arm", "model_id", "cell", "prediction_date", "reason", "stage"])
    return {"predictions": predictions, "invalid_training": invalid_frame}


def _safe_log_loss(y: np.ndarray, probability: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(np.asarray(probability, dtype=float), 1e-15, 1 - 1e-15)
    y = np.asarray(y, dtype=int)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    values = []
    for label in (0, 1):
        mask = y == label
        if mask.any():
            values.append(float((predicted[mask] == label).mean()))
    return float(np.mean(values)) if values else float("nan")


def _calibration(y: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2 or len(np.unique(p)) < 2:
        return float("nan"), float("nan")
    from sklearn.linear_model import LogisticRegression

    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1e6, max_iter=1000, solver="lbfgs")
        model.fit(logit, y)
        return float(model.intercept_[0]), float(model.coef_[0, 0])
    except Exception:
        return float("nan"), float("nan")


def _auroc(y: np.ndarray, probability: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score

    try:
        return float(roc_auc_score(y, probability))
    except Exception:
        return float("nan")


def metric_values(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {"brier_score": np.nan, "log_loss": np.nan, "balanced_accuracy": np.nan, "calibration_intercept": np.nan, "calibration_slope": np.nan, "auroc": np.nan, "prediction_count": 0, "class_0_count": 0, "class_1_count": 0}
    y = frame["actual_class"].to_numpy(dtype=int)
    p = frame["prediction_probability"].to_numpy(dtype=float)
    pred = frame["predicted_class"].to_numpy(dtype=int)
    intercept, slope = _calibration(y, p)
    return {
        "brier_score": float(np.mean((p - y) ** 2)),
        "log_loss": _safe_log_loss(y, p),
        "balanced_accuracy": _balanced_accuracy(y, pred),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "auroc": _auroc(y, p),
        "prediction_count": int(len(frame)),
        "class_0_count": int((y == 0).sum()),
        "class_1_count": int((y == 1).sum()),
    }


def _date_metric_frame(frame: pd.DataFrame, *, arm: str, model_id: str, seed_label: str, period: str) -> pd.DataFrame:
    rows = []
    for on_date, group in frame.groupby("prediction_date", sort=True):
        metrics = metric_values(group)
        rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "seed": seed_label, "period": period, "prediction_date": as_date(on_date), **metrics})
    return pd.DataFrame(rows)


def _macro_metrics(frame: pd.DataFrame, *, arm: str, model_id: str, seed_label: str, period: str, aggregation_level: str) -> dict[str, Any]:
    date_metrics = _date_metric_frame(frame, arm=arm, model_id=model_id, seed_label=seed_label, period=period)
    result: dict[str, Any] = {"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "seed": seed_label, "period": period, "aggregation_level": aggregation_level, "n_dates": int(len(date_metrics)), "prediction_coverage": np.nan}
    for metric in ("brier_score", "log_loss", "balanced_accuracy", "calibration_intercept", "calibration_slope", "auroc"):
        result[metric] = float(pd.to_numeric(date_metrics[metric], errors="coerce").mean()) if not date_metrics.empty else np.nan
    result["prediction_count"] = int(len(frame))
    result["class_0_count"] = int((frame["actual_class"].astype(int) == 0).sum()) if not frame.empty else 0
    result["class_1_count"] = int((frame["actual_class"].astype(int) == 1).sum()) if not frame.empty else 0
    return result


def performance_tables(predictions: Mapping[str, Mapping[str, pd.DataFrame]], data: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    by_arm_model_rows = []
    by_date_rows = []
    by_security_rows = []
    invalid_rows = []
    for arm, levels in predictions.items():
        collapsed = levels["collapsed"]
        seed = levels["seed"]
        feature_frame = data["features"][arm]
        final_feature = feature_frame[feature_frame["is_final_test"].astype(bool)] if not feature_frame.empty else feature_frame
        final_dates = set(final_feature["prediction_date"].tolist()) if not final_feature.empty else set()
        for model_id in MODEL_SPECS:
            model_collapsed = collapsed[collapsed["model_id"].astype(str).eq(model_id)] if not collapsed.empty else collapsed
            model_seed = seed[seed["model_id"].astype(str).eq(model_id)] if not seed.empty else seed
            for period in ("calibration", "final_test"):
                period_frame = model_collapsed[model_collapsed["prediction_date"].map(lambda value: _period_for_date(as_date(value) or TARGET_START) == period)] if not model_collapsed.empty else model_collapsed
                seed_period = model_seed[model_seed["prediction_date"].map(lambda value: _period_for_date(as_date(value) or TARGET_START) == period)] if not model_seed.empty else model_seed
                by_arm_model_rows.append(_macro_metrics(period_frame, arm=arm, model_id=model_id, seed_label="COLLAPSED", period=period, aggregation_level="seed_collapsed"))
                for seed_value in SEEDS:
                    seed_frame = seed_period[seed_period["seed"].astype(str).eq(str(seed_value))]
                    by_arm_model_rows.append(_macro_metrics(seed_frame, arm=arm, model_id=model_id, seed_label=str(seed_value), period=period, aggregation_level="seed"))
                if period == "final_test":
                    final_dates_present = len(set(period_frame["prediction_date"].tolist())) if not period_frame.empty else 0
                    class_count = len(set(period_frame["actual_class"].astype(int))) if not period_frame.empty else 0
                    reasons = []
                    if final_dates_present < MIN_FINAL_DATES:
                        reasons.append(f"final_test_dates {final_dates_present} < {MIN_FINAL_DATES}")
                    if len(period_frame) < MIN_FINAL_ROWS:
                        reasons.append(f"final_test_rows {len(period_frame)} < {MIN_FINAL_ROWS}")
                    if class_count < 2:
                        reasons.append("both target classes are not represented")
                    if period_frame.duplicated(["prediction_date", "security_id", "model_id"], keep=False).any():
                        reasons.append("duplicate collapsed prediction key")
                    if not period_frame.empty and not np.isfinite(period_frame["prediction_probability"].to_numpy(dtype=float)).all():
                        reasons.append("non-finite prediction probabilities")
                    invalid_rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "cell": "final_test", "valid": not reasons, "blocking_reasons": " | ".join(reasons) or None, "final_test_dates": final_dates_present, "final_test_rows": int(len(period_frame)), "class_count": class_count})

                    for on_date, group in period_frame.groupby("prediction_date", sort=True):
                        by_date_rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "prediction_date": as_date(on_date), "period": period, **metric_values(group)})
                    for security_id, group in period_frame.groupby("security_id", sort=True):
                        by_security_rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "security_id": security_id, "period": period, "**": ""} | metric_values(group))
    by_arm_model = pd.DataFrame(by_arm_model_rows)
    by_date = pd.DataFrame(by_date_rows)
    by_security = pd.DataFrame(by_security_rows).drop(columns=["**"], errors="ignore")
    invalid = pd.DataFrame(invalid_rows)
    return {"by_arm_model": by_arm_model, "by_date": by_date, "by_security": by_security, "invalid": invalid}


def _bh_adjust(values: Sequence[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    result = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    idx = np.flatnonzero(finite)
    if not len(idx):
        return result.tolist()
    order = idx[np.argsort(p[idx])]
    adjusted = np.empty(len(order), dtype=float)
    running = 1.0
    for rank in range(len(order) - 1, -1, -1):
        value = min(running, p[order[rank]] * len(order) / (rank + 1))
        running = value
        adjusted[rank] = value
    result[order] = adjusted
    return result.tolist()


def _date_brier(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    values = frame.copy()
    values["brier"] = (values["prediction_probability"].astype(float) - values["actual_class"].astype(int)) ** 2
    return values.groupby("prediction_date", sort=True)["brier"].mean()


def _common_security_date_predictions(predictions: Mapping[str, Mapping[str, pd.DataFrame]], data: Mapping[str, Any] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict full-arm predictions to common security/date observations.

    This is the intersection-control diagnostic requested by the pilot.  The
    models remain trained under their native full-arm protocols; only the
    evaluation rows are restricted to the common PIT/current security-date
    keys.  It therefore is explicitly diagnostic, not a causal decomposition.
    """
    if data is None or "members" not in data:
        return predictions["intersection"]["collapsed"], predictions["intersection"]["collapsed"]
    pit_members = data["members"]["pit"]
    current_members = data["members"]["current"]
    common_keys = {
        (as_date(row.get("date")), str(row.get("ticker_on_date")))
        for row in pit_members.to_dict("records")
    }
    common_keys &= {
        (as_date(row.get("date")), str(row.get("ticker_on_date")))
        for row in current_members.to_dict("records")
    }
    frames = []
    for arm in ("pit", "current"):
        frame = predictions[arm]["collapsed"].copy()
        if frame.empty:
            frames.append(frame)
            continue
        keys = [(as_date(d), str(t)) for d, t in zip(frame["prediction_date"], frame["ticker_on_date"])]
        frame = frame.loc[[key in common_keys for key in keys]].copy()
        frames.append(frame)
    return frames[0], frames[1]


def _moving_blocks(dates: list[date], block_length: int, rng: np.random.Generator) -> np.ndarray:
    if len(dates) < block_length:
        return np.asarray([], dtype=object)
    starts = rng.integers(0, len(dates) - block_length + 1, size=max(1, math.ceil(len(dates) / block_length)))
    sampled: list[date] = []
    for start in starts:
        sampled.extend(dates[int(start): int(start) + block_length])
    return np.asarray(sampled[: len(dates)], dtype=object)


def bootstrap_comparison(predictions: Mapping[str, Mapping[str, pd.DataFrame]], data: Mapping[str, Any] | None = None) -> pd.DataFrame:
    rows = []
    control_pit, control_current = _common_security_date_predictions(predictions, data)
    comparisons = (("full_universe", predictions["pit"]["collapsed"], predictions["current"]["collapsed"], "all valid full-arm rows"),
                   ("intersection_control", control_pit, control_current, "common PIT/current security-date evaluation rows; diagnostic only"))
    for comparison_name, pit_frame, current_frame, scope in comparisons:
        for model_id in MODEL_SPECS:
            pit_model = pit_frame[(pit_frame["model_id"] == model_id) & pit_frame["is_final_test"].astype(bool)]
            current_model = current_frame[(current_frame["model_id"] == model_id) & current_frame["is_final_test"].astype(bool)]
            pit_dates = _date_brier(pit_model)
            current_dates = _date_brier(current_model)
            paired = pd.concat([pit_dates.rename("pit_brier"), current_dates.rename("current_brier")], axis=1, join="inner").dropna()
            paired["delta_brier"] = paired["current_brier"] - paired["pit_brier"]
            dates = [as_date(value) for value in paired.index.tolist() if as_date(value)]
            for block_length in [PRIMARY_BLOCK_LENGTH, *ROBUSTNESS_BLOCK_LENGTHS]:
                if len(dates) < block_length or len(dates) < MIN_FINAL_DATES:
                    rows.append({"comparison": comparison_name, "comparison_scope": scope, "model_id": model_id, "block_length": block_length, "is_primary_block": block_length == PRIMARY_BLOCK_LENGTH, "valid": False, "point_estimate": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "raw_p_value": np.nan, "adjusted_p_value": np.nan, "n_dates": len(dates), "n_blocks": len(dates) // block_length, "reason": "insufficient final-test trading dates for moving blocks"})
                    continue
                values = paired["delta_brier"].to_numpy(dtype=float)
                point = float(values.mean())
                rng = np.random.default_rng(BOOTSTRAP_SEED + block_length * 100 + list(MODEL_SPECS).index(model_id) + (0 if comparison_name == "full_universe" else 1000))
                boot = []
                for _ in range(BOOTSTRAP_SAMPLES):
                    sampled_dates = _moving_blocks(dates, block_length, rng)
                    if len(sampled_dates) < len(dates):
                        continue
                    positions = [dates.index(value) for value in sampled_dates]
                    boot.append(float(values[positions].mean()))
                boot_values = np.asarray(boot, dtype=float)
                p_value = float(2 * min((boot_values <= 0).mean(), (boot_values >= 0).mean())) if len(boot_values) else np.nan
                rows.append({"comparison": comparison_name, "comparison_scope": scope, "model_id": model_id, "block_length": block_length, "is_primary_block": block_length == PRIMARY_BLOCK_LENGTH, "valid": bool(len(boot_values)), "point_estimate": point, "ci_lower": float(np.quantile(boot_values, 0.025)) if len(boot_values) else np.nan, "ci_upper": float(np.quantile(boot_values, 0.975)) if len(boot_values) else np.nan, "raw_p_value": p_value, "adjusted_p_value": np.nan, "n_dates": len(dates), "n_blocks": len(dates) // block_length, "reason": None if len(boot_values) else "bootstrap produced no samples"})
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_p_value"] = np.nan
        for block_length, group in result[result["valid"].astype(bool)].groupby("block_length"):
            adjusted = _bh_adjust(group["raw_p_value"].to_numpy(dtype=float))
            result.loc[group.index, "adjusted_p_value"] = adjusted
    return result


def universe_reports(data: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    members = data["members"]
    observations = data["observations"]
    rows = []
    overlap_rows = []
    for on_date in _eligible_dates(data["inputs"] if "inputs" in data else data):
        pit = members["pit"][members["pit"]["date"] == on_date]
        current = members["current"][members["current"]["date"] == on_date]
        pit_tickers = set(pit["ticker_on_date"].astype(str))
        current_tickers = set(current["ticker_on_date"].astype(str))
        common = pit_tickers.intersection(current_tickers)
        pit_only = sorted(pit_tickers - current_tickers)
        current_only = sorted(current_tickers - pit_tickers)
        interval = _interval_for_date(data["inputs"]["eligibility"], on_date)
        pit_obs = observations["pit"][observations["pit"]["prediction_date"] == on_date]
        current_obs = observations["current"][observations["current"]["prediction_date"] == on_date]
        intersection_obs = observations["intersection"][observations["intersection"]["prediction_date"] == on_date]
        raw_price_coverage = float(len(data["inputs"]["pit_panel"][data["inputs"]["pit_panel"]["date"] == on_date]) / len(pit)) if len(pit) else 0.0
        row = {
            "report_level": "date",
            "date": on_date,
            "interval_start": as_date(interval.get("interval_start")),
            "interval_end_exclusive": as_date(interval.get("interval_end")),
            "eligibility_interval_id": interval.get("eligibility_interval_id"),
            "pit_constituent_count": len(pit_tickers),
            "current_universe_constituent_count": len(current_tickers),
            "intersection_count": len(common),
            "union_count": len(pit_tickers.union(current_tickers)),
            "jaccard_overlap": len(common) / len(pit_tickers.union(current_tickers)) if pit_tickers.union(current_tickers) else 0.0,
            "pit_only_securities": ";".join(pit_only),
            "current_only_securities": ";".join(current_only),
            "additions": ";".join(current_only),
            "removals": ";".join(pit_only),
            "raw_price_coverage": raw_price_coverage,
            "pit_valid_forecasting_observations": int(len(pit_obs)),
            "current_valid_forecasting_observations": int(len(current_obs)),
            "intersection_valid_forecasting_observations": int(len(intersection_obs)),
        }
        rows.append(row)
        overlap_rows.append({
            "date": on_date,
            "market": "Thailand",
            "index_id": "SET50",
            "pit_constituent_count": len(pit_tickers),
            "current_constituent_count": len(current_tickers),
            "intersection_count": len(common),
            "union_count": len(pit_tickers.union(current_tickers)),
            "jaccard_overlap": row["jaccard_overlap"],
            "pit_only_securities": ";".join(pit_only),
            "current_only_securities": ";".join(current_only),
            "raw_price_coverage": raw_price_coverage,
            "valid_forecasting_observations_pit": len(pit_obs),
            "valid_forecasting_observations_current": len(current_obs),
            "valid_forecasting_observations_intersection": len(intersection_obs),
        })
    daily = pd.DataFrame(rows)
    interval_rows = []
    if not daily.empty:
        for interval_id, group in daily.groupby("eligibility_interval_id", sort=True):
            interval_rows.append({
                "report_level": "interval",
                "date": None,
                "interval_start": group["interval_start"].min(),
                "interval_end_exclusive": group["interval_end_exclusive"].max(),
                "eligibility_interval_id": interval_id,
                "pit_constituent_count": float(group["pit_constituent_count"].mean()),
                "current_universe_constituent_count": float(group["current_universe_constituent_count"].mean()),
                "intersection_count": float(group["intersection_count"].mean()),
                "union_count": float(group["union_count"].mean()),
                "jaccard_overlap": float(group["jaccard_overlap"].mean()),
                "pit_only_securities": ";".join(sorted(set(";".join(group["pit_only_securities"].astype(str)).split(";")) - {""})),
                "current_only_securities": ";".join(sorted(set(";".join(group["current_only_securities"].astype(str)).split(";")) - {""})),
                "additions": ";".join(sorted(set(";".join(group["additions"].astype(str)).split(";")) - {""})),
                "removals": ";".join(sorted(set(";".join(group["removals"].astype(str)).split(";")) - {""})),
                "raw_price_coverage": float(group["raw_price_coverage"].mean()),
                "pit_valid_forecasting_observations": int(group["pit_valid_forecasting_observations"].sum()),
                "current_valid_forecasting_observations": int(group["current_valid_forecasting_observations"].sum()),
                "intersection_valid_forecasting_observations": int(group["intersection_valid_forecasting_observations"].sum()),
            })
    descriptive = pd.concat([daily, pd.DataFrame(interval_rows)], ignore_index=True) if interval_rows else daily
    turnover_rows = []
    previous_pit: set[str] | None = None
    previous_interval = None
    for interval_id, group in daily.groupby("eligibility_interval_id", sort=True):
        first = group.sort_values("date").iloc[0]
        current_set = set(str(first["current_only_securities"]).split(";") if first["current_only_securities"] else [])
        pit_set = set(str(first["pit_only_securities"]).split(";") if first["pit_only_securities"] else [])
        members_set = set(members["pit"].loc[members["pit"]["eligibility_interval_id"] == interval_id, "ticker_on_date"].astype(str))
        if previous_pit is not None:
            for ticker in sorted(members_set - previous_pit):
                turnover_rows.append({"event_date": first["date"], "interval_id": interval_id, "event_type": "PIT_ADDITION", "ticker": ticker, "previous_interval_id": previous_interval})
            for ticker in sorted(previous_pit - members_set):
                turnover_rows.append({"event_date": first["date"], "interval_id": interval_id, "event_type": "PIT_REMOVAL", "ticker": ticker, "previous_interval_id": previous_interval})
        for ticker in sorted(current_set):
            turnover_rows.append({"event_date": first["date"], "interval_id": interval_id, "event_type": "CURRENT_ONLY_RELATIVE_TO_PIT", "ticker": ticker, "previous_interval_id": previous_interval})
        for ticker in sorted(pit_set):
            turnover_rows.append({"event_date": first["date"], "interval_id": interval_id, "event_type": "PIT_ONLY_RELATIVE_TO_CURRENT", "ticker": ticker, "previous_interval_id": previous_interval})
        previous_pit, previous_interval = members_set, interval_id
    overlap = pd.DataFrame(overlap_rows)
    turnover = pd.DataFrame(turnover_rows)
    return {"descriptive": descriptive, "overlap": overlap, "turnover": turnover}


def save_metrics_and_reports(data: Mapping[str, Any], model_results: Mapping[str, Any]) -> dict[str, Any]:
    reports = universe_reports(data)
    for name, frame in reports.items():
        stem = "composition" if name == "descriptive" else "overlap" if name == "overlap" else "security_turnover"
        path = PIT_ROOT / "reports" / f"set50_universe_{stem}_v4.csv"
        write_csv(path, frame)
        artifact_manifest(path, f"SET50 v4 universe {name} report", len(frame))
        # Keep the user-facing required turnover filename as a versioned alias.
        if name == "turnover":
            required_path = PIT_ROOT / "reports" / "set50_security_turnover_v4.csv"
            write_csv(required_path, frame)
            artifact_manifest(required_path, "SET50 v4 security turnover report (required filename)", len(frame))

    predictions = model_results["predictions"]
    perf = performance_tables(predictions, data)
    bootstrap = bootstrap_comparison(predictions, data)

    # Model ranking and seed/bootstrap uncertainty.
    ranking_rows = []
    seed_robustness_rows = []
    for arm in ("pit", "current", "intersection"):
        collapsed = predictions[arm]["collapsed"]
        seed = predictions[arm]["seed"]
        final = collapsed[collapsed["is_final_test"].astype(bool)] if not collapsed.empty else collapsed
        metrics_by_model = {}
        for model_id in MODEL_SPECS:
            group = final[final["model_id"] == model_id] if not final.empty else final
            metrics_by_model[model_id] = _macro_metrics(group, arm=arm, model_id=model_id, seed_label="COLLAPSED", period="final_test", aggregation_level="seed_collapsed")
            for seed_value in SEEDS:
                seed_group = seed[(seed["model_id"] == model_id) & (seed["seed"].astype(str) == str(seed_value)) & seed["is_final_test"].astype(bool)] if not seed.empty else seed
                seed_metric = _macro_metrics(seed_group, arm=arm, model_id=model_id, seed_label=str(seed_value), period="final_test", aggregation_level="seed")
                seed_robustness_rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "seed": seed_value, "period": "final_test", **{key: value for key, value in seed_metric.items() if key not in {"arm", "universe_semantics", "model_id", "seed", "period", "aggregation_level"}}})
        ordered = sorted(MODEL_SPECS, key=lambda model_id: (float(metrics_by_model[model_id].get("brier_score", np.inf)) if np.isfinite(metrics_by_model[model_id].get("brier_score", np.nan)) else np.inf, model_id))
        ranks = {model_id: rank + 1 for rank, model_id in enumerate(ordered)}
        brier_values = [metrics_by_model[model_id].get("brier_score", np.nan) for model_id in MODEL_SPECS]
        # Seed rank dispersion.
        seed_rank_std: dict[str, float] = {}
        for model_id in MODEL_SPECS:
            vals = []
            for seed_value in SEEDS:
                row = next((item for item in seed_robustness_rows if item["arm"] == arm and item["model_id"] == model_id and item["seed"] == seed_value), None)
                vals.append(row.get("brier_score", np.nan) if row else np.nan)
            seed_rank_std[model_id] = float(np.nanstd(np.argsort(np.argsort(vals)) + 1)) if np.isfinite(vals).any() else np.nan
        for model_id in MODEL_SPECS:
            ranking_rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "final_brier_score": metrics_by_model[model_id].get("brier_score"), "final_log_loss": metrics_by_model[model_id].get("log_loss"), "brier_rank": ranks.get(model_id), "seed_rank_std": seed_rank_std.get(model_id), "top_model_agreement": None, "rank_correlation_pit_current": None, "bootstrap_rank_std": None, "bootstrap_top_probability": None})
    ranking = pd.DataFrame(ranking_rows)
    seed_robustness = pd.DataFrame(seed_robustness_rows)
    # Fill rank agreement on the full PIT/current pair.
    rank_by_arm = ranking.pivot(index="model_id", columns="arm", values="brier_rank") if not ranking.empty else pd.DataFrame()
    rank_corr = float(rank_by_arm["pit"].corr(rank_by_arm["current"], method="spearman")) if {"pit", "current"}.issubset(rank_by_arm.columns) else np.nan
    pit_top = ranking[ranking["arm"] == "pit"].sort_values("brier_rank").iloc[0]["model_id"] if not ranking[ranking["arm"] == "pit"].empty else None
    current_top = ranking[ranking["arm"] == "current"].sort_values("brier_rank").iloc[0]["model_id"] if not ranking[ranking["arm"] == "current"].empty else None
    ranking.loc[ranking["arm"].isin(["pit", "current"]), "rank_correlation_pit_current"] = rank_corr
    ranking.loc[ranking["arm"].isin(["pit", "current"]), "top_model_agreement"] = pit_top == current_top
    # Bootstrap rank uncertainty across primary bootstrap date resamples.
    if not bootstrap.empty:
        for arm in ("pit", "current", "intersection"):
            frame = predictions[arm]["collapsed"]
            final = frame[frame["is_final_test"].astype(bool)] if not frame.empty else frame
            date_sets = sorted(set(as_date(value) for value in final["prediction_date"].tolist() if as_date(value))) if not final.empty else []
            if len(date_sets) >= PRIMARY_BLOCK_LENGTH:
                rank_samples = {model_id: [] for model_id in MODEL_SPECS}
                rng = np.random.default_rng(BOOTSTRAP_SEED + 999 + list(("pit", "current", "intersection")).index(arm))
                date_metric_by_model = {model_id: _date_brier(final[final["model_id"] == model_id]) for model_id in MODEL_SPECS}
                for _ in range(BOOTSTRAP_SAMPLES):
                    sampled = _moving_blocks(date_sets, PRIMARY_BLOCK_LENGTH, rng)
                    if len(sampled) < len(date_sets):
                        continue
                    scores = {model_id: float(date_metric_by_model[model_id].reindex(sampled).mean()) for model_id in MODEL_SPECS}
                    order = sorted(MODEL_SPECS, key=lambda model_id: (scores[model_id], model_id))
                    for rank, model_id in enumerate(order, 1):
                        rank_samples[model_id].append(rank)
                for model_id in MODEL_SPECS:
                    mask = (ranking["arm"] == arm) & (ranking["model_id"] == model_id)
                    values = np.asarray(rank_samples[model_id], dtype=float)
                    if len(values):
                        ranking.loc[mask, "bootstrap_rank_std"] = float(values.std())
                        ranking.loc[mask, "bootstrap_top_probability"] = float((values == 1).mean())

    comp_rows = []
    # Full-universe delta and intersection-control delta using the same date macro estimator.
    for model_id in MODEL_SPECS:
        pit = perf["by_date"][(perf["by_date"]["arm"] == "pit") & (perf["by_date"]["model_id"] == model_id)]
        current = perf["by_date"][(perf["by_date"]["arm"] == "current") & (perf["by_date"]["model_id"] == model_id)]
        inter = perf["by_date"][(perf["by_date"]["arm"] == "intersection") & (perf["by_date"]["model_id"] == model_id)]
        pair = pit[["prediction_date", "brier_score", "log_loss", "balanced_accuracy", "calibration_intercept", "calibration_slope"]].merge(current[["prediction_date", "brier_score", "log_loss", "balanced_accuracy", "calibration_intercept", "calibration_slope"]], on="prediction_date", suffixes=("_pit", "_current"))
        # Evaluate the full-arm predictions on common security/date keys for
        # the intersection-control comparison.  The separately trained Arm C
        # remains available as its own diagnostic arm in all primary tables.
        control_pit, control_current = _common_security_date_predictions(predictions, data)
        control_pit = control_pit[control_pit["model_id"] == model_id]
        control_current = control_current[control_current["model_id"] == model_id]
        control_pit_date = _date_metric_frame(control_pit[control_pit["is_final_test"].astype(bool)], arm="pit", model_id=model_id, seed_label="CONTROL", period="final_test")
        control_current_date = _date_metric_frame(control_current[control_current["is_final_test"].astype(bool)], arm="current", model_id=model_id, seed_label="CONTROL", period="final_test")
        inter_pair = control_pit_date[["prediction_date", "brier_score", "log_loss", "balanced_accuracy", "calibration_intercept", "calibration_slope"]].merge(control_current_date[["prediction_date", "brier_score", "log_loss", "balanced_accuracy", "calibration_intercept", "calibration_slope"]], on="prediction_date", suffixes=("_pit", "_current"))
        if not pair.empty:
            for _, row in pair.iterrows():
                comp_rows.append({"comparison": "full_universe", "model_id": model_id, "prediction_date": row["prediction_date"], "delta_brier_current_minus_pit": row["brier_score_current"] - row["brier_score_pit"], "delta_log_loss_current_minus_pit": row["log_loss_current"] - row["log_loss_pit"], "delta_balanced_accuracy_current_minus_pit": row["balanced_accuracy_current"] - row["balanced_accuracy_pit"], "delta_calibration_intercept_current_minus_pit": row["calibration_intercept_current"] - row["calibration_intercept_pit"], "delta_calibration_slope_current_minus_pit": row["calibration_slope_current"] - row["calibration_slope_pit"]})
        if not inter_pair.empty:
            for _, row in inter_pair.iterrows():
                comp_rows.append({"comparison": "intersection_control", "model_id": model_id, "prediction_date": row["prediction_date"], "delta_brier_current_minus_pit": row["brier_score_current"] - row["brier_score_pit"], "delta_log_loss_current_minus_pit": row["log_loss_current"] - row["log_loss_pit"], "delta_balanced_accuracy_current_minus_pit": row["balanced_accuracy_current"] - row["balanced_accuracy_pit"], "delta_calibration_intercept_current_minus_pit": row["calibration_intercept_current"] - row["calibration_intercept_pit"], "delta_calibration_slope_current_minus_pit": row["calibration_slope_current"] - row["calibration_slope_pit"], "comparison_scope": "common PIT/current security-date evaluation rows; diagnostic only"})
    pairwise = pd.DataFrame(comp_rows)
    diagnostic_rows = []
    for model_id in MODEL_SPECS:
        full = bootstrap[(bootstrap["comparison"] == "full_universe") & (bootstrap["model_id"] == model_id) & (bootstrap["block_length"] == PRIMARY_BLOCK_LENGTH)]
        control = bootstrap[(bootstrap["comparison"] == "intersection_control") & (bootstrap["model_id"] == model_id) & (bootstrap["block_length"] == PRIMARY_BLOCK_LENGTH)]
        if not full.empty and not control.empty:
            diagnostic_rows.append({"model_id": model_id, "full_universe_delta_brier": full.iloc[0]["point_estimate"], "intersection_control_delta_brier": control.iloc[0]["point_estimate"], "full_minus_intersection_delta_brier": full.iloc[0]["point_estimate"] - control.iloc[0]["point_estimate"], "interpretation": "diagnostic decomposition only; not causal"})
    diagnostic = pd.DataFrame(diagnostic_rows)
    invalid = perf["invalid"].copy()
    if not model_results["invalid_training"].empty:
        invalid = pd.concat([invalid, model_results["invalid_training"].assign(valid=False)], ignore_index=True, sort=False)
    return {"universe": reports, "performance": perf, "bootstrap": bootstrap, "ranking": ranking, "seed_robustness": seed_robustness, "pairwise": pairwise, "composition": diagnostic, "invalid": invalid}


def persist_analysis_outputs(analysis: Mapping[str, Any]) -> None:
    output_map = {
        "universe_descriptive_summary": (analysis["universe"]["descriptive"], "SET50 v4 descriptive universe summary"),
        "forecast_performance_by_arm_model": (analysis["performance"]["by_arm_model"], "SET50 v4 performance by universe arm and model"),
        "forecast_performance_by_date": (analysis["performance"]["by_date"], "SET50 v4 equal-date performance records"),
        "forecast_performance_by_security": (analysis["performance"]["by_security"], "SET50 v4 security-level performance records"),
        "current_vs_pit_pairwise_comparisons": (analysis["pairwise"], "SET50 v4 paired current-versus-PIT date comparisons"),
        "model_ranking_comparison": (analysis["ranking"], "SET50 v4 model ranking comparison and uncertainty"),
        "composition_diagnostic": (analysis["composition"], "SET50 v4 intersection-control diagnostic decomposition"),
        "seed_robustness": (analysis["seed_robustness"], "SET50 v4 seed-level robustness summary"),
        "bootstrap_inference": (analysis["bootstrap"], "SET50 v4 moving-block bootstrap inference"),
        "invalid_cells": (analysis["invalid"], "SET50 v4 invalid model-universe cells and reasons"),
    }
    for stem, (frame, purpose) in output_map.items():
        csv_path = PIT_ROOT / "reports" / f"set50_{stem}_v4.csv"
        parquet_path = PIT_ROOT / "reports" / f"set50_{stem}_v4.parquet"
        write_csv(csv_path, frame)
        write_parquet(parquet_path, frame)
        artifact_manifest(csv_path, purpose + " (CSV)", len(frame))
        artifact_manifest(parquet_path, purpose + " (Parquet)", len(frame))


def _model_cell_status(analysis: Mapping[str, Any]) -> pd.DataFrame:
    invalid = analysis["invalid"].copy()
    rows = []
    for arm in ("pit", "current", "intersection"):
        for model_id in MODEL_SPECS:
            match = invalid[(invalid.get("arm", pd.Series(dtype=str)).astype(str) == arm) & (invalid.get("model_id", pd.Series(dtype=str)).astype(str) == model_id)] if not invalid.empty else pd.DataFrame()
            final_rows = analysis["performance"]["by_arm_model"][(analysis["performance"]["by_arm_model"]["arm"] == arm) & (analysis["performance"]["by_arm_model"]["model_id"] == model_id) & (analysis["performance"]["by_arm_model"]["period"] == "final_test") & (analysis["performance"]["by_arm_model"]["aggregation_level"] == "seed_collapsed")]
            row = final_rows.iloc[0].to_dict() if not final_rows.empty else {}
            rows.append({"arm": arm, "universe_semantics": ARM_SPECS[arm]["universe_semantics"], "model_id": model_id, "valid": bool(not match.get("valid", pd.Series(dtype=bool)).eq(False).any()) and int(row.get("n_dates", 0) or 0) >= MIN_FINAL_DATES and int(row.get("prediction_count", 0) or 0) >= MIN_FINAL_ROWS and int(row.get("class_0_count", 0) or 0) > 0 and int(row.get("class_1_count", 0) or 0) > 0, "blocking_reasons": "; ".join(match.get("blocking_reasons", pd.Series(dtype=str)).dropna().astype(str).tolist()) or None, "final_test_dates": int(row.get("n_dates", 0) or 0), "final_test_rows": int(row.get("prediction_count", 0) or 0), "class_0_count": int(row.get("class_0_count", 0) or 0), "class_1_count": int(row.get("class_1_count", 0) or 0)})
    return pd.DataFrame(rows)


def build_status_payload(inputs: Mapping[str, Any], identity: pd.DataFrame, data: Mapping[str, Any], model_results: Mapping[str, Any], analysis: Mapping[str, Any], tests: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cells = _model_cell_status(analysis)
    coverage = _coverage_stats(inputs)
    features = data["features"]
    predictions = model_results["predictions"]
    universe = analysis["universe"]["overlap"]
    mean_jaccard = float(pd.to_numeric(universe["jaccard_overlap"], errors="coerce").mean()) if not universe.empty else 0.0
    turnover_count = int(len(analysis["universe"]["turnover"]))
    status = "COMPLETE" if bool(not cells.empty and cells["valid"].astype(bool).all() and not analysis["bootstrap"].empty and analysis["bootstrap"].query("is_primary_block == True and valid == True").shape[0] >= len(MODEL_SPECS) * 2 and (tests or {}).get("failed", 1) == 0) else "PARTIAL"
    return {
        "status": status,
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "overall_five_market_pit_status": "PARTIAL",
        "status_reconciliation": {
            "full_history_coverage": coverage["full_reconstructed"],
            "prior_report_0_920_definition": coverage["old_report_0_920"],
            "eligible_interval_coverage": coverage["eligible_intervals"],
            "snapshot_to_price_join": coverage["snapshot_to_price_join"],
            "pit_raw_ohlc_rows_v3": int(len(inputs.get("pit_panel", []))),
            "matched_current_universe_rows_v3": int(len(inputs.get("current_panel", []))),
            "forecasting_panel_coverage": {arm: {"observation_rows": int(len(data["observations"][arm])), "feature_rows": int(len(features[arm])), "final_feature_rows": int(features[arm]["is_final_test"].astype(bool).sum()) if not features[arm].empty else 0} for arm in ("pit", "current", "intersection")},
            "resolved_unique_securities": 221,
            "unresolved_unique_securities": int(identity["published_name"].nunique()) if not identity.empty else 0,
            "unresolved_event_records": int(len(identity)),
        },
        "universe_composition": {
            "pit_securities": int(data["members"]["pit"]["ticker_on_date"].nunique()),
            "current_securities": int(data["members"]["current"]["ticker_on_date"].nunique()),
            "common_securities": int(data["members"]["intersection"]["ticker_on_date"].nunique()),
            "mean_jaccard_overlap": mean_jaccard,
            "turnover_events": turnover_count,
        },
        "forecasting_data": {
            "training_rows_by_arm": {arm: int(features[arm][features[arm]["prediction_date"].map(lambda value: _period_for_date(as_date(value) or TARGET_START) == "initial_training")].shape[0]) if not features[arm].empty else 0 for arm in ("pit", "current", "intersection")},
            "calibration_rows_by_arm": {arm: int(features[arm][features[arm]["prediction_date"].map(lambda value: _period_for_date(as_date(value) or TARGET_START) == "calibration")].shape[0]) if not features[arm].empty else 0 for arm in ("pit", "current", "intersection")},
            "final_test_rows_by_arm": {arm: int(features[arm][features[arm]["is_final_test"].astype(bool)].shape[0]) if not features[arm].empty else 0 for arm in ("pit", "current", "intersection")},
            "prediction_rows_by_arm_and_model": {arm: {model_id: int(len(predictions[arm]["collapsed"][predictions[arm]["collapsed"]["model_id"] == model_id])) if not predictions[arm]["collapsed"].empty else 0 for model_id in MODEL_SPECS} for arm in ("pit", "current", "intersection")},
            "valid_model_cells": int(cells["valid"].astype(bool).sum()) if not cells.empty else 0,
            "invalid_model_cells": int((~cells["valid"].astype(bool)).sum()) if not cells.empty else 0,
        },
        "performance": {"by_arm_model": analysis["performance"]["by_arm_model"].to_dict("records"), "pairwise": analysis["pairwise"].to_dict("records"), "ranking": analysis["ranking"].to_dict("records"), "composition_control": analysis["composition"].to_dict("records")},
        "inference": analysis["bootstrap"].to_dict("records"),
        "tests": dict(tests or {"passed": None, "failed": None, "status": "NOT_RECORDED"}),
        "limitations": ["one market only", "approximately one year of PIT coverage", "current-constituent comparator is applied backward and is not historically investable", "missing-price observations remain explicit", "no profitability test", "no causal interpretation"],
    }


def write_status_report(payload: Mapping[str, Any]) -> tuple[Path, Path]:
    json_path = REPORT_ROOT / "SET50_PIT_FORECASTING_PILOT_V4.json"
    md_path = REPORT_ROOT / "SET50_PIT_FORECASTING_PILOT_V4.md"
    write_json(json_path, payload)
    reconciliation = payload.get("status_reconciliation", {})
    composition = payload.get("universe_composition", {})
    forecast = payload.get("forecasting_data", {})
    performance = payload.get("performance", {})
    final_metrics = [
        row for row in performance.get("by_arm_model", [])
        if row.get("period") == "final_test" and row.get("aggregation_level") == "seed_collapsed"
    ]
    primary_inference = [
        row for row in payload.get("inference", [])
        if row.get("comparison") == "full_universe" and bool(row.get("is_primary_block"))
    ]
    ranking = performance.get("ranking", [])
    composition_control = performance.get("composition_control", [])
    lines = [
        "# SET50 PIT forecasting pilot v4",
        "",
        f"Status: **{payload.get('status')}**",
        "",
        "This is a controlled validation pilot. It contains no manuscript sections, profitability claims, causal interpretation, or five-market completion claim.",
        "",
        "## STATUS RECONCILIATION",
        "",
        f"- Full reconstructed-history coverage: {json.dumps(reconciliation.get('full_history_coverage', {}), sort_keys=True)}.",
        f"- Prior v3 0.920 definition: {json.dumps(reconciliation.get('prior_report_0_920_definition', {}), sort_keys=True)}.",
        f"- Eligible-interval coverage: {json.dumps(reconciliation.get('eligible_interval_coverage', []), sort_keys=True)}.",
        f"- PIT snapshot-to-price join coverage: {json.dumps(reconciliation.get('snapshot_to_price_join', {}), sort_keys=True)}.",
        f"- Validated v3 PIT raw-OHLC rows: {reconciliation.get('pit_raw_ohlc_rows_v3')}; matched current-universe rows: {reconciliation.get('matched_current_universe_rows_v3')}.",
        f"- Forecasting-panel coverage: {json.dumps(reconciliation.get('forecasting_panel_coverage', {}), sort_keys=True)}.",
        f"- Resolved unique securities: {reconciliation.get('resolved_unique_securities')}; unresolved unique published names: {reconciliation.get('unresolved_unique_securities')}; unresolved event records: {reconciliation.get('unresolved_event_records')}.",
        "",
        "## UNIVERSE COMPOSITION",
        "",
        f"- PIT securities: {composition.get('pit_securities')}; current-universe securities: {composition.get('current_securities')}; common securities: {composition.get('common_securities')}.",
        f"- Mean daily Jaccard overlap: {composition.get('mean_jaccard_overlap')}; turnover/event rows: {composition.get('turnover_events')}.",
        "",
        "## FORECASTING DATA",
        "",
        f"- Training rows by arm: {json.dumps(forecast.get('training_rows_by_arm', {}), sort_keys=True)}.",
        f"- Calibration rows by arm: {json.dumps(forecast.get('calibration_rows_by_arm', {}), sort_keys=True)}.",
        f"- Final-test rows by arm: {json.dumps(forecast.get('final_test_rows_by_arm', {}), sort_keys=True)}.",
        f"- Prediction rows by arm/model: {json.dumps(forecast.get('prediction_rows_by_arm_and_model', {}), sort_keys=True)}.",
        f"- Valid model cells: {forecast.get('valid_model_cells')}; invalid model cells: {forecast.get('invalid_model_cells')}.",
        "",
        "## PERFORMANCE",
        "",
        "The machine-readable performance tables provide Brier score, log loss, balanced accuracy, calibration intercept/slope, AUROC where defined, and prediction coverage at seed, seed-collapsed, security, date, and final-test levels.",
        "",
        "Final-test seed-collapsed metrics by arm/model:",
        *[
            f"- {row.get('arm')}/{row.get('model_id')}: Brier={row.get('brier_score')}; log loss={row.get('log_loss')}; balanced accuracy={row.get('balanced_accuracy')}; calibration intercept={row.get('calibration_intercept')}; calibration slope={row.get('calibration_slope')}; dates={row.get('n_dates')}; rows={row.get('prediction_count')}."
            for row in final_metrics
        ],
        "",
        "## CURRENT VERSUS PIT",
        "",
        "Paired moving-block bootstrap estimates, confidence intervals, adjusted p-values, model-ranking agreement, and intersection-control diagnostics are retained in the v4 report tables.",
        *[
            f"- Full-universe current-minus-PIT Brier ({row.get('model_id')}, block {row.get('block_length')}): estimate={row.get('point_estimate')}; 95% CI=[{row.get('ci_lower')}, {row.get('ci_upper')}]; raw p={row.get('raw_p_value')}; BH-adjusted p={row.get('adjusted_p_value')}."
            for row in primary_inference
        ],
        *[
            f"- Model-ranking comparison: {row.get('arm')}/{row.get('model_id')} rank={row.get('brier_rank')}; PIT/current rank correlation={row.get('rank_correlation_pit_current')}; top-model agreement={row.get('top_model_agreement')}."
            for row in ranking if row.get("arm") in {"pit", "current"}
        ],
        *[
            f"- Composition-control diagnostic ({row.get('model_id')}): full delta={row.get('full_universe_delta_brier')}; common-security/date control delta={row.get('intersection_control_delta_brier')}; difference={row.get('full_minus_intersection_delta_brier')}; interpretation={row.get('interpretation')}."
            for row in composition_control
        ],
        "",
        "## TESTS",
        "",
        f"- Passed: {payload.get('tests', {}).get('passed')}; failed: {payload.get('tests', {}).get('failed')}; status: {payload.get('tests', {}).get('status')}.",
        "",
        "## LIMITATIONS",
        "",
        *[f"- {item}." for item in payload.get("limitations", [])],
        "",
        "The overall five-market PIT project remains **PARTIAL**.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    artifact_manifest(md_path, "SET50 PIT forecasting pilot v4 status report", None)
    artifact_manifest(json_path, "SET50 PIT forecasting pilot v4 status JSON", None)
    return md_path, json_path


def build_all() -> dict[str, Any]:
    ensure_dirs()
    inputs = load_v3_inputs()
    identity = reconcile_identity()
    config = freeze_config(inputs)
    data = write_arm_data(inputs, identity)
    model_results = run_models(data, config)
    analysis = save_metrics_and_reports(data, model_results)
    persist_analysis_outputs(analysis)
    payload = build_status_payload(inputs, identity, data, model_results, analysis)
    write_status_report(payload)
    run_manifest = {
        "pipeline": "set50_pit_forecasting_pilot_v4",
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "target_scope": {"market": "Thailand", "index_id": "SET50", "start": iso(TARGET_START), "end_exclusive": iso(TARGET_END_EXCLUSIVE)},
        "config_path": "research_data/point_in_time/config/set50_pit_forecasting_pilot_v4.yaml",
        "config_sha256": sha256_file(PIT_ROOT / "config" / "set50_pit_forecasting_pilot_v4.yaml"),
        "input_checksums": input_checksums(),
        "artifact_paths": sorted(str(path.relative_to(PIT_ROOT)) for path in PIT_ROOT.rglob("*") if path.is_file() and "v4" in path.name),
        "created_at": now_utc(),
    }
    run_manifest_path = PIT_ROOT / "manifests" / "set50_pit_forecasting_pilot_v4_run.manifest.json"
    write_json(run_manifest_path, run_manifest)
    return {"status": payload["status"], "identity_records": len(identity), "pit_feature_rows": len(data["features"]["pit"]), "current_feature_rows": len(data["features"]["current"]), "intersection_feature_rows": len(data["features"]["intersection"]), "pit_prediction_rows": len(model_results["predictions"]["pit"]["collapsed"]), "current_prediction_rows": len(model_results["predictions"]["current"]["collapsed"]), "intersection_prediction_rows": len(model_results["predictions"]["intersection"]["collapsed"]), "valid_model_cells": int(_model_cell_status(analysis)["valid"].astype(bool).sum()), "invalid_model_cells": int((~_model_cell_status(analysis)["valid"].astype(bool)).sum()), "status_report": str(REPORT_ROOT / "SET50_PIT_FORECASTING_PILOT_V4.md")}


def finalize_tests(passed: int, failed: int, command: str) -> dict[str, Any]:
    json_path = REPORT_ROOT / "SET50_PIT_FORECASTING_PILOT_V4.json"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["tests"] = {"passed": int(passed), "failed": int(failed), "status": "PASS" if failed == 0 else "FAIL", "command": command, "recorded_at": now_utc()}
    # The pilot status is separate from the overall five-market status.  Once
    # the recorded test suite passes, promote only the SET50 pilot if its
    # locked model-cell and bootstrap gates are already satisfied.
    valid_cells = int(payload.get("forecasting_data", {}).get("valid_model_cells", 0) or 0)
    invalid_cells = int(payload.get("forecasting_data", {}).get("invalid_model_cells", 0) or 0)
    primary_bootstrap = [
        row for row in payload.get("inference", [])
        if bool(row.get("is_primary_block")) and bool(row.get("valid"))
    ]
    payload["status"] = "COMPLETE" if failed == 0 and valid_cells == len(MODEL_SPECS) * 3 and invalid_cells == 0 and len(primary_bootstrap) >= len(MODEL_SPECS) * 2 else "PARTIAL"
    write_status_report(payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("reconcile")
    sub.add_parser("build-data")
    sub.add_parser("run-models")
    sub.add_parser("build-all")
    final = sub.add_parser("finalize-tests")
    final.add_argument("--passed", required=True, type=int)
    final.add_argument("--failed", required=True, type=int)
    final.add_argument("--command", required=True)
    args = parser.parse_args(argv)
    if args.command == "reconcile":
        inputs = load_v3_inputs(); identity = reconcile_identity(); config = freeze_config(inputs); payload = write_status_reconciliation(inputs, identity)
        print(json.dumps({"status": "RECONCILED", "identity_records": len(identity), "config": str(PIT_ROOT / "config" / "set50_pit_forecasting_pilot_v4.yaml"), "reconciliation": payload}, indent=2, default=json_default))
        return 0
    if args.command == "build-data":
        inputs = load_v3_inputs(); identity = reconcile_identity(); freeze_config(inputs); data = write_arm_data(inputs, identity)
        print(json.dumps({"status": "DATA_BUILT", "observation_rows": {arm: len(data["observations"][arm]) for arm in ("pit", "current", "intersection")}, "feature_rows": {arm: len(data["features"][arm]) for arm in ("pit", "current", "intersection")}}, indent=2, default=json_default))
        return 0
    if args.command == "run-models":
        inputs = load_v3_inputs(); identity = reconcile_identity(); config = freeze_config(inputs); data = write_arm_data(inputs, identity); model_results = run_models(data, config); analysis = save_metrics_and_reports(data, model_results); persist_analysis_outputs(analysis); payload = build_status_payload(inputs, identity, data, model_results, analysis); write_status_report(payload)
        print(json.dumps({"status": payload["status"], "valid_model_cells": payload["forecasting_data"]["valid_model_cells"], "invalid_model_cells": payload["forecasting_data"]["invalid_model_cells"]}, indent=2, default=json_default))
        return 0
    if args.command == "finalize-tests":
        print(json.dumps(finalize_tests(args.passed, args.failed, args.command), indent=2, default=json_default))
        return 0
    print(json.dumps(build_all(), indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
