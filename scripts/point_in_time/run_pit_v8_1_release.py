#!/usr/bin/env python3
"""Build and validate the publication-hardened ASEAN PIT V8.1 release.

The V8 forecasting execution is immutable.  V8.1 validates and reuses those
predictions when every input checksum and prediction invariant is unchanged,
then adds evidence hardening, support diagnostics, higher-replication
bootstrap inference, robustness analyses, and reproducibility controls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import locale
import math
import mimetypes
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.point_in_time import pit_multimarket_forecasting_v8 as v8  # noqa: E402


EXPERIMENT_ID = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V8_1"
STUDY_LABEL = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V8_1_HARDENED"
EXPERIMENT_VERSION = "v8.1"
MANIFEST_VERSION = "manifest-portability-v8.1.0"
RELEASE_STATUS = "FINAL_HARDENED_COMPLETE_FIVE_MARKET"
BASELINE_COMMIT = "a3d737d2688e15fbcf67b686fab52884dd1c0e75"
FINAL_TAG = "pit-five-market-v8.1-final"
PRIMARY_BOOTSTRAP_SAMPLES = 10_000
SECONDARY_BOOTSTRAP_SAMPLES = 5_000
BOOTSTRAP_SEED = 271_828
PRIMARY_BLOCK_LENGTH = 20
ROBUSTNESS_BLOCK_LENGTHS = [10, 40]
BLOCK_LENGTHS = [PRIMARY_BLOCK_LENGTH, *ROBUSTNESS_BLOCK_LENGTHS]

PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
REPORTS = PIT_ROOT / "reports"
QA = PIT_ROOT / "qa"
MANIFESTS = PIT_ROOT / "manifests"
CONFIG = PIT_ROOT / "config"
SOURCE_V8 = PIT_ROOT / "sources" / "fbm_klci_v8"
SOURCE_V8_1 = PIT_ROOT / "sources" / "fbm_klci_v8_1"
QUARANTINE = QA / "http_challenge_quarantine_v8_1"
CHECKPOINT_ROOT = QA / "pit_v8_model_checkpoints"

CONFIG_PATH = CONFIG / "pit_multimarket_forecasting_v8_1.yaml"
ENVIRONMENT_PATH = CONFIG / "environment_v8_1.json"
REQUIREMENTS_PATH = REPO_ROOT / "requirements-v8.1-lock.txt"
REPRODUCE_PATH = REPO_ROOT / "REPRODUCE_PIT_V8_1.md"
STATUS_JSON_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8_1.json"
STATUS_MD_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8_1.md"
RECONCILIATION_PATH = REPORT_ROOT / "V8_V8_1_RECONCILIATION.md"
HARDENING_REPORT_PATH = REPORT_ROOT / "V8_1_HARDENING_REPORT.md"

SOURCE_REGISTRY_CSV = SOURCE_V8_1 / "fbm_klci_source_registry_v8_1.csv"
SOURCE_REGISTRY_JSON = SOURCE_V8_1 / "fbm_klci_source_registry_v8_1.json"
HTTP_AUDIT_PATH = QA / "http_retrieval_audit_v8_1.csv"
CHECKPOINT_CSV_PATH = QA / "pit_v8_1_checkpoint_inventory.csv"
CHECKPOINT_JSON_PATH = QA / "pit_v8_1_checkpoint_inventory.json"
ARTIFACT_VALIDATION_PATH = QA / "pit_v8_1_artifact_validation.json"
TEST_SUMMARY_PATH = QA / "pit_v8_1_test_summary.json"

RECOVERY_CSV_PATH = REPORTS / "fbm_klci_5014_recovery_audit_v8_1.csv"
RECOVERY_MD_PATH = REPORTS / "fbm_klci_5014_recovery_audit_v8_1.md"
SENSITIVITY_CSV_PATH = REPORTS / "fbm_klci_5014_sensitivity_v8_1.csv"
SENSITIVITY_MD_PATH = REPORTS / "fbm_klci_5014_sensitivity_v8_1.md"
COMMON_SUPPORT_CSV_PATH = REPORTS / "pit_multimarket_common_support_inference_v8_1.csv"
COMMON_SUPPORT_PARQUET_PATH = (
    REPORTS / "pit_multimarket_common_support_inference_v8_1.parquet"
)
SUPPORT_DIAGNOSTIC_CSV_PATH = (
    REPORTS / "pit_multimarket_support_composition_diagnostic_v8_1.csv"
)
SUPPORT_DIAGNOSTIC_MD_PATH = (
    REPORTS / "pit_multimarket_support_composition_diagnostic_v8_1.md"
)
PRIMARY_BOOTSTRAP_CSV_PATH = (
    REPORTS / "pit_multimarket_primary_bootstrap_10000_v8_1.csv"
)
PRIMARY_BOOTSTRAP_PARQUET_PATH = (
    REPORTS / "pit_multimarket_primary_bootstrap_10000_v8_1.parquet"
)
SECONDARY_BOOTSTRAP_CSV_PATH = (
    REPORTS / "pit_multimarket_secondary_bootstrap_v8_1.csv"
)
FAMILY_REGISTRY_PATH = REPORTS / "multiple_testing_family_registry_v8_1.csv"
AGGREGATION_PATH = REPORTS / "pit_multimarket_aggregation_robustness_v8_1.csv"
LOMO_PATH = REPORTS / "pit_multimarket_leave_one_market_out_v8_1.csv"
LOIO_PATH = REPORTS / "pit_multimarket_leave_one_interval_out_v8_1.csv"

RELEASE_MANIFEST_JSON = (
    MANIFESTS / "pit_multimarket_forecasting_v8_1_release_manifest.json"
)
RELEASE_MANIFEST_CSV = (
    MANIFESTS / "pit_multimarket_forecasting_v8_1_release_manifest.csv"
)
WORKSPACE_INVENTORY_CSV = MANIFESTS / "pit_v8_1_workspace_inventory.csv"
WORKSPACE_INVENTORY_JSON = MANIFESTS / "pit_v8_1_workspace_inventory.json"
RELEASE_ENVELOPE_PATH = MANIFESTS / "pit_v8_1_release_envelope.json"

V8_CORE_HASHES = {
    "v7_manifest": (
        v8.V7_MANIFEST_PATH,
        "9b30a42ec2c48d9395e6aba3a9b9af7c539346ebf39ad1050841a56d8eab11d1",
    ),
    "v8_manifest": (
        v8.MANIFEST_PATH,
        "14d5cd7ffc799d7faf1626d4dbddd2cdbc95382a5fdbc935c5fbcae5b24a8e70",
    ),
    "v8_seed_predictions": (
        v8.SEED_PATH,
        "47b551ff4359aae68981507f12602ed6c5e903c16196b42308dddd0815c72dbb",
    ),
    "v8_collapsed_predictions": (
        v8.COLLAPSED_PATH,
        "c37aa8deb7339ad6c634d47c2c58a31df3cfd50f2def993e66dc72a1d41bb15d",
    ),
    "v8_membership_intervals": (
        v8.MEMBERSHIP_PATH,
        "e77675b53d9d8ec7b09295749a87815e805714d620760f21229576ac6d344369",
    ),
    "v8_membership_daily": (
        v8.MEMBERSHIP_DAILY_PATH,
        "cd4c50fe4dbfe1e49e013a65c688c38186663cc154e4c7e4f52b592792fa72ca",
    ),
    "v8_feature_panel": (
        v8.FEATURE_PATH,
        "fc5b4e0e3c20442cfd291ee2ce2699fc580b3dc941d07a35db6ce415900afb3c",
    ),
    "v8_configuration": (
        v8.CONFIG_PATH,
        "9c0e91f0887b399b22acc40f3fd3ac23e7939f2d47068211e8cef9a57f38480e",
    ),
}

PREDICTION_KEY = [
    "interval_id",
    "universe_semantics",
    "fold_id",
    "security_id",
    "prediction_date",
    "model_id",
]
MATCH_KEY = [
    "market",
    "index_id",
    "interval_id",
    "fold_id",
    "model_id",
    "security_id",
    "prediction_date",
    "target_date",
]
METRICS = [
    "brier_score",
    "log_loss",
    "balanced_accuracy",
    "calibration_intercept",
    "calibration_slope",
    "prediction_coverage",
]
DELTA_COLUMNS = {
    metric: f"delta_{metric}_current_minus_pit" for metric in METRICS
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
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


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False, engine="pyarrow")
    else:
        frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def assert_v8_immutable(*, deep_predictions: bool = True) -> dict[str, Any]:
    failures: list[str] = []
    manifest = json.loads(v8.MANIFEST_PATH.read_text(encoding="utf-8"))
    validation = v8.validate_artifacts(manifest)
    if validation["status"] != "PASS" or len(manifest["records"]) != 76:
        failures.extend(validation["checksum_failures"])
        if len(manifest["records"]) != 76:
            failures.append(f"V8 manifest record count {len(manifest['records'])} != 76")
    v7_integrity = v8.validate_v7_immutability()
    if v7_integrity["status"] != "PASS":
        failures.extend(v7_integrity["failures"])
    observed_hashes: dict[str, str] = {}
    for label, (path, expected) in V8_CORE_HASHES.items():
        observed = sha256_file(path)
        observed_hashes[label] = observed
        if observed != expected:
            failures.append(f"{label} checksum drift: {observed} != {expected}")
    prediction_checks: dict[str, Any] = {"deep_checked": deep_predictions}
    if deep_predictions:
        seed = pd.read_parquet(v8.SEED_PATH)
        collapsed = pd.read_parquet(v8.COLLAPSED_PATH)
        if seed.duplicated(PREDICTION_KEY + ["seed"]).any():
            failures.append("duplicate V8 seed-level prediction key")
        if collapsed.duplicated(PREDICTION_KEY).any():
            failures.append("duplicate V8 collapsed prediction key")
        if not collapsed["seed_count"].eq(5).all():
            failures.append("one or more collapsed V8 predictions do not contain five seeds")
        if len(seed) != 5 * len(collapsed):
            failures.append("V8 seed/collapsed row-count ratio is not five")
        prediction_checks.update(
            {
                "seed_rows": len(seed),
                "collapsed_rows": len(collapsed),
                "seed_values": sorted(set(seed["seed"].astype(int))),
                "collapsed_seed_counts": sorted(
                    set(collapsed["seed_count"].astype(int))
                ),
                "unique_seed_keys": not seed.duplicated(
                    PREDICTION_KEY + ["seed"]
                ).any(),
                "unique_collapsed_keys": not collapsed.duplicated(
                    PREDICTION_KEY
                ).any(),
            }
        )
    result = {
        "status": "PASS" if not failures else "FAIL",
        "v8_manifest_records": len(manifest["records"]),
        "v8_manifest_validation": validation["status"],
        "v7_integrity": v7_integrity,
        "core_hashes": observed_hashes,
        "prediction_checks": prediction_checks,
        "failures": failures,
    }
    if failures:
        raise RuntimeError("V8 integrity failure: " + " | ".join(failures))
    return result


def _metric_record(frame: pd.DataFrame) -> dict[str, Any]:
    return v8.v7.analytics.metric_record(frame)


def _load_predictions() -> pd.DataFrame:
    predictions = pd.read_parquet(v8.COLLAPSED_PATH)
    predictions["prediction_date"] = pd.to_datetime(
        predictions["prediction_date"]
    ).dt.date
    predictions["target_date"] = pd.to_datetime(predictions["target_date"]).dt.date
    return predictions


def _performance_hierarchy(
    predictions: pd.DataFrame, members: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    composition = v8.v7.build_composition(members)
    analytics = v8.v7.analytics
    by_date = analytics.build_performance_by_date(predictions, composition)
    by_interval, by_market, macro = analytics.build_performance_hierarchy(by_date)
    current_minus_pit, date_diff = analytics.build_current_minus_pit(
        by_date, by_interval, by_market, macro
    )
    return {
        "by_date": by_date,
        "by_interval": by_interval,
        "by_market": by_market,
        "macro": macro,
        "current_minus_pit": current_minus_pit,
        "date_diff": date_diff,
    }


def match_common_support_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return a deterministic one-to-one arm match with equality/leakage gates."""
    pit = predictions[predictions["universe_semantics"].eq(v8.PIT)].copy()
    current = predictions[predictions["universe_semantics"].eq(v8.CURRENT)].copy()
    if pit.duplicated(MATCH_KEY).any() or current.duplicated(MATCH_KEY).any():
        raise RuntimeError("duplicate arm-native key before common-support matching")
    pit = pit[
        MATCH_KEY
        + [
            "actual_class",
            "prediction_probability",
            "predicted_class",
            "label_maturity_cutoff",
        ]
    ].rename(
        columns={
            "actual_class": "actual_class_pit",
            "prediction_probability": "prediction_probability_pit",
            "predicted_class": "predicted_class_pit",
            "label_maturity_cutoff": "label_maturity_cutoff_pit",
        }
    )
    current = current[
        MATCH_KEY
        + [
            "actual_class",
            "prediction_probability",
            "predicted_class",
            "label_maturity_cutoff",
        ]
    ].rename(
        columns={
            "actual_class": "actual_class_current",
            "prediction_probability": "prediction_probability_current",
            "predicted_class": "predicted_class_current",
            "label_maturity_cutoff": "label_maturity_cutoff_current",
        }
    )
    matched = pit.merge(current, on=MATCH_KEY, how="inner", validate="one_to_one")
    if not matched["actual_class_pit"].eq(matched["actual_class_current"]).all():
        raise RuntimeError("target mismatch across matched PIT/current predictions")
    for arm in ["pit", "current"]:
        if not (
            pd.to_datetime(matched[f"label_maturity_cutoff_{arm}"])
            < pd.to_datetime(matched["prediction_date"])
        ).all():
            raise RuntimeError(f"{arm} common-support label maturity leakage")
    matched["actual_class"] = matched["actual_class_pit"].astype(int)
    matched["common_support_key"] = [
        hashlib.sha256(
            "|".join(str(row[column]) for column in MATCH_KEY).encode("utf-8")
        ).hexdigest()[:24]
        for row in matched.to_dict("records")
    ]
    if matched["common_support_key"].duplicated().any():
        raise RuntimeError("duplicate common-support key")
    return matched.sort_values(MATCH_KEY, kind="stable").reset_index(drop=True)


def build_common_support(
    predictions: pd.DataFrame,
    full_hierarchy: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched = match_common_support_predictions(predictions)

    date_keys = [
        "market",
        "index_id",
        "interval_id",
        "fold_id",
        "model_id",
        "prediction_date",
    ]
    date_rows: list[dict[str, Any]] = []
    for values, group in matched.groupby(date_keys, sort=True):
        row = dict(zip(date_keys, values))
        pit_frame = pd.DataFrame(
            {
                "actual_class": group["actual_class"],
                "prediction_probability": group["prediction_probability_pit"],
            }
        )
        current_frame = pd.DataFrame(
            {
                "actual_class": group["actual_class"],
                "prediction_probability": group[
                    "prediction_probability_current"
                ],
            }
        )
        pit_metrics = _metric_record(pit_frame)
        current_metrics = _metric_record(current_frame)
        row["aggregation_level"] = "date"
        row["common_support_count"] = len(group)
        for metric in METRICS:
            if metric == "prediction_coverage":
                pit_value = current_value = 1.0
            else:
                pit_value = pit_metrics[metric]
                current_value = current_metrics[metric]
            row[f"pit_{metric}"] = pit_value
            row[f"current_{metric}"] = current_value
            row[DELTA_COLUMNS[metric]] = (
                float(current_value - pit_value)
                if pd.notna(current_value) and pd.notna(pit_value)
                else np.nan
            )
        date_rows.append(row)
    date_metrics = pd.DataFrame(date_rows)

    hierarchy_rows = list(date_metrics.to_dict("records"))
    lower = date_metrics.copy()
    for level, group_keys, unit in [
        (
            "interval",
            ["market", "index_id", "interval_id", "model_id"],
            "prediction_date",
        ),
        ("market", ["market", "index_id", "model_id"], "interval_id"),
        ("selected_five_market_macro", ["model_id"], "index_id"),
    ]:
        rows: list[dict[str, Any]] = []
        for values, group in lower.groupby(group_keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            row = dict(zip(group_keys, values))
            row["aggregation_level"] = level
            row["aggregation_units"] = int(group[unit].nunique())
            row["common_support_count"] = int(group["common_support_count"].sum())
            for metric in METRICS:
                for prefix in ["pit", "current"]:
                    row[f"{prefix}_{metric}"] = float(
                        pd.to_numeric(
                            group[f"{prefix}_{metric}"], errors="coerce"
                        ).mean()
                    )
                row[DELTA_COLUMNS[metric]] = (
                    row[f"current_{metric}"] - row[f"pit_{metric}"]
                )
            rows.append(row)
        lower = pd.DataFrame(rows)
        hierarchy_rows.extend(rows)
    common = pd.DataFrame(hierarchy_rows)
    intersection = full_hierarchy["macro"][
        full_hierarchy["macro"]["universe_semantics"].eq(v8.INTERSECTION)
    ].copy()
    intersection["aggregation_level"] = "separately_trained_intersection_control"
    intersection["support_specification"] = (
        "SEPARATELY_TRAINED_INTERSECTION_CONTROL"
    )
    intersection = intersection.rename(
        columns={
            metric: f"intersection_control_{metric}" for metric in METRICS
        }
    )
    common = pd.concat([common, intersection], ignore_index=True, sort=False)
    write_frame(COMMON_SUPPORT_CSV_PATH, common)
    write_frame(COMMON_SUPPORT_PARQUET_PATH, common)

    full_date = full_hierarchy["date_diff"].copy()
    full_levels = full_hierarchy["current_minus_pit"].copy()
    full_date["aggregation_level"] = "date"
    full = pd.concat([full_date, full_levels], ignore_index=True, sort=False)
    support_keys = {
        "date": [
            "aggregation_level",
            "market",
            "index_id",
            "interval_id",
            "model_id",
            "prediction_date",
        ],
        "interval": [
            "aggregation_level",
            "market",
            "index_id",
            "interval_id",
            "model_id",
        ],
        "market": [
            "aggregation_level",
            "market",
            "index_id",
            "model_id",
        ],
        "selected_market_macro": ["aggregation_level", "model_id"],
    }
    common_for_join = common.copy()
    common_for_join["aggregation_level"] = common_for_join[
        "aggregation_level"
    ].replace({"selected_five_market_macro": "selected_market_macro"})
    diagnostics: list[pd.DataFrame] = []
    for level, keys in support_keys.items():
        full_part = full[full["aggregation_level"].eq(level)].copy()
        common_part = common_for_join[
            common_for_join["aggregation_level"].eq(level)
        ].copy()
        if full_part.empty or common_part.empty:
            continue
        keep = keys + [DELTA_COLUMNS[metric] for metric in METRICS]
        merged = full_part[keep].merge(
            common_part[keep + ["common_support_count"]],
            on=keys,
            how="inner",
            suffixes=("_full_support", "_common_support"),
            validate="one_to_one",
        )
        for metric in METRICS:
            merged[
                f"descriptive_support_composition_component_{metric}"
            ] = (
                merged[f"{DELTA_COLUMNS[metric]}_full_support"]
                - merged[f"{DELTA_COLUMNS[metric]}_common_support"]
            )
        merged["interpretation"] = (
            "arithmetic support/composition diagnostic; not a causal decomposition"
        )
        diagnostics.append(merged)
    diagnostic = pd.concat(diagnostics, ignore_index=True, sort=False)
    write_frame(SUPPORT_DIAGNOSTIC_CSV_PATH, diagnostic)
    return matched, common, diagnostic


def build_support_markdown(
    common: pd.DataFrame, diagnostic: pd.DataFrame
) -> None:
    common_macro = common[
        common["aggregation_level"].eq("selected_five_market_macro")
    ]
    diagnostic_macro = diagnostic[
        diagnostic["aggregation_level"].eq("selected_market_macro")
    ]
    lines = [
        "# V8.1 Prediction-Support and Composition Diagnostic",
        "",
        "The locked V8 full-arm comparison remains primary. This diagnostic evaluates "
        "the existing arm-native models on exact common prediction keys; it does not retrain.",
        "",
        "The separately trained `PIT_CURRENT_INTERSECTION_CONTROL` arm remains a third, "
        "distinct diagnostic and is not interchangeable with exact common support.",
        "",
        "| Model | Full-support Brier delta | Common-support Brier delta | Arithmetic difference | Common rows |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in common_macro.to_dict("records"):
        selected = diagnostic_macro[
            diagnostic_macro["model_id"].eq(row["model_id"])
        ].iloc[0]
        lines.append(
            f"| {row['model_id']} | "
            f"{selected['delta_brier_score_current_minus_pit_full_support']:.8f} | "
            f"{selected['delta_brier_score_current_minus_pit_common_support']:.8f} | "
            f"{selected['descriptive_support_composition_component_brier_score']:.8f} | "
            f"{int(row['common_support_count']):,} |"
        )
    lines.extend(
        [
            "",
            "The arithmetic difference is descriptive. It combines support and "
            "composition effects and must not be described as a causal decomposition.",
            "",
            "## Separately trained intersection-control arm",
            "",
            "| Model | Intersection-control Brier | Intersection-control log loss | Intersection-control balanced accuracy |",
            "|---|---:|---:|---:|",
        ]
    )
    intersection = common[
        common["aggregation_level"].eq(
            "separately_trained_intersection_control"
        )
    ]
    for row in intersection.to_dict("records"):
        lines.append(
            f"| {row['model_id']} | "
            f"{row['intersection_control_brier_score']:.8f} | "
            f"{row['intersection_control_log_loss']:.8f} | "
            f"{row['intersection_control_balanced_accuracy']:.8f} |"
        )
    SUPPORT_DIAGNOSTIC_MD_PATH.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def build_5014_sensitivity(
    predictions: pd.DataFrame, members: pd.DataFrame
) -> pd.DataFrame:
    security = v8.v7.v6.security_id("FBM_KLCI", "5014")
    fbm_intervals = pd.read_parquet(v8.FBM_INTERVAL_PATH)
    affected_intervals = sorted(
        set(
            fbm_intervals.loc[
                fbm_intervals["security_id"].astype(str).eq(security),
                "interval_id",
            ].astype(str)
        )
    )
    specifications = {
        "A_LOCKED_PRIMARY": (predictions, members),
        "B_EXCLUDE_5014_ALL_ARMS": (
            predictions[~predictions["security_id"].astype(str).eq(security)],
            members[~members["security_id"].astype(str).eq(security)],
        ),
        "C_EXCLUDE_5014_ACTIVE_INTERVALS": (
            predictions[
                ~predictions["interval_id"].astype(str).isin(affected_intervals)
            ],
            members[~members["interval_id"].astype(str).isin(affected_intervals)],
        ),
    }
    output_rows: list[dict[str, Any]] = []
    for specification, (pred, member) in specifications.items():
        hierarchy = _performance_hierarchy(pred.copy(), member.copy())
        current = hierarchy["current_minus_pit"]
        for level_name, frame in [
            ("market", hierarchy["by_market"]),
            ("selected_five_market_macro", hierarchy["macro"]),
        ]:
            for record in frame.to_dict("records"):
                row = {
                    "specification": specification,
                    "aggregation_level": level_name,
                    **record,
                }
                output_rows.append(row)
        for record in current.to_dict("records"):
            if record["aggregation_level"] not in {
                "market",
                "selected_market_macro",
            }:
                continue
            row = {
                "specification": specification,
                "aggregation_level": record["aggregation_level"],
                "universe_semantics": "CURRENT_MINUS_PIT",
                **record,
            }
            output_rows.append(row)
    result = pd.DataFrame(output_rows)
    macro_arm = result[
        result["aggregation_level"].eq("selected_five_market_macro")
        & result["universe_semantics"].isin(
            [v8.PIT, v8.CURRENT, v8.INTERSECTION]
        )
    ].copy()
    macro_arm["brier_rank"] = macro_arm.groupby(
        ["specification", "universe_semantics"]
    )["brier_score"].rank(method="min")
    result = result.merge(
        macro_arm[
            [
                "specification",
                "aggregation_level",
                "universe_semantics",
                "model_id",
                "brier_rank",
            ]
        ],
        on=[
            "specification",
            "aggregation_level",
            "universe_semantics",
            "model_id",
        ],
        how="left",
    )
    result["affected_interval_ids"] = ";".join(affected_intervals)
    result["5014_recovered"] = False
    write_frame(SENSITIVITY_CSV_PATH, result)
    lines = [
        "# FBM KLCI 5014 Sensitivity Analysis V8.1",
        "",
        "Specification A is the unchanged V8 primary result.",
        "",
        "Specification B removes security 5014 from every prediction and membership "
        "arm. Because no valid 5014 predictions exist, predictive-loss rows are "
        "unchanged; the PIT coverage denominator changes transparently.",
        "",
        "Specification C removes every Malaysian interval in which 5014 was an active "
        "PIT constituent from all arms and recomputes all five-market hierarchies.",
        "",
        f"Affected intervals: `{';'.join(affected_intervals)}`.",
        "",
        "| Specification | Model | Brier current-minus-PIT | Log-loss current-minus-PIT | Coverage current-minus-PIT |",
        "|---|---|---:|---:|---:|",
    ]
    macro_delta = result[
        result["aggregation_level"].eq("selected_market_macro")
        & result["universe_semantics"].eq("CURRENT_MINUS_PIT")
    ]
    for row in macro_delta.to_dict("records"):
        lines.append(
            f"| {row['specification']} | {row['model_id']} | "
            f"{row['delta_brier_score_current_minus_pit']:.8f} | "
            f"{row['delta_log_loss_current_minus_pit']:.8f} | "
            f"{row['delta_prediction_coverage_current_minus_pit']:.8f} |"
        )
    SENSITIVITY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _hierarchical_bootstrap_replicates(
    frame: pd.DataFrame,
    value_column: str,
    *,
    block_length: int,
    samples: int,
    seed_parts: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, float], str | None]:
    working = frame.dropna(subset=[value_column]).copy()
    if working.empty:
        return {}, {}, "no finite paired date differences"
    interval_replicates: dict[str, np.ndarray] = {}
    interval_points: dict[str, float] = {}
    interval_market: dict[str, str] = {}
    for ordinal, (interval_id, group) in enumerate(
        working.groupby("interval_id", sort=True)
    ):
        vector = (
            group.sort_values("prediction_date")[value_column]
            .to_numpy(dtype=float)
        )
        if block_length == 40 and len(vector) < 80:
            continue
        if len(vector) < block_length:
            continue
        seed_sequence = np.random.SeedSequence(
            [BOOTSTRAP_SEED, *seed_parts, ordinal, block_length]
        )
        derived_seed = int(seed_sequence.generate_state(1)[0])
        block = min(int(block_length), len(vector))
        blocks_needed = int(math.ceil(len(vector) / block))
        remainder = len(vector) - block * (blocks_needed - 1)
        doubled = np.concatenate([vector, vector])
        cumulative = np.concatenate([[0.0], np.cumsum(doubled)])
        starts = np.random.default_rng(derived_seed).integers(
            0, len(vector), size=(samples, blocks_needed)
        )
        complete = cumulative[starts[:, :-1] + block] - cumulative[starts[:, :-1]]
        final = (
            cumulative[starts[:, -1] + remainder]
            - cumulative[starts[:, -1]]
        )
        interval_replicates[str(interval_id)] = (
            complete.sum(axis=1) + final
        ) / len(vector)
        interval_points[str(interval_id)] = float(vector.mean())
        interval_market[str(interval_id)] = str(group.iloc[0]["index_id"])
    if not interval_replicates:
        return {}, {}, "insufficient independent blocks"
    market_replicates: dict[str, np.ndarray] = {}
    market_points: dict[str, float] = {}
    for market in sorted(set(interval_market.values())):
        ids = sorted(
            interval_id
            for interval_id, value in interval_market.items()
            if value == market
        )
        market_replicates[market] = np.mean(
            np.vstack([interval_replicates[item] for item in ids]), axis=0
        )
        market_points[market] = float(
            np.mean([interval_points[item] for item in ids])
        )
    macro_replicates = np.mean(
        np.vstack([market_replicates[item] for item in sorted(market_replicates)]),
        axis=0,
    )
    macro_point = float(np.mean(list(market_points.values())))
    replicates = {
        **{f"market:{key}": value for key, value in market_replicates.items()},
        "selected_five_market_macro": macro_replicates,
    }
    points = {
        **{f"market:{key}": value for key, value in market_points.items()},
        "selected_five_market_macro": macro_point,
    }
    return replicates, points, None


def _finite_sample_summary(
    point: float, replicates: np.ndarray
) -> dict[str, float]:
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    centered_null = replicates - float(point)
    extreme = int(np.sum(np.abs(centered_null) >= abs(float(point))))
    p_value = (extreme + 1.0) / (len(replicates) + 1.0)
    return {
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "raw_p_value": float(p_value),
        "finite_sample_extreme_count": extreme,
    }


def _bootstrap_family(
    date_diff: pd.DataFrame,
    *,
    support: str,
    metrics: Sequence[str],
    samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = sorted(set(date_diff["model_id"].astype(str)))
    for model_ordinal, model_id in enumerate(models):
        selected_model = date_diff[
            date_diff["model_id"].astype(str).eq(model_id)
        ]
        for metric_ordinal, metric in enumerate(metrics):
            column = DELTA_COLUMNS[metric]
            for block in BLOCK_LENGTHS:
                replicates, points, reason = _hierarchical_bootstrap_replicates(
                    selected_model,
                    column,
                    block_length=block,
                    samples=samples,
                    seed_parts=[
                        1 if support == "COMMON_SUPPORT" else 0,
                        model_ordinal,
                        metric_ordinal,
                    ],
                )
                scope_names = [
                    *(f"market:{market}" for market in v8.MARKETS),
                    "selected_five_market_macro",
                ]
                for scope in scope_names:
                    base = {
                        "experiment_id": EXPERIMENT_ID,
                        "support_specification": support,
                        "inference_level": (
                            "market" if scope.startswith("market:") else scope
                        ),
                        "index_id": (
                            scope.split(":", 1)[1]
                            if scope.startswith("market:")
                            else None
                        ),
                        "model_id": model_id,
                        "metric": metric,
                        "block_length": block,
                        "bootstrap_samples": samples,
                        "bootstrap_seed": BOOTSTRAP_SEED,
                        "paired_sampling": True,
                        "blocks_confined_to_interval": True,
                        "block_truncation": "truncate final concatenated block to interval length",
                        "equal_market_aggregation": True,
                        "valid": scope in replicates,
                    }
                    if scope not in replicates:
                        rows.append(
                            {
                                **base,
                                "point_estimate": np.nan,
                                "ci_lower": np.nan,
                                "ci_upper": np.nan,
                                "raw_p_value": np.nan,
                                "finite_sample_extreme_count": 0,
                                "adjusted_p_value": np.nan,
                                "reason": reason or "scope unavailable",
                            }
                        )
                        continue
                    summary = _finite_sample_summary(
                        points[scope], replicates[scope]
                    )
                    rows.append(
                        {
                            **base,
                            "point_estimate": points[scope],
                            **summary,
                            "adjusted_p_value": np.nan,
                            "reason": None,
                        }
                    )
    result = pd.DataFrame(rows)
    return result


def _family_id(row: Mapping[str, Any]) -> str:
    support = str(row["support_specification"]).lower()
    level = str(row["inference_level"]).lower()
    metric = str(row["metric"]).lower()
    block = int(row["block_length"])
    if (
        support == "full_support"
        and level == "selected_five_market_macro"
        and metric == "brier_score"
        and block == PRIMARY_BLOCK_LENGTH
    ):
        return "CONFIRMATORY_MACRO_BRIER_BLOCK20"
    if metric == "prediction_coverage":
        return f"OPERATIONAL_{support}_{level}_BLOCK{block}"
    return f"EXPLORATORY_{support}_{level}_{metric}_BLOCK{block}"


def apply_bh_families(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["family_id"] = [
        _family_id(row) for row in result.to_dict("records")
    ]
    result["confirmatory_or_exploratory"] = np.where(
        result["family_id"].eq("CONFIRMATORY_MACRO_BRIER_BLOCK20"),
        "CONFIRMATORY",
        np.where(
            result["metric"].eq("prediction_coverage"),
            "OPERATIONAL",
            "EXPLORATORY",
        ),
    )
    valid = result["valid"].astype(bool) & result["raw_p_value"].notna()
    for _, indexes in result[valid].groupby("family_id").groups.items():
        result.loc[indexes, "adjusted_p_value"] = (
            v8.v7.analytics.benjamini_hochberg(
                result.loc[indexes, "raw_p_value"].to_numpy(dtype=float)
            )
        )
    result["result_classification"] = np.where(
        ~result["valid"].astype(bool),
        "INSUFFICIENT_INFORMATION",
        np.where(
            (result["adjusted_p_value"] <= 0.05)
            & (
                (result["ci_lower"] > 0)
                | (result["ci_upper"] < 0)
            ),
            "SUPPORTED_CURRENT_PIT_DIFFERENCE",
            "NO_SUPPORTED_DIFFERENCE",
        ),
    )
    return result


def build_bootstrap_outputs(
    full_date_diff: pd.DataFrame, common: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_date = common[common["aggregation_level"].eq("date")].copy()
    primary = pd.concat(
        [
            _bootstrap_family(
                full_date_diff,
                support="FULL_SUPPORT",
                metrics=["brier_score"],
                samples=PRIMARY_BOOTSTRAP_SAMPLES,
            ),
            _bootstrap_family(
                common_date,
                support="COMMON_SUPPORT",
                metrics=["brier_score"],
                samples=PRIMARY_BOOTSTRAP_SAMPLES,
            ),
        ],
        ignore_index=True,
    )
    primary = apply_bh_families(primary)
    write_frame(PRIMARY_BOOTSTRAP_CSV_PATH, primary)
    write_frame(PRIMARY_BOOTSTRAP_PARQUET_PATH, primary)

    secondary_metrics = [
        "log_loss",
        "balanced_accuracy",
        "prediction_coverage",
    ]
    secondary = pd.concat(
        [
            _bootstrap_family(
                full_date_diff,
                support="FULL_SUPPORT",
                metrics=secondary_metrics,
                samples=SECONDARY_BOOTSTRAP_SAMPLES,
            ),
            _bootstrap_family(
                common_date,
                support="COMMON_SUPPORT",
                metrics=secondary_metrics,
                samples=SECONDARY_BOOTSTRAP_SAMPLES,
            ),
        ],
        ignore_index=True,
    )
    secondary = apply_bh_families(secondary)
    write_frame(SECONDARY_BOOTSTRAP_CSV_PATH, secondary)

    registry_rows: list[dict[str, Any]] = []
    for family_id, group in pd.concat(
        [primary, secondary], ignore_index=True
    ).groupby("family_id", sort=True):
        first = group.iloc[0]
        registry_rows.append(
            {
                "family_id": family_id,
                "confirmatory_or_exploratory": first[
                    "confirmatory_or_exploratory"
                ],
                "support_specification": first["support_specification"],
                "inference_level": first["inference_level"],
                "metric": first["metric"],
                "block_length": int(first["block_length"]),
                "models": ";".join(sorted(set(group["model_id"].astype(str)))),
                "hypothesis_count": int(
                    group[group["valid"].astype(bool)]["raw_p_value"].notna().sum()
                ),
                "adjustment_method": "Benjamini-Hochberg",
                "inclusion_rule": (
                    "all valid current-minus-PIT hypotheses matching support, "
                    "level, metric, and block length"
                ),
                "exclusion_rule": (
                    "invalid block support, missing finite statistic, other "
                    "metrics, other levels, and other support specifications"
                ),
            }
        )
    registry = pd.DataFrame(registry_rows)
    write_frame(FAMILY_REGISTRY_PATH, registry)
    return primary, secondary, registry


def _equal_market_delta(frame: pd.DataFrame, column: str) -> float:
    interval = frame.groupby(
        ["index_id", "interval_id"], sort=True
    )[column].mean()
    market = interval.groupby(level="index_id").mean()
    return float(market.mean())


def build_aggregation_robustness(
    predictions: pd.DataFrame,
    members: pd.DataFrame,
    hierarchy: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_diff = hierarchy["date_diff"].copy()
    affected = set(
        pd.read_parquet(v8.FBM_INTERVAL_PATH)
        .loc[
            lambda frame: frame["ticker_on_date"].astype(str).eq("5014"),
            "interval_id",
        ]
        .astype(str)
    )
    date_counts = date_diff.groupby("interval_id")["prediction_date"].nunique()
    long_intervals = set(date_counts[date_counts >= 80].index.astype(str))
    rows: list[dict[str, Any]] = []
    for model_id, model in date_diff.groupby("model_id", sort=True):
        for metric in [
            "brier_score",
            "log_loss",
            "balanced_accuracy",
            "prediction_coverage",
        ]:
            column = DELTA_COLUMNS[metric]
            finite = model.dropna(subset=[column])
            specifications = {
                "PRIMARY_EQUAL_MARKET": _equal_market_delta(finite, column),
                "EQUAL_OOS_DATE": float(finite[column].mean()),
                "EQUAL_INTERVAL": float(
                    finite.groupby("interval_id")[column].mean().mean()
                ),
                "EXCLUDE_INTERVALS_LT_80_OOS_DATES": _equal_market_delta(
                    finite[
                        finite["interval_id"].astype(str).isin(long_intervals)
                    ],
                    column,
                ),
                "EXCLUDE_5014_ACTIVE_INTERVALS": _equal_market_delta(
                    finite[~finite["interval_id"].astype(str).isin(affected)],
                    column,
                ),
            }
            for specification, value in specifications.items():
                rows.append(
                    {
                        "aggregation_specification": specification,
                        "model_id": model_id,
                        "metric": metric,
                        "point_estimate_current_minus_pit": value,
                        "primary": specification == "PRIMARY_EQUAL_MARKET",
                    }
                )
    # Security-date weighting is arm-native rather than paired-date weighted.
    for model_id in sorted(set(predictions["model_id"].astype(str))):
        selected = predictions[predictions["model_id"].astype(str).eq(model_id)]
        arm_metrics = {}
        for arm in [v8.PIT, v8.CURRENT]:
            arm_metrics[arm] = _metric_record(
                selected[selected["universe_semantics"].eq(arm)]
            )
        for metric in ["brier_score", "log_loss", "balanced_accuracy"]:
            rows.append(
                {
                    "aggregation_specification": "SECURITY_DATE_WEIGHTING",
                    "model_id": model_id,
                    "metric": metric,
                    "point_estimate_current_minus_pit": (
                        arm_metrics[v8.CURRENT][metric]
                        - arm_metrics[v8.PIT][metric]
                    ),
                    "primary": False,
                }
            )
        by_date = hierarchy["by_date"]
        model_date = by_date[by_date["model_id"].astype(str).eq(model_id)]
        coverage = {}
        for arm in [v8.PIT, v8.CURRENT]:
            selected_arm = model_date[model_date["universe_semantics"].eq(arm)]
            coverage[arm] = float(
                selected_arm["prediction_count"].sum()
                / selected_arm["universe_security_count"].sum()
            )
        rows.append(
            {
                "aggregation_specification": "SECURITY_DATE_WEIGHTING",
                "model_id": model_id,
                "metric": "prediction_coverage",
                "point_estimate_current_minus_pit": (
                    coverage[v8.CURRENT] - coverage[v8.PIT]
                ),
                "primary": False,
            }
        )
    aggregation = pd.DataFrame(rows)
    write_frame(AGGREGATION_PATH, aggregation)

    lomo_rows: list[dict[str, Any]] = []
    for excluded in sorted(set(date_diff["index_id"].astype(str))):
        subset = date_diff[~date_diff["index_id"].astype(str).eq(excluded)]
        for model_id, model in subset.groupby("model_id", sort=True):
            for metric in [
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "prediction_coverage",
            ]:
                column = DELTA_COLUMNS[metric]
                lomo_rows.append(
                    {
                        "excluded_market": excluded,
                        "remaining_market_count": int(model["index_id"].nunique()),
                        "model_id": model_id,
                        "metric": metric,
                        "point_estimate_current_minus_pit": _equal_market_delta(
                            model.dropna(subset=[column]), column
                        ),
                    }
                )
    lomo = pd.DataFrame(lomo_rows)
    write_frame(LOMO_PATH, lomo)

    loio_rows: list[dict[str, Any]] = []
    for excluded in sorted(set(date_diff["interval_id"].astype(str))):
        subset = date_diff[~date_diff["interval_id"].astype(str).eq(excluded)]
        excluded_market = str(
            date_diff.loc[
                date_diff["interval_id"].astype(str).eq(excluded), "index_id"
            ].iloc[0]
        )
        for model_id, model in subset.groupby("model_id", sort=True):
            for metric in [
                "brier_score",
                "log_loss",
                "balanced_accuracy",
                "prediction_coverage",
            ]:
                column = DELTA_COLUMNS[metric]
                loio_rows.append(
                    {
                        "excluded_interval": excluded,
                        "excluded_market": excluded_market,
                        "remaining_interval_count": int(
                            model["interval_id"].nunique()
                        ),
                        "model_id": model_id,
                        "metric": metric,
                        "point_estimate_current_minus_pit": _equal_market_delta(
                            model.dropna(subset=[column]), column
                        ),
                    }
                )
    loio = pd.DataFrame(loio_rows)
    write_frame(LOIO_PATH, loio)
    return aggregation, lomo, loio


def detect_mime(path: Path) -> str:
    """Identify the evidence payload from bytes, never from its extension."""
    payload = path.read_bytes()
    head = payload[:4096].lstrip().lower()
    if payload.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<html" in head:
        return "text/html"
    try:
        payload.decode("utf-8")
        return "text/plain"
    except UnicodeDecodeError:
        return "application/octet-stream"


def is_http_challenge(path: Path) -> bool:
    if detect_mime(path) != "text/html":
        return False
    lowered = path.read_text(encoding="utf-8", errors="replace").lower()
    tokens = [
        "cloudflare",
        "just a moment",
        "challenge-platform",
        "cf-chl-",
        "proof-of-work",
        "captcha",
        "requires javascript to verify your browser",
        "crypto.subtle.digest",
        'fetch("/__verify"',
    ]
    return any(token in lowered for token in tokens)


def validate_source_document(path: Path, *, expected_text: str) -> dict[str, Any]:
    """Validate signature/parser/content for an evidence document."""
    detected = detect_mime(path)
    failures: list[str] = []
    details: dict[str, Any] = {
        "detected_mime": detected,
        "challenge_detected": is_http_challenge(path),
    }
    if detected == "application/pdf":
        if not path.read_bytes().startswith(b"%PDF-"):
            failures.append("invalid PDF signature")
        try:
            from pypdf import PdfReader

            reader = PdfReader(path)
            details["pdf_page_count"] = len(reader.pages)
            extracted = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            ).lower()
            details["expected_content_present"] = expected_text.lower() in extracted
        except Exception as exc:  # pragma: no cover - defensive release gate
            failures.append(f"PDF parse failed: {exc}")
    elif detected == "text/html":
        content = path.read_text(encoding="utf-8", errors="replace").lower()
        details["expected_content_present"] = expected_text.lower() in content
        if details["challenge_detected"]:
            failures.append("HTTP challenge response cannot be evidence")
    else:
        failures.append(f"unsupported source-document MIME {detected}")
    if not details.get("expected_content_present", False):
        failures.append(f"expected publication content missing: {expected_text}")
    details["status"] = "PASS" if not failures else "FAIL"
    details["failures"] = failures
    return details


def build_source_registry() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Split original-document integrity from factual-extraction integrity."""
    source_path = SOURCE_V8 / "fbm_klci_source_registry_v8.csv"
    prior = pd.read_csv(source_path, keep_default_na=False)
    extraction_paths = {
        "SRC_ANCHOR_2017_12_31": (
            SOURCE_V8 / "fbmklci_etf_annual_report_2017_factual_extraction.txt"
        ),
        "SRC_REVIEW_2018_06": (
            SOURCE_V8 / "fbmklci_2018_review_factual_extractions.txt"
        ),
        "SRC_REVIEW_2018_12": (
            SOURCE_V8 / "fbmklci_2018_review_factual_extractions.txt"
        ),
        "SRC_ANCHOR_2019_12_23": (
            SOURCE_V8
            / "bursa_fbmklci_constituents_2019-12-23_factual_extraction.txt"
        ),
        "SRC_CORP_CELCOMDIGI_MERGER": (
            SOURCE_V8 / "fbmklci_structural_events_factual_extractions.txt"
        ),
        "SRC_CORP_CELCOMDIGI_RENAME": (
            SOURCE_V8 / "fbmklci_structural_events_factual_extractions.txt"
        ),
        "SRC_CORP_SDG_RENAME": (
            SOURCE_V8 / "fbmklci_structural_events_factual_extractions.txt"
        ),
        "SRC_CORP_MAHB_DELIST": (
            SOURCE_V8 / "fbmklci_structural_events_factual_extractions.txt"
        ),
    }
    expected_fragments = {
        "SRC_FTSE_2016_06_NOTICE": "FTSE Bursa Malaysia Index Series",
        "SRC_REVIEW_2016_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2017_06": "FTSE Bursa Malaysia Index Series",
        "SRC_FTSE_2017_DEMERGER": "Sime Darby",
        "SRC_FTSE_2017_DEMERGER_UPDATE": "Sime Darby",
        "SRC_REVIEW_2017_12": "FTSE Bursa Malaysia Index Series",
        "SRC_REVIEW_2019_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2019_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2020_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2020_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2021_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2021_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2022_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2022_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2023_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2023_12": "FTSE Bursa Malaysia KLCI",
        "SRC_FTSE_GROUND_RULES_2024": "FTSE Bursa Malaysia",
        "SRC_REVIEW_2024_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2024_12": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2025_06": "FTSE Bursa Malaysia KLCI",
        "SRC_REVIEW_2025_12": "FTSE Bursa Malaysia KLCI",
    }
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for old in prior.to_dict("records"):
        source_id = str(old["source_id"])
        old_cache = REPO_ROOT / str(old["local_cache_path"])
        extraction = extraction_paths.get(source_id)
        source_document = None if extraction is not None else old_cache
        validation: dict[str, Any] | None = None
        if source_document is not None:
            validation = validate_source_document(
                source_document,
                expected_text=expected_fragments[source_id],
            )
            if validation["status"] != "PASS":
                raise RuntimeError(
                    f"source document validation failed for {source_id}: "
                    + " | ".join(validation["failures"])
                )
            validations.append(
                {
                    "source_id": source_id,
                    "path": relative(source_document),
                    **validation,
                }
            )
        evidentiary_status = str(old["source_quality_status"])
        notes = str(old["notes"])
        if source_id == "SRC_ANCHOR_2017_12_31":
            evidentiary_status = (
                "ACCEPTED_TIER2_FACTUAL_EXTRACTION_WITH_REMOTE_DOCUMENT_REFERENCE"
            )
            notes = (
                notes
                + " Authentic exchange-filed PDF could not be cached after "
                "Bursa, AmFunds, ETF archive, Internet Archive, and filing-mirror "
                "searches; the extraction hash is not an original-document hash."
            ).strip()
        if source_id == "SRC_REVIEW_2018_06":
            evidentiary_status = (
                "ACCEPTED_TIER3_EXACT_ENDPOINT_CROSS_VALIDATED"
            )
            notes = (
                notes
                + " No stronger accessible primary/archive was found. The Tier 3 "
                "attributed report is cross-validated by the 2019 anchor endpoint; "
                "it is not represented as a primary source."
            ).strip()
        retrieval_status = (
            "LOCAL_VALIDATED"
            if source_document is not None
            else "REMOTE_REFERENCE_ONLY_DOCUMENT_NOT_CACHED"
        )
        row = {
            "source_id": source_id,
            "document_title": old["document_title"],
            "document_type": old["document_type"],
            "publication_date": old["publication_date"] or None,
            "as_of_date": old["as_of_date"] or None,
            "effective_date": old["effective_date"] or None,
            "source_document_local_path": (
                relative(source_document) if source_document is not None else None
            ),
            "source_document_sha256": (
                sha256_file(source_document) if source_document is not None else None
            ),
            "source_document_mime_type": (
                detect_mime(source_document) if source_document is not None else None
            ),
            "source_document_retrieval_status": retrieval_status,
            "extraction_local_path": (
                relative(extraction) if extraction is not None else None
            ),
            "extraction_sha256": (
                sha256_file(extraction) if extraction is not None else None
            ),
            "extraction_method": (
                "MANUAL_FACTUAL_TRANSCRIPTION_WITH_EXPLICIT_PROVENANCE"
                if extraction is not None
                else None
            ),
            "extraction_verified_by": (
                "V8.1 deterministic content and checksum validation"
                if extraction is not None
                else None
            ),
            "source_url": old["source_url"] or None,
            "archive_url": old["archive_url"] or None,
            "retrieval_timestamp": (
                f"{old['retrieval_date']}T00:00:00+07:00"
                if old["retrieval_date"]
                else None
            ),
            "publisher": old["publisher"],
            "source_tier": int(old["source_tier"]),
            "evidence_role": old["evidence_role"],
            "evidentiary_status": evidentiary_status,
            "redistribution_status": old["redistribution_status"],
            "notes": notes,
        }
        if row["source_document_local_path"] and row["extraction_local_path"]:
            if row["source_document_sha256"] == row["extraction_sha256"]:
                raise RuntimeError(f"document/extraction checksum conflation: {source_id}")
        rows.append(row)
    registry = pd.DataFrame(rows)
    write_frame(SOURCE_REGISTRY_CSV, registry)
    write_json(
        SOURCE_REGISTRY_JSON,
        {
            "experiment_id": EXPERIMENT_ID,
            "source_count": len(registry),
            "registry_schema": "document-and-extraction-integrity-v8.1",
            "records": registry.to_dict("records"),
        },
    )
    return registry, validations


def build_http_audit() -> pd.DataFrame:
    """Record failed responses already isolated under the quarantine directory."""
    attempts = {
        "bursa_fbmklci_constituents_2019-12-23.cloudflare-blocked.html": {
            "requested_url": "https://www.bursamalaysia.com/trade/trading_resources/listing_directory/ftse_bursa_malaysia_indices",
            "final_url": "https://www.bursamalaysia.com/trade/trading_resources/listing_directory/ftse_bursa_malaysia_indices",
            "http_status": 403,
        },
        "fbmklci_etf_annual_report_2017.cloudflare-blocked.html": {
            "requested_url": "https://www.aminvest.com/docs/annual-reports/fbm-klci-etf-annual-report-2017.pdf",
            "final_url": "https://www.aminvest.com/docs/annual-reports/fbm-klci-etf-annual-report-2017.pdf",
            "http_status": 403,
        },
        "stooq_5014.my.csv": {
            "requested_url": "https://stooq.com/q/d/l/?s=5014.my&d1=20180618&d2=20200619&i=d",
            "final_url": "https://stooq.com/q/d/l/?s=5014.my&d1=20180618&d2=20200619&i=d",
            "http_status": 200,
        },
        "stooq_airport.my.csv": {
            "requested_url": "https://stooq.com/q/d/l/?s=airport.my&d1=20180618&d2=20200619&i=d",
            "final_url": "https://stooq.com/q/d/l/?s=airport.my&d1=20180618&d2=20200619&i=d",
            "http_status": 200,
        },
        "stooq_mahb.my.csv": {
            "requested_url": "https://stooq.com/q/d/l/?s=mahb.my&d1=20180618&d2=20200619&i=d",
            "final_url": "https://stooq.com/q/d/l/?s=mahb.my&d1=20180618&d2=20200619&i=d",
            "http_status": 200,
        },
        "yahoo_5014.KL.json": {
            "requested_url": "https://query2.finance.yahoo.com/v8/finance/chart/5014.KL?period1=1529280000&period2=1592697600&interval=1d&events=history",
            "final_url": "https://query2.finance.yahoo.com/v8/finance/chart/5014.KL",
            "http_status": 429,
        },
        "yahoo_AIRPORT.KL.json": {
            "requested_url": "https://query2.finance.yahoo.com/v8/finance/chart/AIRPORT.KL?period1=1529280000&period2=1592697600&interval=1d&events=history",
            "final_url": "https://query2.finance.yahoo.com/v8/finance/chart/AIRPORT.KL",
            "http_status": 429,
        },
        "yahoo_MAHB.KL.json": {
            "requested_url": "https://query2.finance.yahoo.com/v8/finance/chart/MAHB.KL?period1=1529280000&period2=1592697600&interval=1d&events=history",
            "final_url": "https://query2.finance.yahoo.com/v8/finance/chart/MAHB.KL",
            "http_status": 429,
        },
        "yahoo_MYL5014OO005.KL.json": {
            "requested_url": "https://query2.finance.yahoo.com/v8/finance/chart/MYL5014OO005.KL?period1=1529280000&period2=1592697600&interval=1d&events=history",
            "final_url": "https://query2.finance.yahoo.com/v8/finance/chart/MYL5014OO005.KL",
            "http_status": 429,
        },
    }
    rows: list[dict[str, Any]] = []
    for name, metadata in attempts.items():
        path = QUARANTINE / name
        if not path.exists():
            raise RuntimeError(f"expected quarantined response missing: {path}")
        detected = detect_mime(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"<title[^>]*>(.*?)</title>", text, re.I | re.S)
        challenge = is_http_challenge(path) or metadata["http_status"] in {403, 429}
        rows.append(
            {
                **metadata,
                "content_type": (
                    "text/html"
                    if detected == "text/html"
                    else "text/plain; charset=utf-8"
                ),
                "detected_mime": detected,
                "file_extension": path.suffix.lower(),
                "response_title": (
                    re.sub(r"\s+", " ", match.group(1)).strip()
                    if match
                    else text[:80].strip()
                ),
                "challenge_detected": challenge,
                "used_as_evidence": False,
                "quarantine_path": relative(path),
                "sha256": sha256_file(path),
            }
        )
    audit = pd.DataFrame(rows)
    if audit["used_as_evidence"].astype(bool).any():
        raise RuntimeError("quarantined response marked as evidence")
    write_frame(HTTP_AUDIT_PATH, audit)
    return audit


def build_5014_recovery_audit() -> pd.DataFrame:
    rows = [
        {
            "search_lane": "repository_raw_price_and_provider_cache_scan",
            "query_or_alias": "5014; Malaysia Airports; MAHB",
            "source_or_location": "repository excluding V8.1 quarantine",
            "result": "NO_VALID_RAW_OHLC_FOUND",
            "raw_ohlc_rows": 0,
            "used": False,
            "notes": (
                "The only MAHB-named repository JSON was a different Philippine "
                "security; no Malaysia 5014 native OHLC response was present."
            ),
        },
        {
            "search_lane": "Yahoo_chart_aliases",
            "query_or_alias": "5014.KL; AIRPORT.KL; MAHB.KL; MYL5014OO005.KL",
            "source_or_location": "query2.finance.yahoo.com",
            "result": "HTTP_429_RATE_LIMIT_RESPONSES_QUARANTINED",
            "raw_ohlc_rows": 0,
            "used": False,
            "notes": "No chart payload was returned; adjusted close was not used.",
        },
        {
            "search_lane": "Stooq_aliases",
            "query_or_alias": "5014.my; airport.my; mahb.my",
            "source_or_location": "stooq.com",
            "result": "HTTP_200_PROOF_OF_WORK_HTML_QUARANTINED",
            "raw_ohlc_rows": 0,
            "used": False,
            "notes": "The response was a challenge, not CSV; no bypass attempted.",
        },
        {
            "search_lane": "official_exchange",
            "query_or_alias": "Bursa 5014 historical price",
            "source_or_location": "Bursa Malaysia official publications",
            "result": "NO_DAILY_NATIVE_OHLC_RECOVERED",
            "raw_ohlc_rows": 0,
            "used": False,
            "notes": (
                "Official year-end securities lists contain closing-price "
                "snapshots, not the required daily raw OHLC history."
            ),
        },
    ]
    audit = pd.DataFrame(rows)
    audit["security_id"] = v8.v7.v6.security_id("FBM_KLCI", "5014")
    audit["ticker"] = "5014"
    audit["requested_active_member_days"] = 493
    audit["joined_active_member_days"] = 0
    audit["recovery_status"] = "UNRECOVERED_EXPLICIT_GAP"
    audit["eligibility_threshold_changed"] = False
    audit["release_eligibility_status"] = "PASS_WITH_DOCUMENTED_LIMITATIONS"
    write_frame(RECOVERY_CSV_PATH, audit)
    RECOVERY_MD_PATH.write_text(
        "\n".join(
            [
                "# Malaysia Airports 5014 Recovery Audit V8.1",
                "",
                "**Outcome:** no legitimate provider-native raw daily OHLC was recovered.",
                "",
                "The locked gap remains **0 / 493 active-member days joined**. "
                "Eligibility thresholds were not changed; no adjusted close, "
                "interpolation, inference, or forward fill was used.",
                "",
                "Repository caches, Yahoo chart aliases, Stooq aliases, Bursa code "
                "5014, provider mappings, and delisting/rename metadata were checked. "
                "Yahoo returned rate-limit text and Stooq returned proof-of-work HTML; "
                "all such payloads are quarantined and excluded from evidence.",
                "",
                "Release status for this limitation: `PASS_WITH_DOCUMENTED_LIMITATIONS`. "
                "Specifications A/B/C in the sensitivity artifact quantify the effect.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return audit


def _component_fingerprint(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[int, int, int]:
    values = pd.util.hash_pandas_object(
        frame.loc[:, list(columns)], index=False, categorize=False
    ).to_numpy(dtype=np.uint64)
    total = int(values.sum(dtype=np.uint64))
    xor = int(np.bitwise_xor.reduce(values)) if len(values) else 0
    return len(values), total, xor


def _merge_fingerprint(
    current: tuple[int, int, int] | None,
    incoming: tuple[int, int, int],
) -> tuple[int, int, int]:
    if current is None:
        return incoming
    mask = (1 << 64) - 1
    return (
        current[0] + incoming[0],
        (current[1] + incoming[1]) & mask,
        current[2] ^ incoming[2],
    )


def build_checkpoint_inventory(*, remove_after_validation: bool) -> pd.DataFrame:
    """Prove each checkpoint component is in V8 seed output, then optionally delete."""
    if not CHECKPOINT_ROOT.exists():
        existing = pd.read_csv(CHECKPOINT_CSV_PATH)
        if not existing["incorporated_into_final_output"].astype(bool).all():
            raise RuntimeError("checkpoint directory absent but prior inventory is not validated")
        return existing
    resolved = CHECKPOINT_ROOT.resolve()
    expected = (QA / "pit_v8_model_checkpoints").resolve()
    if resolved != expected or not str(resolved).startswith(str(QA.resolve())):
        raise RuntimeError(f"unsafe checkpoint target: {resolved}")
    parquet_paths = sorted(CHECKPOINT_ROOT.glob("*.parquet"))
    marker_paths = sorted(CHECKPOINT_ROOT.glob("*.complete.json"))
    if len(parquet_paths) != 165 or len(marker_paths) != 165:
        raise RuntimeError(
            f"checkpoint inventory expected 165+165, observed "
            f"{len(parquet_paths)}+{len(marker_paths)}"
        )
    component_keys = [
        "market",
        "index_id",
        "universe_semantics",
        "model_id",
        "fold_id",
        "seed",
    ]
    fingerprint_columns: list[str] | None = None
    checkpoint_fingerprints: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    marker_by_fold: dict[str, tuple[Path, dict[str, Any]]] = {}
    for marker in marker_paths:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        marker_by_fold[str(payload["fold_id"])] = (marker, payload)
    total_checkpoint_rows = 0
    for checkpoint in parquet_paths:
        frame = pd.read_parquet(checkpoint)
        if fingerprint_columns is None:
            fingerprint_columns = list(frame.columns)
        elif fingerprint_columns != list(frame.columns):
            raise RuntimeError(f"checkpoint schema drift: {checkpoint}")
        fold_values = set(frame["fold_id"].astype(str))
        if len(fold_values) != 1:
            raise RuntimeError(f"checkpoint contains multiple folds: {checkpoint}")
        fold_id = next(iter(fold_values))
        marker_info = marker_by_fold.get(fold_id)
        if marker_info is None:
            raise RuntimeError(f"completion marker missing for {fold_id}")
        marker_path, marker_payload = marker_info
        if int(marker_payload["prediction_rows"]) != len(frame):
            raise RuntimeError(f"marker row-count mismatch for {fold_id}")
        total_checkpoint_rows += len(frame)
        normalized = frame.copy()
        normalized["model_version"] = normalized["model_version"].astype(str).str.replace(
            "v7-", "v8-", n=1, regex=False
        )
        for values, group in normalized.groupby(component_keys, sort=True):
            values = values if isinstance(values, tuple) else (values,)
            component = dict(zip(component_keys, values))
            key = tuple(component[item] for item in component_keys)
            fingerprint = _component_fingerprint(group, fingerprint_columns)
            checkpoint_fingerprints[key] = _merge_fingerprint(
                checkpoint_fingerprints.get(key), fingerprint
            )
            checkpoint_rows.append(
                {
                    **component,
                    "artifact_type": "CHECKPOINT_PARQUET_COMPONENT",
                    "file_path": relative(checkpoint),
                    "row_count": len(group),
                    "file_size_bytes": checkpoint.stat().st_size,
                    "file_sha256": sha256_file(checkpoint),
                    "component_fingerprint": (
                        f"{fingerprint[0]}:{fingerprint[1]:016x}:"
                        f"{fingerprint[2]:016x}"
                    ),
                    "completion_status": "COMPLETE_MARKER_VALIDATED",
                    "incorporated_into_final_output": False,
                    "safe_to_delete": False,
                    "post_validation_file_status": "PRESENT_PENDING_VALIDATION",
                }
            )
        checkpoint_rows.append(
            {
                "market": str(frame.iloc[0]["market"]),
                "index_id": str(frame.iloc[0]["index_id"]),
                "universe_semantics": "MULTIPLE",
                "model_id": "MULTIPLE",
                "fold_id": fold_id,
                "seed": "MULTIPLE",
                "artifact_type": "COMPLETION_MARKER",
                "file_path": relative(marker_path),
                "row_count": int(marker_payload["prediction_rows"]),
                "file_size_bytes": marker_path.stat().st_size,
                "file_sha256": sha256_file(marker_path),
                "component_fingerprint": marker_payload["checkpoint_signature"],
                "completion_status": (
                    "COMPLETE"
                    if not marker_payload.get("failures")
                    else "FAILED"
                ),
                "incorporated_into_final_output": False,
                "safe_to_delete": False,
                "post_validation_file_status": "PRESENT_PENDING_VALIDATION",
            }
        )
    if fingerprint_columns is None:
        raise RuntimeError("no checkpoint schema")

    # Stream the immutable seed file once. Order-independent sum+xor fingerprints
    # establish component-wise multiset equality without a multi-gigabyte join.
    import pyarrow.parquet as pq

    final_fingerprints: dict[tuple[Any, ...], tuple[int, int, int]] = {}
    parquet_file = pq.ParquetFile(v8.SEED_PATH)
    for batch in parquet_file.iter_batches(
        batch_size=100_000, columns=fingerprint_columns
    ):
        frame = batch.to_pandas()
        for values, group in frame.groupby(component_keys, sort=False):
            values = values if isinstance(values, tuple) else (values,)
            key = tuple(values)
            final_fingerprints[key] = _merge_fingerprint(
                final_fingerprints.get(key),
                _component_fingerprint(group, fingerprint_columns),
            )
    failures: list[str] = []
    if set(checkpoint_fingerprints) != set(final_fingerprints):
        failures.append("checkpoint/final component-key sets differ")
    for key in sorted(set(checkpoint_fingerprints) | set(final_fingerprints)):
        if checkpoint_fingerprints.get(key) != final_fingerprints.get(key):
            failures.append(f"component fingerprint mismatch: {key}")
            if len(failures) >= 20:
                break
    if total_checkpoint_rows != sum(value[0] for value in final_fingerprints.values()):
        failures.append("checkpoint/final total row counts differ")
    if failures:
        raise RuntimeError("checkpoint incorporation validation failed: " + " | ".join(failures))

    inventory = pd.DataFrame(checkpoint_rows)
    inventory["incorporated_into_final_output"] = True
    inventory["safe_to_delete"] = True
    policy = (
        "REMOVE_AFTER_VALIDATED_INCORPORATION"
        if remove_after_validation
        else "RETAIN_PENDING_RELEASE_BUILD"
    )
    inventory["release_policy"] = policy
    inventory["post_validation_file_status"] = (
        "DELETED_AFTER_INCORPORATION_VALIDATION"
        if remove_after_validation
        else "PRESENT_VALIDATED"
    )
    write_frame(CHECKPOINT_CSV_PATH, inventory)
    aggregate_material = "\n".join(
        f"{'|'.join(map(str, key))}:{value[0]}:{value[1]:016x}:{value[2]:016x}"
        for key, value in sorted(checkpoint_fingerprints.items())
    ).encode("utf-8")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "source_experiment_id": v8.EXPERIMENT_ID,
        "checkpoint_files": len(parquet_paths),
        "completion_markers": len(marker_paths),
        "component_records": int(
            inventory["artifact_type"].eq("CHECKPOINT_PARQUET_COMPONENT").sum()
        ),
        "checkpoint_prediction_rows": total_checkpoint_rows,
        "final_seed_prediction_rows": sum(
            value[0] for value in final_fingerprints.values()
        ),
        "incorporation_validation": "PASS",
        "fingerprint_method": (
            "component-wise row-hash multiset count, uint64 sum, and uint64 xor; "
            "all prediction columns included after the documented deterministic "
            "checkpoint-to-release model_version v7-to-v8 relabel"
        ),
        "aggregate_component_root_sha256": hashlib.sha256(
            aggregate_material
        ).hexdigest(),
        "release_policy": policy,
        "checkpoint_directory": relative(CHECKPOINT_ROOT),
        "post_validation_file_status": (
            "DELETED_AFTER_INCORPORATION_VALIDATION"
            if remove_after_validation
            else "PRESENT_VALIDATED"
        ),
        "records_csv": relative(CHECKPOINT_CSV_PATH),
    }
    write_json(CHECKPOINT_JSON_PATH, payload)
    if remove_after_validation:
        shutil.rmtree(resolved)
        if CHECKPOINT_ROOT.exists():
            raise RuntimeError("checkpoint directory deletion did not complete")
    return inventory


def build_configuration() -> dict[str, Any]:
    configuration = {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "release_status": RELEASE_STATUS,
        "release_kind": "hardening_release_reusing_checksum_validated_v8_predictions",
        "locked_parent_experiment": v8.EXPERIMENT_ID,
        "locked_parent_core_hashes": {
            key: value for key, (_, value) in V8_CORE_HASHES.items()
        },
        "forecasting_inputs_changed": False,
        "prediction_reuse_policy": {
            "required": [
                "all V8 manifest checksums pass",
                "V7 immutability passes",
                "seed and collapsed keys are unique",
                "each collapsed prediction has exactly five seeds",
                "membership, price, feature, target, and model inputs unchanged",
            ],
            "outcome": "REUSE_AUTHORIZED",
        },
        "support_inference": {
            "primary": "full-arm locked V8 comparison",
            "diagnostic": "exact common arm-native prediction-key support",
            "control": "separately trained PIT_CURRENT_INTERSECTION_CONTROL",
            "common_support_retraining": False,
            "arithmetic_component_is_causal": False,
            "match_key": MATCH_KEY + ["actual_class"],
        },
        "bootstrap": {
            "primary_samples": PRIMARY_BOOTSTRAP_SAMPLES,
            "secondary_samples": SECONDARY_BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "primary_metric": "brier_score",
            "primary_block_length": PRIMARY_BLOCK_LENGTH,
            "robustness_block_lengths": ROBUSTNESS_BLOCK_LENGTHS,
            "paired_sampling": True,
            "blocks_confined_to_interval": True,
            "equal_market_primary": True,
            "finite_sample_p_value": "(extreme_count + 1) / (B + 1)",
        },
        "multiple_testing_families": {
            "confirmatory": {
                "family_id": "CONFIRMATORY_MACRO_BRIER_BLOCK20",
                "support": "FULL_SUPPORT",
                "inference_level": "selected_five_market_macro",
                "metric": "brier_score",
                "block_length": 20,
                "models": sorted(v8.MODEL_SPECS),
                "hypothesis_count": 4,
                "comparison": "current minus PIT",
                "adjustment": "Benjamini-Hochberg",
            },
            "secondary_predictive": (
                "separate by support, inference level, metric, and block length"
            ),
            "operational": {
                "metric": "prediction_coverage",
                "narrative_separated_from_predictive_loss": True,
            },
        },
        "aggregation": {
            "primary": "equal_market_macro",
            "secondary": [
                "equal_oos_date",
                "equal_interval",
                "security_date",
                "leave_one_market_out",
                "leave_one_interval_out",
                "exclude_intervals_lt_80_oos_dates",
                "exclude_5014_active_intervals",
            ],
        },
        "malaysia_airports_5014": {
            "recovery_status": "UNRECOVERED_EXPLICIT_GAP",
            "joined_active_member_days": 0,
            "required_active_member_days": 493,
            "eligibility_threshold_changed": False,
            "sensitivity_specifications": [
                "A_LOCKED_PRIMARY",
                "B_EXCLUDE_5014_ALL_ARMS",
                "C_EXCLUDE_5014_ACTIVE_INTERVALS",
            ],
        },
        "checkpoint_policy": "REMOVE_AFTER_VALIDATED_INCORPORATION",
        "external_retrieval": {
            "silent_download_allowed": False,
            "explicit_flag": "--allow-external-retrieval",
        },
        "random_seeds": sorted(int(item) for item in v8.SEEDS),
        "paths": {
            "release_manifest": relative(RELEASE_MANIFEST_JSON),
            "release_envelope": relative(RELEASE_ENVELOPE_PATH),
            "environment_lock": relative(ENVIRONMENT_PATH),
        },
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return configuration


def _installed_package_lock() -> list[str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        packages[normalized] = distribution.version
    return [f"{name}=={packages[name]}" for name in sorted(packages)]


def build_environment_lock() -> dict[str, Any]:
    packages = _installed_package_lock()
    REQUIREMENTS_PATH.write_text("\n".join(packages) + "\n", encoding="utf-8")
    try:
        locale_value = locale.setlocale(locale.LC_ALL, None)
    except locale.Error:
        locale_value = "UNAVAILABLE"
    existing_timestamp = None
    if ENVIRONMENT_PATH.exists():
        try:
            existing_timestamp = json.loads(
                ENVIRONMENT_PATH.read_text(encoding="utf-8")
            ).get("created_at_utc")
        except (json.JSONDecodeError, OSError):
            existing_timestamp = None
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": existing_timestamp
        or datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "locale": locale_value,
        "timezone": "Asia/Saigon",
        "timezone_runtime_source": "repository execution context",
        "environment_variables_used": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "TZ": os.environ.get("TZ"),
        },
        "package_lock_path": relative(REQUIREMENTS_PATH),
        "package_count": len(packages),
        "package_lock_sha256": sha256_file(REQUIREMENTS_PATH),
        "git_repository": {
            "baseline_v8_commit": BASELINE_COMMIT,
            "release_content_commit": (
                "recorded by the release envelope after content commit; "
                "self-referential commit hashes cannot be embedded in that commit"
            ),
            "branch_at_capture": git_output("branch", "--show-current"),
            "final_tag": FINAL_TAG,
        },
        "random_seeds": sorted(int(item) for item in v8.SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "external_data_requirements": [
            "all V7 and V8 portable-manifest records",
            "provider-native raw OHLC already recorded in V8 evidence caches",
            "FBM KLCI source documents/extractions enumerated in V8.1 registry",
        ],
        "validation_download_policy": "NEVER_DOWNLOAD",
        "external_retrieval_requires_explicit_flag": True,
    }
    write_json(ENVIRONMENT_PATH, payload)
    return payload


def build_reproduce_document() -> None:
    REPRODUCE_PATH.write_text(
        """# Reproduce and Validate ASEAN PIT V8.1

V8.1 is a hardening release. It reuses the immutable V8 predictions only after
all input checksums and prediction invariants pass.

## Environment

Use the Python version recorded in
`research_data/point_in_time/config/environment_v8_1.json`, then install the
exact package snapshot:

```powershell
python -m pip install -r requirements-v8.1-lock.txt
```

## One-command modes

```powershell
python scripts/point_in_time/run_pit_v8_1_release.py --validate-only
python scripts/point_in_time/run_pit_v8_1_release.py --rebuild-inference
python scripts/point_in_time/run_pit_v8_1_release.py --full-rerun
```

`--validate-only` is read-only and never retrieves external data.
`--rebuild-inference` validates and reuses the immutable V8 predictions.
`--full-rerun` rebuilds predictions in isolated V8.1 staging, compares them
against the locked V8 outputs, and never overwrites V8. External retrieval is
disabled in every mode unless `--allow-external-retrieval` is supplied; the
current release does not need retrieval.

## Expected checksums

Every expected release-file checksum and size is enumerated in
`pit_multimarket_forecasting_v8_1_release_manifest.json`. Its checksum, the
workspace-inventory checksum, validation checksum, configuration checksum, and
environment-lock checksum are authenticated by
`pit_v8_1_release_envelope.json`. The envelope is intentionally outside the
manifest it authenticates, avoiding unsafe self-hashing.

The checkpoint inventory proves component-wise incorporation of all 165
completed checkpoint Parquets and 165 markers into the final V8 seed output.
After that proof, the checkpoint directory is deleted as disposable execution
state; it is not needed for reproduction.
""",
        encoding="utf-8",
    )


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], *, float_digits: int = 8
) -> list[str]:
    labels = [str(column) for column in columns]
    lines = [
        "| " + " | ".join(labels) + " |",
        "|" + "|".join("---" for _ in labels) + "|",
    ]
    for record in frame.loc[:, list(columns)].to_dict("records"):
        values: list[str] = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)) and pd.notna(value):
                values.append(f"{float(value):.{float_digits}f}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_reconciliation() -> None:
    lines = [
        "# V8 to V8.1 Reconciliation",
        "",
        f"- Locked V8 experiment: `{v8.EXPERIMENT_ID}`",
        f"- Hardened release: `{EXPERIMENT_ID}`",
        f"- V8 baseline Git commit: `{BASELINE_COMMIT}`",
        "- Empirical predictions changed: **no**",
        "- Forecasting membership, price, feature, target, and model inputs changed: **no**",
        "",
        "## Immutable core checksums",
        "",
        "| Artifact | V8 path | Locked SHA-256 | V8.1 treatment |",
        "|---|---|---|---|",
    ]
    for label, (path, expected) in V8_CORE_HASHES.items():
        lines.append(
            f"| {label} | `{relative(path)}` | `{expected}` | reused unchanged |"
        )
    lines.extend(
        [
            "",
            "All 76 V8 selected-manifest records and all 34 V7 records validated. "
            "No V8 or V7 artifact was overwritten.",
            "",
            "## Regenerated analyses",
            "",
            "- Common-support inference was calculated from exact matched existing "
            "arm-native prediction keys; no models were retrained.",
            "- Publication bootstrap inference was regenerated with 10,000 primary "
            "and 5,000 secondary replications.",
            "- Malaysia Airports specifications A/B/C and aggregation robustness "
            "were newly calculated.",
            "",
            "## Newly created controls",
            "",
            "- Separate document/extraction source registry and MIME validation.",
            "- HTTP quarantine audit, checkpoint incorporation/deletion inventory, "
            "release manifest, workspace inventory, validation report, environment "
            "lock, reproduction guide, and release envelope.",
            "",
            "## Checksum differences",
            "",
            "There are no changed checksums for reused V8/V7 artifacts. Every V8.1 "
            "checksum difference is due to a newly created hardening artifact, not "
            "silent replacement of an empirical input or output.",
        ]
    )
    RECONCILIATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_status_and_hardening_reports(
    *,
    integrity: Mapping[str, Any],
    predictions: pd.DataFrame,
    common: pd.DataFrame,
    diagnostic: pd.DataFrame,
    primary: pd.DataFrame,
    aggregation: pd.DataFrame,
    source_registry: pd.DataFrame,
    http_audit: pd.DataFrame,
    checkpoint_inventory: pd.DataFrame,
) -> dict[str, Any]:
    common_macro = common[
        common["aggregation_level"].eq("selected_five_market_macro")
    ].copy()
    diagnostic_macro = diagnostic[
        diagnostic["aggregation_level"].eq("selected_market_macro")
    ].copy()
    full_macro = diagnostic_macro[
        [
            "model_id",
            "delta_brier_score_current_minus_pit_full_support",
            "delta_brier_score_current_minus_pit_common_support",
            "descriptive_support_composition_component_brier_score",
        ]
    ].sort_values("model_id")
    primary_macro = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["inference_level"].eq("selected_five_market_macro")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(PRIMARY_BLOCK_LENGTH)
    ].sort_values("model_id")
    status = {
        "experiment_id": EXPERIMENT_ID,
        "study_label": STUDY_LABEL,
        "experiment_version": EXPERIMENT_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "release_status": RELEASE_STATUS,
        "status": RELEASE_STATUS,
        "five_market_status": "COMPLETE_FIVE_MARKET",
        "locked_parent_experiment": v8.EXPERIMENT_ID,
        "forecasting_inputs_changed": False,
        "predictions_changed_from_v8": False,
        "v8_seed_level_predictions": int(
            json.loads(v8.STATUS_JSON_PATH.read_text(encoding="utf-8"))[
                "seed_level_predictions"
            ]
        ),
        "v8_seed_collapsed_predictions": len(predictions),
        "markets": v8.MARKETS,
        "market_count": len(v8.MARKETS),
        "v7_integrity": integrity["v7_integrity"]["status"],
        "v8_integrity": integrity["status"],
        "v8_manifest_records": integrity["v8_manifest_records"],
        "core_hashes": integrity["core_hashes"],
        "malaysia_airports_5014": {
            "recovered": False,
            "joined_active_member_days": 0,
            "required_active_member_days": 493,
            "limitation_status": "PASS_WITH_DOCUMENTED_LIMITATIONS",
            "sensitivity_complete": True,
        },
        "common_support": {
            "matched_arm_pairs": int(common_macro["common_support_count"].iloc[0]),
            "model_results": common_macro[
                [
                    "model_id",
                    "pit_brier_score",
                    "current_brier_score",
                    "delta_brier_score_current_minus_pit",
                ]
            ].to_dict("records"),
        },
        "full_vs_common_support_brier": full_macro.to_dict("records"),
        "primary_bootstrap": primary_macro[
            [
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "raw_p_value",
                "adjusted_p_value",
                "result_classification",
            ]
        ].to_dict("records"),
        "aggregation_robustness": aggregation[
            aggregation["metric"].eq("brier_score")
        ].to_dict("records"),
        "source_evidence": {
            "registry_records": len(source_registry),
            "local_documents_validated": int(
                source_registry["source_document_local_path"].notna().sum()
            ),
            "2017_anchor_status": str(
                source_registry.loc[
                    source_registry["source_id"].eq("SRC_ANCHOR_2017_12_31"),
                    "evidentiary_status",
                ].iloc[0]
            ),
            "2018_june_status": str(
                source_registry.loc[
                    source_registry["source_id"].eq("SRC_REVIEW_2018_06"),
                    "evidentiary_status",
                ].iloc[0]
            ),
        },
        "challenge_quarantine": {
            "files": len(http_audit),
            "used_as_evidence": bool(
                http_audit["used_as_evidence"].astype(bool).any()
            ),
        },
        "checkpoint_policy": "REMOVE_AFTER_VALIDATED_INCORPORATION",
        "checkpoint_components_validated": int(
            checkpoint_inventory["artifact_type"]
            .eq("CHECKPOINT_PARQUET_COMPONENT")
            .sum()
        ),
        "acceptance_gates": {
            "common_support_complete": True,
            "bootstrap_10000_complete": (
                not primary_macro.empty
                and primary_macro["bootstrap_samples"].eq(
                    PRIMARY_BOOTSTRAP_SAMPLES
                ).all()
            ),
            "source_hash_separation": True,
            "challenge_not_evidence": not http_audit[
                "used_as_evidence"
            ].astype(bool).any(),
            "checkpoint_incorporation": checkpoint_inventory[
                "incorporated_into_final_output"
            ].astype(bool).all(),
            "git_commit_and_tag": "SEALED_AFTER_CONTENT_COMMIT",
        },
        "unresolved_limitations": [
            "Malaysia Airports 5014 raw daily OHLC remains unavailable for 493 active-member days.",
            "The 2017 anchor original PDF could not be cached; a Tier 2 factual extraction with remote filing reference is retained.",
            "The June 2018 review remains Tier 3 attributed evidence with exact endpoint and 2019-anchor cross-validation.",
            "Common-support arithmetic differences are descriptive, not causal decompositions.",
        ],
    }
    write_json(STATUS_JSON_PATH, status)
    lines = [
        "# ASEAN Five-Market PIT Forecasting Status V8.1",
        "",
        f"**Final status:** `{RELEASE_STATUS}`",
        "",
        "V8.1 preserves every forecasting input and prediction from immutable V8. "
        "It adds publication-grade source integrity, exact common-support inference, "
        "10,000-replication primary bootstrap inference, sensitivity analyses, "
        "aggregation robustness, and release controls.",
        "",
        "## Full support versus common support (Brier, current minus PIT)",
        "",
        *_markdown_table(
            full_macro,
            [
                "model_id",
                "delta_brier_score_current_minus_pit_full_support",
                "delta_brier_score_current_minus_pit_common_support",
                "descriptive_support_composition_component_brier_score",
            ],
        ),
        "",
        "The last column is an arithmetic support/composition diagnostic and is "
        "not a causal decomposition. The separately trained intersection-control "
        "arm is reported independently in the inference artifacts.",
        "",
        "## Primary 10,000-replication bootstrap",
        "",
        *_markdown_table(
            primary_macro,
            [
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "raw_p_value",
                "adjusted_p_value",
                "result_classification",
            ],
        ),
        "",
        "Confidence intervals and BH-adjusted p-values must both support a "
        "difference; prediction coverage belongs to a separate operational family.",
        "",
        "## Explicit limitations",
        "",
        "- Malaysia Airports 5014 remains unrecovered (0/493 active-member days).",
        "- The 2017 anchor original PDF remains remotely referenced; its factual "
        "extraction is preserved and labeled Tier 2 without claiming a PDF hash.",
        "- The June 2018 event remains Tier 3 attributed evidence; no primary-source "
        "claim is made.",
        "- Nine failed/challenge responses are quarantined and none is evidence.",
    ]
    STATUS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hardening = [
        "# V8.1 Publication Hardening Report",
        "",
        f"Release decision: `{RELEASE_STATUS}`.",
        "",
        "## Controls completed",
        "",
        "- V7 and all 76 V8 selected-manifest checksums revalidated.",
        "- Prediction-key uniqueness, five-seed collapse, and unchanged-input gates passed.",
        "- Original-document and factual-extraction paths/hashes are separated.",
        "- All local PDF/HTML source payloads passed content-derived MIME and parser checks.",
        "- Failed HTTP and challenge payloads are quarantined and excluded from evidence.",
        "- Exact common-support and separately trained intersection-control results are distinct.",
        "- Primary Brier bootstrap uses 10,000 paired, interval-confined moving-block samples.",
        "- BH families and operational coverage narratives are isolated.",
        "- Equal-market remains primary; six requested robustness specifications plus LOMO/LOIO are secondary.",
        "- All checkpoint components were reconciled to the immutable seed output before deletion.",
        "",
        "## Source decisions",
        "",
        "The 2017 annual-report filing metadata was confirmed through an exchange-filing "
        "mirror, but an authentic downloadable PDF was not recovered. Its prior factual "
        "extraction remains Tier 2 with a remote-document reference. No stronger accessible "
        "primary/archived primary June 2018 notice was located, so its Tier 3 status is "
        "retained with endpoint and 2019-anchor equality cross-validation.",
        "",
        "## Release design",
        "",
        "The release content manifest hashes reproducibility/evidence files. It cannot "
        "hash itself. The separate envelope authenticates the manifest, defined-scope "
        "workspace inventory, validation report, configuration, and environment lock. "
        "The envelope records the content commit; the annotated tag seals the following "
        "envelope-only commit.",
    ]
    HARDENING_REPORT_PATH.write_text(
        "\n".join(hardening) + "\n", encoding="utf-8"
    )
    return status


CONTROL_PLANE_PATHS = {
    RELEASE_MANIFEST_JSON,
    RELEASE_MANIFEST_CSV,
    WORKSPACE_INVENTORY_CSV,
    WORKSPACE_INVENTORY_JSON,
    RELEASE_ENVELOPE_PATH,
    ARTIFACT_VALIDATION_PATH,
}


def _manifest_paths_from(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        REPO_ROOT / str(record["artifact_relative_path"])
        for record in payload["records"]
    ]


def release_content_paths() -> list[Path]:
    """Return the complete portable content set, excluding control-plane cycles."""
    paths: set[Path] = set()
    for manifest_path in [v8.V7_MANIFEST_PATH, v8.MANIFEST_PATH]:
        paths.update(_manifest_paths_from(manifest_path))
        paths.add(manifest_path)
    paths.update(
        {
            v8.V7_STATUS_PATH,
            PIT_ROOT / "qa" / "pit_v7_artifact_validation.json",
            v8.ARTIFACT_VALIDATION_PATH,
            REPO_ROOT / ".gitignore",
            REPO_ROOT / ".gitattributes",
            REPO_ROOT / "scripts" / "manifest_portability_v6.py",
            REPO_ROOT / "scripts" / "point_in_time" / "__init__.py",
            REPO_ROOT
            / "scripts"
            / "point_in_time"
            / "pit_multimarket_forecasting_v6.py",
            REPO_ROOT
            / "scripts"
            / "point_in_time"
            / "pit_multimarket_analytics_v6.py",
            REPO_ROOT / "scripts" / "point_in_time" / "pit_pipeline_v2.py",
            REPO_ROOT / "scripts" / "point_in_time" / "pit_pipeline_v3.py",
            SCRIPT_PATH,
            REPO_ROOT / "tests" / "point_in_time" / "test_pit_v8_1_hardening.py",
            CONFIG_PATH,
            ENVIRONMENT_PATH,
            REQUIREMENTS_PATH,
            REPRODUCE_PATH,
            STATUS_JSON_PATH,
            STATUS_MD_PATH,
            RECONCILIATION_PATH,
            HARDENING_REPORT_PATH,
            SOURCE_REGISTRY_CSV,
            SOURCE_REGISTRY_JSON,
            HTTP_AUDIT_PATH,
            CHECKPOINT_CSV_PATH,
            CHECKPOINT_JSON_PATH,
            TEST_SUMMARY_PATH,
            RECOVERY_CSV_PATH,
            RECOVERY_MD_PATH,
            SENSITIVITY_CSV_PATH,
            SENSITIVITY_MD_PATH,
            COMMON_SUPPORT_CSV_PATH,
            COMMON_SUPPORT_PARQUET_PATH,
            SUPPORT_DIAGNOSTIC_CSV_PATH,
            SUPPORT_DIAGNOSTIC_MD_PATH,
            PRIMARY_BOOTSTRAP_CSV_PATH,
            PRIMARY_BOOTSTRAP_PARQUET_PATH,
            SECONDARY_BOOTSTRAP_CSV_PATH,
            FAMILY_REGISTRY_PATH,
            AGGREGATION_PATH,
            LOMO_PATH,
            LOIO_PATH,
        }
    )
    if SOURCE_REGISTRY_CSV.exists():
        registry = pd.read_csv(SOURCE_REGISTRY_CSV)
        for column in ["source_document_local_path", "extraction_local_path"]:
            for value in registry[column].dropna().astype(str):
                if value:
                    paths.add(REPO_ROOT / value)
    missing = sorted(relative(path) for path in paths if not path.is_file())
    if missing:
        raise RuntimeError("release content files missing: " + " | ".join(missing))
    if paths & CONTROL_PLANE_PATHS:
        raise RuntimeError("control-plane self-reference entered release content")
    return sorted(paths, key=relative)


def artifact_class(path: Path) -> str:
    rel = relative(path)
    if "/sources/" in rel or "/raw_sources/" in rel:
        return "SOURCE_EVIDENCE"
    if "/processed/" in rel or "/reconstructed/" in rel:
        return "DATA_PRODUCT"
    if "/reports/" in rel or rel.startswith("reports/"):
        return "ANALYTICAL_REPORT"
    if "/qa/" in rel:
        return "QUALITY_ASSURANCE"
    if "/config/" in rel or rel.startswith("requirements"):
        return "CONFIGURATION_ENVIRONMENT"
    if rel.startswith("scripts/") or rel.startswith("tests/"):
        return "CODE_TEST"
    if "/manifests/" in rel:
        return "DEPENDENCY_MANIFEST"
    return "RELEASE_CONTROL"


def build_release_manifest() -> dict[str, Any]:
    records = []
    for path in release_content_paths():
        records.append(
            {
                "artifact_relative_path": relative(path),
                "file_type": path.suffix.lower().lstrip(".") or "no_extension",
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "artifact_class": artifact_class(path),
                "reproducibility_required": True,
                "evidence_status": (
                    "RELEASE_EVIDENCE"
                    if artifact_class(path) == "SOURCE_EVIDENCE"
                    else "RELEASE_CONTENT"
                ),
                "manifest_version": MANIFEST_VERSION,
            }
        )
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "manifest_version": MANIFEST_VERSION,
        "manifest_role": "RELEASE_CONTENT_MANIFEST_NOT_WORKSPACE_INVENTORY",
        "artifact_count": len(records),
        "total_release_bytes": sum(item["size_bytes"] for item in records),
        "records": records,
        "control_plane_exclusions": [
            relative(RELEASE_MANIFEST_JSON),
            relative(RELEASE_MANIFEST_CSV),
            relative(WORKSPACE_INVENTORY_CSV),
            relative(WORKSPACE_INVENTORY_JSON),
            relative(ARTIFACT_VALIDATION_PATH),
            relative(RELEASE_ENVELOPE_PATH),
        ],
        "control_plane_design": (
            "The manifest does not hash itself. A separate envelope authenticates "
            "the manifest, workspace inventory, validation, configuration, and "
            "environment lock."
        ),
    }
    write_json(RELEASE_MANIFEST_JSON, payload)
    write_frame(RELEASE_MANIFEST_CSV, pd.DataFrame(records))
    return payload


WORKSPACE_ROOTS = [
    REPORT_ROOT,
    PIT_ROOT,
    REPO_ROOT / "scripts" / "point_in_time",
    REPO_ROOT / "tests" / "point_in_time",
]


def _workspace_related_files(release_paths: set[Path]) -> set[Path]:
    selected: set[Path] = set()
    explicit = {
        SCRIPT_PATH.resolve(),
        (REPO_ROOT / "tests" / "point_in_time" / "test_pit_v8_1_hardening.py").resolve(),
    }
    for root in WORKSPACE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix.lower() in {
                ".pyc",
                ".pyo",
            }:
                continue
            resolved = path.resolve()
            rel_lower = relative(path).lower()
            related_name = any(
                token in rel_lower
                for token in ["v8_1", "v8.1", "pit_v8_1", "fbm_klci_v8_1"]
            )
            if resolved in release_paths or resolved in explicit or related_name:
                selected.add(resolved)
    return selected


def _workspace_record(
    path: Path, *, release_paths: set[Path]
) -> dict[str, Any]:
    rel = relative(path)
    release_included = path.resolve() in release_paths
    quarantine = str(QUARANTINE.resolve()) in str(path.resolve())
    is_control = path.resolve() in {item.resolve() for item in CONTROL_PLANE_PATHS}
    ephemeral = quarantine
    if release_included:
        exclusion_reason = None
    elif quarantine:
        exclusion_reason = "FAILED_HTTP_RESPONSE_QUARANTINED_NOT_EVIDENCE"
    elif is_control:
        exclusion_reason = (
            "CONTROL_PLANE_AUTHENTICATED_BY_ENVELOPE_TO_AVOID_SELF_REFERENCE"
        )
    else:
        exclusion_reason = "V8_1_WORKSPACE_SUPPORT_NOT_REQUIRED_FOR_PORTABLE_RELEASE"
    return {
        "relative_path": rel,
        "file_type": path.suffix.lower().lstrip(".") or "no_extension",
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "artifact_class": artifact_class(path),
        "release_included": release_included,
        "reproducibility_required": release_included,
        "ephemeral": ephemeral,
        "generated_or_source": (
            "SOURCE"
            if artifact_class(path) in {"SOURCE_EVIDENCE", "CODE_TEST"}
            else "GENERATED"
        ),
        "evidence_status": (
            "NOT_EVIDENCE_QUARANTINED"
            if quarantine
            else ("RELEASE_EVIDENCE" if release_included else "CONTROL_OR_SUPPORT")
        ),
        "reason_for_exclusion_from_release_manifest": exclusion_reason,
    }


def build_workspace_inventory(manifest: Mapping[str, Any]) -> dict[str, Any]:
    release_paths = {
        (REPO_ROOT / record["artifact_relative_path"]).resolve()
        for record in manifest["records"]
    }
    # Mutually authenticating control-plane files cannot safely hash themselves
    # or one another. They are outside the defined snapshot and authenticated by
    # the envelope.
    self_exclusions = {path.resolve() for path in CONTROL_PLANE_PATHS}
    files = _workspace_related_files(release_paths) - self_exclusions
    records = [
        _workspace_record(path, release_paths=release_paths)
        for path in sorted(files, key=relative)
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "inventory_role": "DEFINED_SCOPE_PHYSICAL_WORKSPACE_INVENTORY",
        "selection_rule": (
            "all physical files under the four required roots that are release "
            "content, explicitly named V8.1, or the V8.1 runner/test"
        ),
        "roots": [relative(root) for root in WORKSPACE_ROOTS],
        "record_count": len(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "records": records,
        "self_exclusions": [
            {
                "relative_path": relative(path),
                "reason": (
                    "control-plane cycle excluded from physical snapshot and "
                    "authenticated by release envelope"
                ),
            }
            for path in sorted(self_exclusions, key=relative)
        ],
        "checkpoint_policy": "REMOVED_AFTER_VALIDATED_INCORPORATION",
    }
    write_frame(WORKSPACE_INVENTORY_CSV, pd.DataFrame(records))
    write_json(WORKSPACE_INVENTORY_JSON, payload)
    return payload


def _validate_manifest_records(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()
    for record in manifest.get("records", []):
        rel = str(record["artifact_relative_path"])
        if rel in seen:
            failures.append(f"duplicate release manifest path: {rel}")
            continue
        seen.add(rel)
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            failures.append(f"unsafe release path: {rel}")
            continue
        path = REPO_ROOT / rel
        if not path.is_file():
            failures.append(f"missing release file: {rel}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            failures.append(f"release size mismatch: {rel}")
        if sha256_file(path) != str(record["sha256"]):
            failures.append(f"release checksum mismatch: {rel}")
    if len(seen) != int(manifest.get("artifact_count", -1)):
        failures.append("release artifact_count mismatch")
    if sum(int(item["size_bytes"]) for item in manifest["records"]) != int(
        manifest.get("total_release_bytes", -1)
    ):
        failures.append("release total bytes mismatch")
    return failures


def _validate_workspace_inventory(
    inventory: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    clean_copy: bool = False,
) -> list[str]:
    failures: list[str] = []
    release_paths = {
        (REPO_ROOT / record["artifact_relative_path"]).resolve()
        for record in manifest["records"]
    }
    expected = _workspace_related_files(release_paths) - {
        path.resolve() for path in CONTROL_PLANE_PATHS
    }
    recorded = {
        (REPO_ROOT / record["relative_path"]).resolve()
        for record in inventory["records"]
    }
    if not clean_copy and expected != recorded:
        missing = sorted(relative(item) for item in expected - recorded)
        extra = sorted(relative(item) for item in recorded - expected)
        failures.append(
            f"workspace defined-scope mismatch missing={missing} extra={extra}"
        )
    for record in inventory["records"]:
        path = REPO_ROOT / record["relative_path"]
        if not path.is_file():
            if not clean_copy or bool(record["release_included"]):
                failures.append(
                    f"workspace file missing: {record['relative_path']}"
                )
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            failures.append(f"workspace size mismatch: {record['relative_path']}")
        if sha256_file(path) != str(record["sha256"]):
            failures.append(
                f"workspace checksum mismatch: {record['relative_path']}"
            )
    return failures


def _portable_text_paths(manifest: Mapping[str, Any]) -> list[Path]:
    allowed_suffixes = {
        ".py",
        ".md",
        ".yaml",
        ".yml",
        ".json",
        ".csv",
        ".txt",
        ".gitignore",
        ".gitattributes",
    }
    selected: list[Path] = []
    for record in manifest["records"]:
        path = REPO_ROOT / record["artifact_relative_path"]
        if path.suffix.lower() not in allowed_suffixes and path.name not in {
            ".gitignore",
            ".gitattributes",
        }:
            continue
        # Locked V7/V8 artifacts remain immutable and may record historical
        # execution commands. Scan authored V8.1 portable controls only.
        rel = str(record["artifact_relative_path"]).lower()
        authored_v8_1 = (
            "v8_1" in rel
            or "v8.1" in rel
            or path == SCRIPT_PATH
            or path
            == REPO_ROOT
            / "tests"
            / "point_in_time"
            / "test_pit_v8_1_hardening.py"
            or path in {REPRODUCE_PATH, REQUIREMENTS_PATH}
        )
        if record["artifact_class"] == "SOURCE_EVIDENCE" or not authored_v8_1:
            continue
        selected.append(path)
    return selected


def _portable_security_checks(manifest: Mapping[str, Any]) -> dict[str, Any]:
    absolute_path_hits: list[str] = []
    secret_hits: list[str] = []
    secret_patterns = [
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|password)"
            r"\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"
        ),
    ]
    for path in _portable_text_paths(manifest):
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(r"(?i)(?:[A-Z]:\\\\|/mnt/[a-z]/)", text):
            absolute_path_hits.append(relative(path))
        if any(pattern.search(text) for pattern in secret_patterns):
            secret_hits.append(relative(path))
    return {
        "absolute_local_path_hits": sorted(set(absolute_path_hits)),
        "secret_pattern_hits": sorted(set(secret_hits)),
        "status": (
            "PASS"
            if not absolute_path_hits and not secret_hits
            else "FAIL"
        ),
    }


def validate_release(
    *,
    require_envelope: bool,
    require_git_seal: bool,
    deep_predictions: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    integrity = assert_v8_immutable(deep_predictions=deep_predictions)
    manifest = json.loads(RELEASE_MANIFEST_JSON.read_text(encoding="utf-8"))
    inventory = json.loads(WORKSPACE_INVENTORY_JSON.read_text(encoding="utf-8"))
    failures.extend(_validate_manifest_records(manifest))
    clean_copy = not (REPO_ROOT / ".git").exists()
    failures.extend(
        _validate_workspace_inventory(
            inventory, manifest, clean_copy=clean_copy
        )
    )
    observed_manifest_csv = pd.read_csv(RELEASE_MANIFEST_CSV)
    manifest_json_core = {
        (
            str(item["artifact_relative_path"]),
            int(item["size_bytes"]),
            str(item["sha256"]),
        )
        for item in manifest["records"]
    }
    manifest_csv_core = {
        (
            str(item["artifact_relative_path"]),
            int(item["size_bytes"]),
            str(item["sha256"]),
        )
        for item in observed_manifest_csv.to_dict("records")
    }
    if manifest_json_core != manifest_csv_core:
        failures.append("release manifest JSON/CSV serializers differ")
    observed_inventory_csv = pd.read_csv(WORKSPACE_INVENTORY_CSV)
    inventory_json_core = {
        (
            str(item["relative_path"]),
            int(item["size_bytes"]),
            str(item["sha256"]),
        )
        for item in inventory["records"]
    }
    inventory_csv_core = {
        (
            str(item["relative_path"]),
            int(item["size_bytes"]),
            str(item["sha256"]),
        )
        for item in observed_inventory_csv.to_dict("records")
    }
    if inventory_json_core != inventory_csv_core:
        failures.append("workspace inventory JSON/CSV serializers differ")

    registry = pd.read_csv(SOURCE_REGISTRY_CSV)
    manifest_paths = {
        str(item["artifact_relative_path"]) for item in manifest["records"]
    }
    for row in registry.to_dict("records"):
        document = row.get("source_document_local_path")
        extraction = row.get("extraction_local_path")
        document_present = pd.notna(document) and bool(str(document))
        extraction_present = pd.notna(extraction) and bool(str(extraction))
        if document_present:
            path = REPO_ROOT / str(document)
            if str(row["source_document_sha256"]) != sha256_file(path):
                failures.append(f"source document checksum mismatch: {row['source_id']}")
            if str(row["source_document_mime_type"]) != detect_mime(path):
                failures.append(f"source MIME mismatch: {row['source_id']}")
            if is_http_challenge(path):
                failures.append(f"challenge referenced as source: {row['source_id']}")
            if relative(path) not in manifest_paths:
                failures.append(f"source document absent from release: {row['source_id']}")
        elif pd.notna(row.get("source_document_sha256")):
            failures.append(
                f"document hash present without document: {row['source_id']}"
            )
        if extraction_present:
            path = REPO_ROOT / str(extraction)
            if str(row["extraction_sha256"]) != sha256_file(path):
                failures.append(f"extraction checksum mismatch: {row['source_id']}")
            if relative(path) not in manifest_paths:
                failures.append(f"extraction absent from release: {row['source_id']}")
            if document_present and row["source_document_sha256"] == row["extraction_sha256"]:
                failures.append(f"document/extraction hash conflated: {row['source_id']}")
    http_audit = pd.read_csv(HTTP_AUDIT_PATH)
    if http_audit["used_as_evidence"].astype(bool).any():
        failures.append("quarantine response used as evidence")
    if set(http_audit["quarantine_path"].astype(str)) & manifest_paths:
        failures.append("quarantine response included in release content")

    common = pd.read_parquet(COMMON_SUPPORT_PARQUET_PATH)
    primary = pd.read_parquet(PRIMARY_BOOTSTRAP_PARQUET_PATH)
    family = pd.read_csv(FAMILY_REGISTRY_PATH)
    if common[
        common["aggregation_level"].eq("selected_five_market_macro")
    ].shape[0] != 4:
        failures.append("common-support macro does not contain four models")
    confirmatory = primary[
        primary["family_id"].eq("CONFIRMATORY_MACRO_BRIER_BLOCK20")
    ]
    if len(confirmatory) != 4 or not confirmatory["bootstrap_samples"].eq(
        PRIMARY_BOOTSTRAP_SAMPLES
    ).all():
        failures.append("confirmatory 10,000-bootstrap family mismatch")
    registry_confirmatory = family[
        family["family_id"].eq("CONFIRMATORY_MACRO_BRIER_BLOCK20")
    ]
    if (
        len(registry_confirmatory) != 1
        or int(registry_confirmatory.iloc[0]["hypothesis_count"]) != 4
    ):
        failures.append("BH confirmatory family registry mismatch")
    checkpoint = json.loads(CHECKPOINT_JSON_PATH.read_text(encoding="utf-8"))
    if checkpoint["incorporation_validation"] != "PASS":
        failures.append("checkpoint incorporation validation not PASS")
    if checkpoint["post_validation_file_status"] != (
        "DELETED_AFTER_INCORPORATION_VALIDATION"
    ):
        failures.append("checkpoint deletion policy incomplete")
    if CHECKPOINT_ROOT.exists():
        failures.append("checkpoint directory remains after removal policy")

    portable = _portable_security_checks(manifest)
    if portable["status"] != "PASS":
        failures.extend(
            [f"absolute local path: {item}" for item in portable["absolute_local_path_hits"]]
        )
        failures.extend(
            [f"secret pattern: {item}" for item in portable["secret_pattern_hits"]]
        )
    large_files = [
        record["artifact_relative_path"]
        for record in manifest["records"]
        if int(record["size_bytes"]) >= 100 * 1024 * 1024
    ]
    if large_files:
        failures.append(f"release files require LFS but exceed 100 MiB: {large_files}")

    envelope_validation: dict[str, Any] = {"required": require_envelope}
    if require_envelope:
        if not RELEASE_ENVELOPE_PATH.is_file():
            failures.append("release envelope missing")
        else:
            envelope = json.loads(RELEASE_ENVELOPE_PATH.read_text(encoding="utf-8"))
            expected_hashes = {
                "release_manifest_sha256": sha256_file(RELEASE_MANIFEST_JSON),
                "workspace_inventory_sha256": sha256_file(WORKSPACE_INVENTORY_JSON),
                "validation_report_sha256": sha256_file(ARTIFACT_VALIDATION_PATH),
                "configuration_sha256": sha256_file(CONFIG_PATH),
                "environment_lock_sha256": sha256_file(ENVIRONMENT_PATH),
            }
            for key, value in expected_hashes.items():
                if str(envelope.get(key)) != value:
                    failures.append(f"release envelope {key} mismatch")
            if envelope.get("git_tag") != FINAL_TAG:
                failures.append("release envelope tag name mismatch")
            envelope_validation = {
                "required": True,
                "content_commit": envelope.get("git_commit_hash"),
                "git_tag": envelope.get("git_tag"),
                "hashes": expected_hashes,
            }
            if require_git_seal:
                try:
                    commit = git_output(
                        "rev-parse", f"{envelope['git_commit_hash']}^{{commit}}"
                    )
                    tag_object = git_output("cat-file", "-t", FINAL_TAG)
                    tag_message = git_output(
                        "for-each-ref",
                        f"refs/tags/{FINAL_TAG}",
                        "--format=%(contents)",
                    )
                    if commit != envelope["git_commit_hash"]:
                        failures.append("envelope content commit does not resolve exactly")
                    if tag_object != "tag":
                        failures.append("final Git tag is not annotated")
                    if RELEASE_STATUS not in tag_message:
                        failures.append("annotated tag message missing final status")
                except (subprocess.CalledProcessError, KeyError) as exc:
                    failures.append(f"Git seal validation failed: {exc}")
    result = {
        "experiment_id": EXPERIMENT_ID,
        "validation_status": "PASS" if not failures else "FAIL",
        "v7_v8_integrity": integrity,
        "release_manifest_records": int(manifest["artifact_count"]),
        "release_manifest_bytes": int(manifest["total_release_bytes"]),
        "workspace_inventory_records": int(inventory["record_count"]),
        "source_registry_records": len(registry),
        "quarantine_records": len(http_audit),
        "portable_security_checks": portable,
        "oversized_release_files": large_files,
        "envelope_validation": envelope_validation,
        "failures": failures,
    }
    return result


def write_validation_report(
    *,
    require_envelope: bool = False,
    require_git_seal: bool = False,
) -> dict[str, Any]:
    result = validate_release(
        require_envelope=require_envelope,
        require_git_seal=require_git_seal,
        deep_predictions=True,
    )
    write_json(ARTIFACT_VALIDATION_PATH, result)
    if result["validation_status"] != "PASS":
        raise RuntimeError(
            "V8.1 validation failed: " + " | ".join(result["failures"])
        )
    return result


def build_release_envelope(*, content_commit: str) -> dict[str, Any]:
    manifest = json.loads(RELEASE_MANIFEST_JSON.read_text(encoding="utf-8"))
    validation = json.loads(ARTIFACT_VALIDATION_PATH.read_text(encoding="utf-8"))
    if validation["validation_status"] != "PASS":
        raise RuntimeError("cannot seal a failing validation report")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "release_manifest_path": relative(RELEASE_MANIFEST_JSON),
        "release_manifest_sha256": sha256_file(RELEASE_MANIFEST_JSON),
        "workspace_inventory_path": relative(WORKSPACE_INVENTORY_JSON),
        "workspace_inventory_sha256": sha256_file(WORKSPACE_INVENTORY_JSON),
        "validation_report_path": relative(ARTIFACT_VALIDATION_PATH),
        "validation_report_sha256": sha256_file(ARTIFACT_VALIDATION_PATH),
        "configuration_path": relative(CONFIG_PATH),
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "environment_lock_path": relative(ENVIRONMENT_PATH),
        "environment_lock_sha256": sha256_file(ENVIRONMENT_PATH),
        "git_commit_hash": content_commit,
        "git_commit_role": "V8.1_CONTENT_COMMIT_AUTHENTICATED_BY_ENVELOPE",
        "git_tag": FINAL_TAG,
        "tag_seal_design": (
            "annotated tag points to the following envelope-only seal commit"
        ),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_release_files": int(manifest["artifact_count"]),
        "total_release_bytes": int(manifest["total_release_bytes"]),
        "validation_status": validation["validation_status"],
        "self_reference_policy": (
            "envelope is excluded from the release manifest it authenticates"
        ),
    }
    write_json(RELEASE_ENVELOPE_PATH, payload)
    return payload


def record_test_summary(
    *,
    passed: int,
    failed: int,
    skipped: int = 0,
    commands: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if failed == 0 and passed > 0 else "FAIL",
        "passed": int(passed),
        "failed": int(failed),
        "skipped": int(skipped),
        "commands": list(commands),
    }
    write_json(TEST_SUMMARY_PATH, payload)
    return payload


def rebuild_inference() -> dict[str, Any]:
    integrity = assert_v8_immutable(deep_predictions=True)
    build_configuration()
    build_environment_lock()
    build_reproduce_document()
    source_registry, source_validations = build_source_registry()
    http_audit = build_http_audit()
    build_5014_recovery_audit()
    predictions = _load_predictions()
    members = pd.read_parquet(v8.MEMBERSHIP_DAILY_PATH)
    hierarchy = _performance_hierarchy(predictions, members)
    _, common, diagnostic = build_common_support(predictions, hierarchy)
    build_support_markdown(common, diagnostic)
    sensitivity = build_5014_sensitivity(predictions, members)
    primary, secondary, family = build_bootstrap_outputs(
        hierarchy["date_diff"], common
    )
    aggregation, lomo, loio = build_aggregation_robustness(
        predictions, members, hierarchy
    )
    checkpoint_inventory = build_checkpoint_inventory(
        remove_after_validation=True
    )
    write_reconciliation()
    if not TEST_SUMMARY_PATH.exists():
        record_test_summary(
            passed=0,
            failed=1,
            commands=[
                {
                    "scope": "repository test suite",
                    "command": "PENDING",
                    "passed": 0,
                    "failed": 1,
                }
            ],
        )
    status = write_status_and_hardening_reports(
        integrity=integrity,
        predictions=predictions,
        common=common,
        diagnostic=diagnostic,
        primary=primary,
        aggregation=aggregation,
        source_registry=source_registry,
        http_audit=http_audit,
        checkpoint_inventory=checkpoint_inventory,
    )
    manifest = build_release_manifest()
    inventory = build_workspace_inventory(manifest)
    validation = write_validation_report()
    return {
        "status": status["status"],
        "predictions_reused": True,
        "source_documents_validated": len(source_validations),
        "common_support_rows": int(
            common[common["aggregation_level"].eq("date")][
                "common_support_count"
            ].sum()
        ),
        "sensitivity_rows": len(sensitivity),
        "primary_bootstrap_rows": len(primary),
        "secondary_bootstrap_rows": len(secondary),
        "testing_families": len(family),
        "aggregation_rows": len(aggregation),
        "lomo_rows": len(lomo),
        "loio_rows": len(loio),
        "release_files": manifest["artifact_count"],
        "workspace_files": inventory["record_count"],
        "validation": validation["validation_status"],
    }


def full_rerun_predictions() -> dict[str, Any]:
    """Rebuild in isolated staging and compare without touching immutable V8."""
    assert_v8_immutable(deep_predictions=True)
    staging_parent = QA.resolve()
    with tempfile.TemporaryDirectory(
        prefix="pit_v8_1_full_rerun_", dir=staging_parent
    ) as temporary:
        staging = Path(temporary)
        replacements = {
            "SEED_PATH": staging / "seed_level.parquet",
            "COLLAPSED_PATH": staging / "seed_collapsed.parquet",
            "ALL_CELLS_PATH": staging / "all_cells.parquet",
            "MODEL_CHECKPOINT_DIR": staging / "checkpoints",
            "REPORTS": staging,
            "RECONCILIATION_PATH": staging / "v7_v8_reconciliation.md",
            "TEST_SUMMARY_PATH": staging / "test_summary.json",
            "ARTIFACT_VALIDATION_PATH": staging / "artifact_validation.json",
        }
        for name, value in replacements.items():
            setattr(v8, name, value)
        result = v8.run_model_phase()
        rebuilt_seed = replacements["SEED_PATH"]
        rebuilt_collapsed = replacements["COLLAPSED_PATH"]
        comparisons = {}
        for label, rebuilt, locked in [
            ("seed_level", rebuilt_seed, V8_CORE_HASHES["v8_seed_predictions"][0]),
            (
                "seed_collapsed",
                rebuilt_collapsed,
                V8_CORE_HASHES["v8_collapsed_predictions"][0],
            ),
        ]:
            rebuilt_hash = sha256_file(rebuilt)
            locked_hash = sha256_file(locked)
            if rebuilt_hash != locked_hash:
                rebuilt_frame = pd.read_parquet(rebuilt)
                locked_frame = pd.read_parquet(locked)
                if not rebuilt_frame.equals(locked_frame):
                    raise RuntimeError(
                        f"full-rerun {label} differs semantically from locked V8"
                    )
            comparisons[label] = {
                "locked_sha256": locked_hash,
                "rebuilt_sha256": rebuilt_hash,
                "byte_identical": rebuilt_hash == locked_hash,
                "semantic_identical": True,
            }
        return {
            "status": "PASS",
            "isolated_staging_removed_on_exit": True,
            "v8_overwritten": False,
            "model_phase": result,
            "comparisons": comparisons,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--rebuild-inference", action="store_true")
    modes.add_argument("--full-rerun", action="store_true")
    parser.add_argument(
        "--allow-external-retrieval",
        action="store_true",
        help=(
            "Explicitly authorize retrieval for workflows that implement it; "
            "V8.1 validation and deterministic rebuilds do not retrieve data."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        # This branch performs reads only. The retrieval flag is intentionally
        # ignored because validation never downloads.
        result = validate_release(
            require_envelope=RELEASE_ENVELOPE_PATH.exists(),
            require_git_seal=(REPO_ROOT / ".git").exists()
            and RELEASE_ENVELOPE_PATH.exists(),
            deep_predictions=True,
        )
    elif args.rebuild_inference:
        result = rebuild_inference()
    else:
        result = full_rerun_predictions()
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0 if result.get("status", result.get("validation_status")) in {
        "PASS",
        RELEASE_STATUS,
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
