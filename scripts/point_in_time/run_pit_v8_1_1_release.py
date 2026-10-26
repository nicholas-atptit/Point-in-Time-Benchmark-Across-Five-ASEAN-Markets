#!/usr/bin/env python3
"""Build and validate the additive V8.1.1 inference-label patch.

V8.1.1 never trains a model and never rewrites a V7, V8, or V8.1 artifact.
It corrects macro-scope labels in the already completed 10,000-replication
bootstrap output and adds deterministic calibration-estimability warnings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.point_in_time import run_pit_v8_1_release as v81  # noqa: E402


EXPERIMENT_ID = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V8_1_1"
EXPERIMENT_VERSION = "v8.1.1"
RELEASE_STATUS = "FINAL_HARDENED_COMPLETE_FIVE_MARKET_INFERENCE_LABEL_PATCH"
MANIFEST_VERSION = "manifest-portability-v8.1.1"
BASE_V81_TAG = "pit-five-market-v8.1-final"
BASE_V81_COMMIT = "d9728f9fe99e20cd78e136aa677bc35908dc9025"
FINAL_TAG = "pit-five-market-v8.1.1-final"

MARKET_ORDER = ["IDX30", "SET50", "VN30", "PSEI", "FBM_KLCI"]
MARKET_COUNT = len(MARKET_ORDER)
PRIMARY_BLOCK_LENGTH = 20
ROBUSTNESS_BLOCK_LENGTH = 40

CALIBRATION_MIN_SAMPLE_SIZE = 30
CALIBRATION_MIN_CLASS_COUNT = 5
CALIBRATION_SEPARATION_ABS_COEFFICIENT = 15.0
CALIBRATION_EXTREME_ABS_COEFFICIENT = 25.0

PIT_ROOT = REPO_ROOT / "research_data" / "point_in_time"
REPORT_ROOT = REPO_ROOT / "reports" / "data_execution"
REPORTS = PIT_ROOT / "reports"
CONFIG = PIT_ROOT / "config"
MANIFESTS = PIT_ROOT / "manifests"
QA = PIT_ROOT / "qa"

STATUS_MD_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8_1_1.md"
STATUS_JSON_PATH = REPORT_ROOT / "PIT_MULTIMARKET_FORECASTING_STATUS_V8_1_1.json"
RECONCILIATION_PATH = REPORT_ROOT / "V8_1_V8_1_1_RECONCILIATION.md"
HARDENING_REPORT_PATH = REPORT_ROOT / "V8_1_1_HARDENING_REPORT.md"
CONFIG_PATH = CONFIG / "pit_multimarket_forecasting_v8_1_1.yaml"

PRIMARY_CSV_PATH = REPORTS / "pit_multimarket_primary_bootstrap_10000_v8_1_1.csv"
PRIMARY_PARQUET_PATH = (
    REPORTS / "pit_multimarket_primary_bootstrap_10000_v8_1_1.parquet"
)
COMMON_SUPPORT_PATH = (
    REPORTS / "pit_multimarket_common_support_inference_v8_1_1.csv"
)
FAMILY_REGISTRY_PATH = REPORTS / "multiple_testing_family_registry_v8_1_1.csv"
AVAILABILITY_AUDIT_PATH = (
    REPORTS / "bootstrap_market_availability_audit_v8_1_1.csv"
)
CALIBRATION_AUDIT_PATH = REPORTS / "calibration_estimability_audit_v8_1_1.csv"

RELEASE_MANIFEST_PATH = (
    MANIFESTS / "pit_multimarket_forecasting_v8_1_1_release_manifest.json"
)
RELEASE_ENVELOPE_PATH = MANIFESTS / "pit_v8_1_1_release_envelope.json"
VALIDATION_PATH = QA / "pit_v8_1_1_artifact_validation.json"
TEST_PATH = (
    REPO_ROOT
    / "tests"
    / "point_in_time"
    / "test_pit_v8_1_1_inference_patch.py"
)
TEST_COMPAT_PATH = REPO_ROOT / "tests" / "point_in_time" / "conftest.py"

BASE_PRIMARY_PARQUET = v81.PRIMARY_BOOTSTRAP_PARQUET_PATH
BASE_PRIMARY_CSV = v81.PRIMARY_BOOTSTRAP_CSV_PATH
BASE_COMMON_SUPPORT = v81.COMMON_SUPPORT_CSV_PATH
BASE_FAMILY_REGISTRY = v81.FAMILY_REGISTRY_PATH
BASE_STATUS_JSON = v81.STATUS_JSON_PATH
BASE_HARDENING_REPORT = v81.HARDENING_REPORT_PATH
BASE_RELEASE_MANIFEST = v81.RELEASE_MANIFEST_JSON
BASE_RELEASE_ENVELOPE = v81.RELEASE_ENVELOPE_PATH
BASE_VALIDATION = v81.ARTIFACT_VALIDATION_PATH
BASE_WORKSPACE_INVENTORY_JSON = v81.WORKSPACE_INVENTORY_JSON
BASE_WORKSPACE_INVENTORY_CSV = v81.WORKSPACE_INVENTORY_CSV
BASE_RELEASE_MANIFEST_CSV = v81.RELEASE_MANIFEST_CSV

BASE_FILE_HASHES = {
    "v8_1_primary_bootstrap_csv": (
        BASE_PRIMARY_CSV,
        "6bb12cb942f3c3dd8c8c81700dcf3f88b051f52629c12c6c90a47c3dadcc9890",
    ),
    "v8_1_primary_bootstrap_parquet": (
        BASE_PRIMARY_PARQUET,
        "435bc887535bcfdd2c863f795b3c9909fbeba1395a4a1de84aab51979574799a",
    ),
    "v8_1_common_support_csv": (
        BASE_COMMON_SUPPORT,
        "9fe079fe0f4e34d90de31a64aedf09fed7c33ff38c2f89bccea7ba48d74e65dd",
    ),
    "v8_1_family_registry_csv": (
        BASE_FAMILY_REGISTRY,
        "9ed727cef3515c7dfa8b1235732eb4641a626c1b44032c27d3b913049ad09ba8",
    ),
    "v8_1_status_json": (
        BASE_STATUS_JSON,
        "f5313e4cc4a38c53ea240154b801f3242275f7f969bc7dd8cad9abbccabad33a",
    ),
    "v8_1_hardening_report": (
        BASE_HARDENING_REPORT,
        "a043a5129ceba3de0ba5d28ed8840cbbe09ef1b7e1dcf02edc6b67907417e5ac",
    ),
    "v8_1_release_envelope": (
        BASE_RELEASE_ENVELOPE,
        "885c620caa68f2d575f28617ed303798b76c89f61de7faee320bed547631f062",
    ),
}

STATISTIC_COLUMNS = [
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "raw_p_value",
    "adjusted_p_value",
]

CALIBRATION_COEFFICIENT_COLUMNS = [
    "pit_calibration_intercept",
    "current_calibration_intercept",
    "pit_calibration_slope",
    "current_calibration_slope",
]

REQUIRED_CALIBRATION_FIELDS = [
    "calibration_estimable",
    "calibration_sample_size",
    "calibration_positive_count",
    "calibration_negative_count",
    "calibration_separation_warning",
    "calibration_extreme_coefficient_warning",
]


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
    if isinstance(value, (datetime, date, pd.Timestamp)):
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
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_base_immutable(*, deep_predictions: bool = True) -> dict[str, Any]:
    integrity = v81.assert_v8_immutable(deep_predictions=deep_predictions)
    failures: list[str] = []
    observed: dict[str, str] = {}
    for label, (path, expected) in BASE_FILE_HASHES.items():
        actual = sha256_file(path)
        observed[label] = actual
        if actual != expected:
            failures.append(f"{label} checksum drift: {actual} != {expected}")
    base_envelope = json.loads(BASE_RELEASE_ENVELOPE.read_text(encoding="utf-8"))
    if base_envelope["git_tag"] != BASE_V81_TAG:
        failures.append("V8.1 envelope tag metadata drift")
    if (REPO_ROOT / ".git").exists():
        try:
            resolved = git_output("rev-parse", f"{BASE_V81_TAG}^{{}}")
            if resolved != BASE_V81_COMMIT:
                failures.append(
                    f"V8.1 tag drift: {resolved} != {BASE_V81_COMMIT}"
                )
        except subprocess.CalledProcessError as exc:
            failures.append(f"V8.1 tag unavailable: {exc}")
    result = {
        "status": "PASS" if not failures else "FAIL",
        "v7_v8_integrity": integrity,
        "v8_1_tag": BASE_V81_TAG,
        "v8_1_commit": BASE_V81_COMMIT,
        "v8_1_file_hashes": observed,
        "prediction_hashes": {
            key: value
            for key, value in integrity["core_hashes"].items()
            if "prediction" in key
        },
        "seed_level_rows": integrity["prediction_checks"].get(
            "seed_rows", 5_695_980
        ),
        "collapsed_rows": integrity["prediction_checks"].get(
            "collapsed_rows", 1_139_196
        ),
        "failures": failures,
    }
    if failures:
        raise RuntimeError("base immutability failed: " + " | ".join(failures))
    return result


def benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _market_lists(rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    valid_by_market = {
        str(record["index_id"]): bool(record["valid"])
        for record in rows.to_dict("records")
    }
    contributing = [
        market for market in MARKET_ORDER if valid_by_market.get(market, False)
    ]
    excluded = [market for market in MARKET_ORDER if market not in contributing]
    return contributing, excluded


def _list_text(values: Sequence[str]) -> str:
    return ";".join(values)


def _available_family_id(support: str, block_length: int) -> str:
    return (
        f"EXPLORATORY_{support.lower()}_available_market_macro_"
        f"brier_score_BLOCK{block_length}"
    )


def _unavailable_five_family_id(support: str, block_length: int) -> str:
    return (
        f"EXPLORATORY_{support.lower()}_five_market_macro_"
        f"brier_score_BLOCK{block_length}_UNAVAILABLE"
    )


def patch_bootstrap_schema(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split incomplete macro estimates into invalid-five and available scopes."""
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    group_keys = [
        "support_specification",
        "model_id",
        "metric",
        "block_length",
    ]
    for values, group in source.groupby(group_keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        group_info = dict(zip(group_keys, values))
        markets = group[group["inference_level"].eq("market")].copy()
        macro = group[
            group["inference_level"].eq("selected_five_market_macro")
        ].copy()
        if len(macro) != 1 or len(markets) != MARKET_COUNT:
            raise RuntimeError(f"unexpected V8.1 bootstrap group shape: {group_info}")
        contributing, excluded = _market_lists(markets)
        block_length = int(group_info["block_length"])
        exclusion_reason = (
            "insufficient block-length-40 information"
            if block_length == ROBUSTNESS_BLOCK_LENGTH and excluded
            else ("scope unavailable" if excluded else None)
        )
        for record in markets.to_dict("records"):
            valid = bool(record["valid"])
            market = str(record["index_id"])
            record.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "required_market_count": 1,
                    "contributing_market_count": 1 if valid else 0,
                    "contributing_markets": market if valid else "",
                    "excluded_markets": "" if valid else market,
                    "all_required_markets_available": valid,
                    "macro_scope": "MARKET_LEVEL",
                    "scope_validity_reason": (
                        "market-level information available"
                        if valid
                        else (
                            "insufficient block-length-40 information"
                            if block_length == ROBUSTNESS_BLOCK_LENGTH
                            else str(record.get("reason") or "scope unavailable")
                        )
                    ),
                }
            )
            output_rows.append(record)

        macro_record = macro.iloc[0].to_dict()
        all_available = len(contributing) == MARKET_COUNT
        if all_available:
            macro_record.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "required_market_count": MARKET_COUNT,
                    "contributing_market_count": MARKET_COUNT,
                    "contributing_markets": _list_text(contributing),
                    "excluded_markets": "",
                    "all_required_markets_available": True,
                    "macro_scope": "FIVE_MARKET_MACRO",
                    "scope_validity_reason": (
                        "all five predefined markets available"
                    ),
                }
            )
            output_rows.append(macro_record)
        else:
            invalid_five = dict(macro_record)
            for column in STATISTIC_COLUMNS:
                invalid_five[column] = np.nan
            invalid_five.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "valid": False,
                    "finite_sample_extreme_count": 0,
                    "reason": "one or more predefined markets unavailable",
                    "result_classification": "INSUFFICIENT_INFORMATION",
                    "family_id": _unavailable_five_family_id(
                        str(group_info["support_specification"]), block_length
                    ),
                    "confirmatory_or_exploratory": "EXPLORATORY",
                    "required_market_count": MARKET_COUNT,
                    "contributing_market_count": len(contributing),
                    "contributing_markets": _list_text(contributing),
                    "excluded_markets": _list_text(excluded),
                    "all_required_markets_available": False,
                    "macro_scope": "FIVE_MARKET_MACRO",
                    "scope_validity_reason": (
                        "one or more predefined markets unavailable"
                    ),
                }
            )
            output_rows.append(invalid_five)

            available = dict(macro_record)
            available.update(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "inference_level": "available_market_macro",
                    "valid": True,
                    "reason": None,
                    "family_id": _available_family_id(
                        str(group_info["support_specification"]), block_length
                    ),
                    "confirmatory_or_exploratory": "EXPLORATORY",
                    "required_market_count": MARKET_COUNT,
                    "contributing_market_count": len(contributing),
                    "contributing_markets": _list_text(contributing),
                    "excluded_markets": _list_text(excluded),
                    "all_required_markets_available": False,
                    "macro_scope": "AVAILABLE_MARKET_MACRO",
                    "scope_validity_reason": (
                        "estimate covers available markets only; "
                        "one or more predefined markets unavailable"
                    ),
                }
            )
            output_rows.append(available)

        audit_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                **group_info,
                "required_market_count": MARKET_COUNT,
                "contributing_market_count": len(contributing),
                "contributing_markets": _list_text(contributing),
                "excluded_markets": _list_text(excluded),
                "all_required_markets_available": all_available,
                "five_market_macro_valid": all_available,
                "available_market_macro_emitted": not all_available,
                "excluded_market_reasons": (
                    ";".join(
                        f"{market}: {exclusion_reason}" for market in excluded
                    )
                    if excluded
                    else ""
                ),
                "scope_validity_reason": (
                    "all five predefined markets available"
                    if all_available
                    else "one or more predefined markets unavailable"
                ),
            }
        )

    result = pd.DataFrame(output_rows)
    available_mask = result["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    for _, indexes in result[available_mask].groupby("family_id").groups.items():
        adjusted = benjamini_hochberg(
            result.loc[indexes, "raw_p_value"].to_numpy(dtype=float)
        )
        result.loc[indexes, "adjusted_p_value"] = adjusted
        supported = (
            (result.loc[indexes, "adjusted_p_value"] <= 0.05)
            & (
                (result.loc[indexes, "ci_lower"] > 0)
                | (result.loc[indexes, "ci_upper"] < 0)
            )
        )
        result.loc[indexes, "result_classification"] = np.where(
            supported,
            "SUPPORTED_CURRENT_PIT_DIFFERENCE",
            "NO_SUPPORTED_DIFFERENCE",
        )
    audit = pd.DataFrame(audit_rows)

    block40 = result[
        result["support_specification"].eq("FULL_SUPPORT")
        & result["block_length"].eq(ROBUSTNESS_BLOCK_LENGTH)
    ]
    invalid_five = block40[block40["macro_scope"].eq("FIVE_MARKET_MACRO")]
    available = block40[
        block40["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    ]
    if len(invalid_five) != 4 or invalid_five["valid"].astype(bool).any():
        raise RuntimeError("block-40 five-market correction is incomplete")
    if len(available) != 4:
        raise RuntimeError("block-40 available-market rows missing")
    if not available["contributing_market_count"].eq(4).all():
        raise RuntimeError("block-40 available macros do not contain four markets")
    if not available["excluded_markets"].eq("IDX30").all():
        raise RuntimeError("IDX30 exclusion is not explicit")
    return result, audit


def build_family_registry(patched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for family_id, group in patched.groupby("family_id", sort=True):
        first = group.iloc[0]
        valid = group[
            group["valid"].astype(bool) & group["raw_p_value"].notna()
        ]
        rows.append(
            {
                "family_id": family_id,
                "confirmatory_or_exploratory": first[
                    "confirmatory_or_exploratory"
                ],
                "support_specification": first["support_specification"],
                "inference_level": first["inference_level"],
                "macro_scope": first["macro_scope"],
                "metric": first["metric"],
                "block_length": int(first["block_length"]),
                "models": ";".join(sorted(set(group["model_id"].astype(str)))),
                "hypothesis_count": int(len(valid)),
                "adjustment_method": "Benjamini-Hochberg",
                "inclusion_rule": (
                    "all valid current-minus-PIT hypotheses with identical "
                    "support, scope, metric, and block length"
                ),
                "exclusion_rule": (
                    "unavailable five-market scope, other scopes, other support "
                    "definitions, other metrics, or other block lengths"
                ),
                "available_market_family_is_separate": bool(
                    first["macro_scope"] == "AVAILABLE_MARKET_MACRO"
                ),
            }
        )
    registry = pd.DataFrame(rows)
    available = registry[
        registry["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    ]
    if available.empty:
        raise RuntimeError("available-market exploratory family missing")
    if not available["confirmatory_or_exploratory"].eq("EXPLORATORY").all():
        raise RuntimeError("available-market family is not exploratory")
    return registry


def calibration_flags(
    *,
    sample_size: int,
    positive_count: int,
    negative_count: int,
    coefficients: Sequence[float],
) -> dict[str, Any]:
    values = np.asarray(coefficients, dtype=float)
    finite = bool(np.isfinite(values).all())
    finite_values = np.abs(values[np.isfinite(values)])
    coefficient_separation = bool(
        len(finite_values)
        and (finite_values > CALIBRATION_SEPARATION_ABS_COEFFICIENT).any()
    )
    class_separation = min(positive_count, negative_count) < (
        CALIBRATION_MIN_CLASS_COUNT
    )
    separation = bool(class_separation or coefficient_separation)
    extreme = bool(
        len(finite_values)
        and (finite_values > CALIBRATION_EXTREME_ABS_COEFFICIENT).any()
    )
    small_sample = sample_size < CALIBRATION_MIN_SAMPLE_SIZE
    estimable = bool(
        not small_sample
        and positive_count >= CALIBRATION_MIN_CLASS_COUNT
        and negative_count >= CALIBRATION_MIN_CLASS_COUNT
        and finite
        and not separation
        and not extreme
    )
    reasons: list[str] = []
    if small_sample:
        reasons.append(
            f"sample_size_below_{CALIBRATION_MIN_SAMPLE_SIZE}"
        )
    if class_separation:
        reasons.append(
            f"class_count_below_{CALIBRATION_MIN_CLASS_COUNT}"
        )
    if not finite:
        reasons.append("nonfinite_coefficient")
    if coefficient_separation:
        reasons.append(
            "near_separation_coefficient_exceeds_"
            f"{CALIBRATION_SEPARATION_ABS_COEFFICIENT:g}"
        )
    if extreme:
        reasons.append(
            "extreme_coefficient_exceeds_"
            f"{CALIBRATION_EXTREME_ABS_COEFFICIENT:g}"
        )
    return {
        "calibration_estimable": estimable,
        "calibration_sample_size": int(sample_size),
        "calibration_positive_count": int(positive_count),
        "calibration_negative_count": int(negative_count),
        "calibration_separation_warning": separation,
        "calibration_extreme_coefficient_warning": extreme,
        "calibration_nonfinite_coefficient_warning": not finite,
        "calibration_small_sample_warning": small_sample,
        "calibration_estimability_reason": (
            "ELIGIBLE" if estimable else ";".join(reasons)
        ),
        "calibration_aggregate_eligible": estimable,
    }


def _date_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    prediction_date = pd.Timestamp(record["prediction_date"]).date().isoformat()
    return (
        str(record["market"]),
        str(record["index_id"]),
        str(record["interval_id"]),
        str(record["fold_id"]),
        str(record["model_id"]),
        prediction_date,
    )


def build_calibration_hardening(
    source: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = v81._load_predictions()
    matched = v81.match_common_support_predictions(predictions)
    date_keys = [
        "market",
        "index_id",
        "interval_id",
        "fold_id",
        "model_id",
        "prediction_date",
    ]
    target_counts: dict[tuple[str, ...], dict[str, int]] = {}
    for values, group in matched.groupby(date_keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        normalized = tuple(
            pd.Timestamp(value).date().isoformat()
            if key == "prediction_date"
            else str(value)
            for key, value in zip(date_keys, values)
        )
        positives = int(group["actual_class"].astype(int).sum())
        target_counts[normalized] = {
            "sample_size": len(group),
            "positive_count": positives,
            "negative_count": len(group) - positives,
        }

    result = source.copy()
    for field in REQUIRED_CALIBRATION_FIELDS + [
        "calibration_nonfinite_coefficient_warning",
        "calibration_small_sample_warning",
        "calibration_estimability_reason",
        "calibration_aggregate_eligible",
        "calibration_total_date_count",
        "calibration_flagged_date_count",
        "calibration_aggregate_inference_status",
    ]:
        result[field] = pd.NA

    audit_rows: list[dict[str, Any]] = []
    date_mask = result["aggregation_level"].eq("date")
    for index, record in result[date_mask].iterrows():
        key = _date_key(record)
        counts = target_counts.get(key)
        if counts is None:
            raise RuntimeError(f"common-support target counts missing: {key}")
        flags = calibration_flags(
            **counts,
            coefficients=[
                record[column] for column in CALIBRATION_COEFFICIENT_COLUMNS
            ],
        )
        for field, value in flags.items():
            result.at[index, field] = value
        result.at[index, "calibration_total_date_count"] = 1
        result.at[index, "calibration_flagged_date_count"] = int(
            not flags["calibration_estimable"]
        )
        result.at[index, "calibration_aggregate_inference_status"] = (
            "ELIGIBLE_DATE_CELL"
            if flags["calibration_estimable"]
            else "FLAGGED_NOT_ELIGIBLE_FOR_UNQUALIFIED_AGGREGATE_INFERENCE"
        )
        audit_rows.append(
            {
                "experiment_id": EXPERIMENT_ID,
                **{key_name: record[key_name] for key_name in date_keys},
                **counts,
                **{
                    column: record[column]
                    for column in CALIBRATION_COEFFICIENT_COLUMNS
                },
                **flags,
                "minimum_sample_size_threshold": CALIBRATION_MIN_SAMPLE_SIZE,
                "minimum_class_count_threshold": CALIBRATION_MIN_CLASS_COUNT,
                "separation_abs_coefficient_threshold": (
                    CALIBRATION_SEPARATION_ABS_COEFFICIENT
                ),
                "extreme_abs_coefficient_threshold": (
                    CALIBRATION_EXTREME_ABS_COEFFICIENT
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)

    def summarize(rows: pd.DataFrame) -> dict[str, Any]:
        total = len(rows)
        flagged = int((~rows["calibration_estimable"].astype(bool)).sum())
        estimable = flagged == 0 and total > 0
        return {
            "calibration_estimable": estimable,
            "calibration_sample_size": int(rows["sample_size"].sum()),
            "calibration_positive_count": int(rows["positive_count"].sum()),
            "calibration_negative_count": int(rows["negative_count"].sum()),
            "calibration_separation_warning": bool(
                rows["calibration_separation_warning"].astype(bool).any()
            ),
            "calibration_extreme_coefficient_warning": bool(
                rows["calibration_extreme_coefficient_warning"].astype(bool).any()
            ),
            "calibration_nonfinite_coefficient_warning": bool(
                rows["calibration_nonfinite_coefficient_warning"].astype(bool).any()
            ),
            "calibration_small_sample_warning": bool(
                rows["calibration_small_sample_warning"].astype(bool).any()
            ),
            "calibration_estimability_reason": (
                "ALL_DATE_CELLS_ELIGIBLE"
                if estimable
                else f"{flagged}_OF_{total}_DATE_CELLS_FLAGGED"
            ),
            "calibration_aggregate_eligible": estimable,
            "calibration_total_date_count": total,
            "calibration_flagged_date_count": flagged,
            "calibration_aggregate_inference_status": (
                "ELIGIBLE_ALL_DATE_CELLS"
                if estimable
                else (
                    "FLAGGED_UNSTABLE_DATE_CELLS_NOT_ELIGIBLE_FOR_"
                    "UNQUALIFIED_AGGREGATE_INFERENCE"
                )
            ),
        }

    for index, record in result[~date_mask].iterrows():
        level = str(record["aggregation_level"])
        selected = audit
        if level == "interval":
            selected = selected[
                selected["index_id"].astype(str).eq(str(record["index_id"]))
                & selected["interval_id"].astype(str).eq(
                    str(record["interval_id"])
                )
                & selected["model_id"].astype(str).eq(str(record["model_id"]))
            ]
        elif level == "market":
            selected = selected[
                selected["index_id"].astype(str).eq(str(record["index_id"]))
                & selected["model_id"].astype(str).eq(str(record["model_id"]))
            ]
        elif level in {
            "selected_five_market_macro",
            "separately_trained_intersection_control",
        }:
            selected = selected[
                selected["model_id"].astype(str).eq(str(record["model_id"]))
            ]
        else:
            raise RuntimeError(f"unknown common-support level: {level}")
        summary = summarize(selected)
        for field, value in summary.items():
            result.at[index, field] = value

    for field in [
        "calibration_estimable",
        "calibration_separation_warning",
        "calibration_extreme_coefficient_warning",
        "calibration_nonfinite_coefficient_warning",
        "calibration_small_sample_warning",
        "calibration_aggregate_eligible",
    ]:
        result[field] = result[field].astype(bool)
    if not set(REQUIRED_CALIBRATION_FIELDS).issubset(result.columns):
        raise RuntimeError("required calibration fields missing")
    return result, audit


def build_configuration() -> dict[str, Any]:
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "release_status": RELEASE_STATUS,
        "release_kind": "inference_reporting_label_patch",
        "immutable_parent_release": {
            "experiment_id": v81.EXPERIMENT_ID,
            "git_tag": BASE_V81_TAG,
            "git_commit": BASE_V81_COMMIT,
        },
        "forecasting_changes": {
            "inputs": False,
            "memberships": False,
            "features": False,
            "folds": False,
            "models": False,
            "seeds": False,
            "predictions": False,
            "primary_point_estimates": False,
        },
        "macro_scope_schema": {
            "required_market_count": MARKET_COUNT,
            "predefined_markets": MARKET_ORDER,
            "scopes": [
                "FIVE_MARKET_MACRO",
                "AVAILABLE_MARKET_MACRO",
                "MARKET_LEVEL",
            ],
            "five_market_validity_rule": (
                "valid only when all five predefined markets contribute"
            ),
            "available_market_family_rule": (
                "always exploratory and BH-adjusted separately"
            ),
        },
        "calibration_thresholds": {
            "minimum_sample_size": CALIBRATION_MIN_SAMPLE_SIZE,
            "minimum_positive_count": CALIBRATION_MIN_CLASS_COUNT,
            "minimum_negative_count": CALIBRATION_MIN_CLASS_COUNT,
            "separation_abs_coefficient_threshold": (
                CALIBRATION_SEPARATION_ABS_COEFFICIENT
            ),
            "extreme_abs_coefficient_threshold": (
                CALIBRATION_EXTREME_ABS_COEFFICIENT
            ),
            "aggregate_rule": (
                "any flagged date cell makes unqualified aggregate calibration "
                "inference ineligible; coefficients are retained with warnings"
            ),
        },
        "primary_inference": {
            "support": "FULL_SUPPORT",
            "macro_scope": "FIVE_MARKET_MACRO",
            "metric": "brier_score",
            "block_length": PRIMARY_BLOCK_LENGTH,
            "models": sorted(v81.v8.MODEL_SPECS),
            "conclusion": "NO_SUPPORTED_DIFFERENCE_FOR_ALL_FOUR_MODELS",
        },
        "robustness_inference": {
            "block_length": ROBUSTNESS_BLOCK_LENGTH,
            "unavailable_market": "IDX30",
            "available_markets": MARKET_ORDER[1:],
            "classification": "EXPLORATORY",
        },
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return payload


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], digits: int = 9
) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for record in frame.loc[:, list(columns)].to_dict("records"):
        values = []
        for column in columns:
            value = record[column]
            if isinstance(value, (float, np.floating)) and pd.notna(value):
                values.append(f"{float(value):.{digits}f}")
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_reports(
    *,
    integrity: Mapping[str, Any],
    bootstrap: pd.DataFrame,
    calibration_audit: pd.DataFrame,
) -> dict[str, Any]:
    confirmatory = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")
        & bootstrap["block_length"].eq(PRIMARY_BLOCK_LENGTH)
    ].sort_values("model_id")
    unavailable40 = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")
        & bootstrap["block_length"].eq(ROBUSTNESS_BLOCK_LENGTH)
    ].sort_values("model_id")
    available40 = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
        & bootstrap["block_length"].eq(ROBUSTNESS_BLOCK_LENGTH)
    ].sort_values("model_id")
    historical = available40[
        available40["model_id"].eq("historical_class_probability")
    ].iloc[0]
    warning_counts = {
        "date_rows": len(calibration_audit),
        "estimable_date_rows": int(
            calibration_audit["calibration_estimable"].astype(bool).sum()
        ),
        "flagged_date_rows": int(
            (~calibration_audit["calibration_estimable"].astype(bool)).sum()
        ),
        "separation_warning_rows": int(
            calibration_audit[
                "calibration_separation_warning"
            ].astype(bool).sum()
        ),
        "extreme_coefficient_warning_rows": int(
            calibration_audit[
                "calibration_extreme_coefficient_warning"
            ].astype(bool).sum()
        ),
        "nonfinite_coefficient_warning_rows": int(
            calibration_audit[
                "calibration_nonfinite_coefficient_warning"
            ].astype(bool).sum()
        ),
        "small_sample_warning_rows": int(
            calibration_audit[
                "calibration_small_sample_warning"
            ].astype(bool).sum()
        ),
    }
    status = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "release_status": RELEASE_STATUS,
        "status": RELEASE_STATUS,
        "five_market_forecasting_status": "COMPLETE_FIVE_MARKET",
        "patch_scope": "INFERENCE_LABELS_FAMILIES_CALIBRATION_FLAGS_REPORTS_METADATA",
        "predictions_changed": False,
        "seed_level_rows": 5_695_980,
        "collapsed_rows": 1_139_196,
        "prediction_hashes": integrity["prediction_hashes"],
        "confirmatory_block20_five_market": confirmatory.to_dict("records"),
        "block40_five_market": unavailable40.to_dict("records"),
        "block40_available_market": available40.to_dict("records"),
        "historical_probability_sensitivity": {
            "point_estimate": float(historical["point_estimate"]),
            "ci_lower": float(historical["ci_lower"]),
            "ci_upper": float(historical["ci_upper"]),
            "adjusted_p_value": float(historical["adjusted_p_value"]),
            "classification": historical["result_classification"],
            "scope": historical["macro_scope"],
            "contributing_markets": historical["contributing_markets"],
            "excluded_markets": historical["excluded_markets"],
        },
        "calibration_warnings": warning_counts,
        "conclusions": {
            "confirmatory": (
                "Under the preregistered five-market confirmatory specification "
                "using block length 20, no model shows a supported "
                "current-minus-PIT Brier-score difference."
            ),
            "exploratory": (
                "Block-length-40 historical-probability evidence is a four-market "
                "available-market sensitivity and does not overturn the "
                "five-market confirmatory conclusion."
            ),
        },
        "base_v8_1_tag": BASE_V81_TAG,
        "base_v8_1_commit": BASE_V81_COMMIT,
    }
    write_json(STATUS_JSON_PATH, status)

    narrative = (
        "Under the preregistered five-market confirmatory specification using "
        "block length 20, no model shows a supported current-minus-PIT "
        "Brier-score difference."
    )
    robustness = (
        "In an exploratory block-length-40 robustness analysis, IDX30 was "
        "unavailable. The resulting four-market available-market macro supported "
        "a positive current-minus-PIT difference for the historical-probability "
        "baseline. This result is not a five-market result and does not overturn "
        "the confirmatory conclusion, but it shows that the "
        "historical-probability finding is sensitive to block length and market "
        "availability."
    )
    lines = [
        "# ASEAN Five-Market PIT Forecasting Status V8.1.1",
        "",
        f"**Release status:** `{RELEASE_STATUS}`",
        "",
        "This additive patch changes inference scope labels, robustness-family "
        "membership, calibration warnings, reports, and release metadata only. "
        "No forecasting input, prediction, or primary point estimate changed.",
        "",
        "## Confirmatory conclusion",
        "",
        narrative,
        "",
        *_markdown_table(
            confirmatory,
            [
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "adjusted_p_value",
                "result_classification",
                "contributing_market_count",
            ],
        ),
        "",
        "## Block-length-40 scope correction",
        "",
        "All block-40 `FIVE_MARKET_MACRO` rows are invalid and classified "
        "`INSUFFICIENT_INFORMATION` because IDX30 cannot support that block length.",
        "",
        robustness,
        "",
        *_markdown_table(
            available40,
            [
                "model_id",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "adjusted_p_value",
                "result_classification",
                "contributing_markets",
                "excluded_markets",
            ],
        ),
        "",
        "## Calibration warnings",
        "",
        f"- Date-level common-support rows: {warning_counts['date_rows']:,}",
        f"- Estimable under declared thresholds: {warning_counts['estimable_date_rows']:,}",
        f"- Flagged: {warning_counts['flagged_date_rows']:,}",
        f"- Separation warnings: {warning_counts['separation_warning_rows']:,}",
        f"- Extreme-coefficient warnings: {warning_counts['extreme_coefficient_warning_rows']:,}",
        f"- Non-finite coefficient warnings: {warning_counts['nonfinite_coefficient_warning_rows']:,}",
        "",
        "Original calibration coefficients are retained. Flagged coefficients are "
        "not eligible for unqualified aggregate calibration inference. Brier score "
        "remains primary and unchanged.",
    ]
    STATUS_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hardening_lines = [
        "# V8.1.1 Inference Hardening Report",
        "",
        narrative,
        "",
        robustness,
        "",
        "## Schema correction",
        "",
        "- Five-market macro validity now requires all five predefined markets.",
        "- Incomplete macro data emit an invalid five-market row and a separate "
        "exploratory available-market row.",
        "- Available-market hypotheses use their own BH families and can never "
        "enter the confirmatory family.",
        "- All block-20 confirmatory estimates, confidence intervals, raw p-values, "
        "BH-adjusted p-values, and classifications are unchanged.",
        "",
        "## Calibration",
        "",
        f"Estimability requires sample size >= {CALIBRATION_MIN_SAMPLE_SIZE}, "
        f"both class counts >= {CALIBRATION_MIN_CLASS_COUNT}, finite coefficients, "
        "no separation warning, and no extreme-coefficient warning.",
        "",
        f"Near-separation is flagged when a class count is below "
        f"{CALIBRATION_MIN_CLASS_COUNT} or an absolute coefficient exceeds "
        f"{CALIBRATION_SEPARATION_ABS_COEFFICIENT:g}. Extreme coefficients are "
        f"flagged above {CALIBRATION_EXTREME_ABS_COEFFICIENT:g}.",
        "",
        "The coefficients remain in the file for auditability. Any aggregate "
        "containing a flagged date cell is explicitly ineligible for unqualified "
        "aggregate calibration inference.",
    ]
    HARDENING_REPORT_PATH.write_text(
        "\n".join(hardening_lines) + "\n", encoding="utf-8"
    )
    return status


def write_reconciliation(
    *,
    integrity: Mapping[str, Any],
    source_bootstrap: pd.DataFrame,
    patched_bootstrap: pd.DataFrame,
    source_common: pd.DataFrame,
    patched_common: pd.DataFrame,
) -> None:
    source_confirmatory = source_bootstrap[
        source_bootstrap["support_specification"].eq("FULL_SUPPORT")
        & source_bootstrap["inference_level"].eq(
            "selected_five_market_macro"
        )
        & source_bootstrap["block_length"].eq(PRIMARY_BLOCK_LENGTH)
    ].sort_values("model_id")
    patched_confirmatory = patched_bootstrap[
        patched_bootstrap["support_specification"].eq("FULL_SUPPORT")
        & patched_bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")
        & patched_bootstrap["block_length"].eq(PRIMARY_BLOCK_LENGTH)
    ].sort_values("model_id")
    compare_columns = [
        "model_id",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "raw_p_value",
        "adjusted_p_value",
        "result_classification",
    ]
    pd.testing.assert_frame_equal(
        source_confirmatory[compare_columns].reset_index(drop=True),
        patched_confirmatory[compare_columns].reset_index(drop=True),
        check_dtype=False,
    )
    pd.testing.assert_frame_equal(
        source_common.reset_index(drop=True),
        patched_common[source_common.columns].reset_index(drop=True),
        check_dtype=False,
    )
    lines = [
        "# V8.1 to V8.1.1 Reconciliation",
        "",
        f"- Immutable V8.1 tag: `{BASE_V81_TAG}`",
        f"- Immutable V8.1 commit: `{BASE_V81_COMMIT}`",
        f"- V8.1.1 experiment: `{EXPERIMENT_ID}`",
        "- Forecasting inputs changed: **no**",
        "- Memberships/features/folds/models/seeds changed: **no**",
        "- Prediction probabilities changed: **no**",
        "- Primary block-20 point estimates or p-values changed: **no**",
        "",
        "## Prediction invariants",
        "",
        f"- Seed-level rows: {integrity['seed_level_rows']:,}",
        f"- Collapsed rows: {integrity['collapsed_rows']:,}",
        f"- Seed prediction SHA-256: `{integrity['prediction_hashes']['v8_seed_predictions']}`",
        f"- Collapsed prediction SHA-256: `{integrity['prediction_hashes']['v8_collapsed_predictions']}`",
        "",
        "## V8.1 source artifact checksums",
        "",
    ]
    for label, checksum in integrity["v8_1_file_hashes"].items():
        lines.append(f"- {label}: `{checksum}`")
    lines.extend(
        [
            "",
            "## Additive changes only",
            "",
            "- Added explicit market-count and macro-scope fields.",
            "- Split incomplete block-40 macros into invalid five-market and "
            "exploratory available-market rows.",
            "- Assigned available-market rows to separate exploratory BH families.",
            "- Added deterministic calibration estimability and instability flags.",
            "- Added V8.1.1 reports, audits, configuration, manifest, and envelope.",
            "",
            "Every pre-existing common-support column is value-identical to V8.1. "
            "Only new calibration columns were appended.",
        ]
    )
    RECONCILIATION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def artifact_class(path: Path) -> str:
    rel = relative(path)
    if rel.startswith("scripts/") or rel.startswith("tests/"):
        return "CODE_TEST"
    if "/config/" in rel:
        return "CONFIGURATION"
    if "/reports/" in rel or rel.startswith("reports/"):
        return "REPORT"
    if "/qa/" in rel:
        return "QUALITY_ASSURANCE"
    if "/manifests/" in rel:
        return "MANIFEST_CONTROL"
    return "BASE_RELEASE_CONTENT"


def release_content_paths() -> list[Path]:
    base_manifest = json.loads(BASE_RELEASE_MANIFEST.read_text(encoding="utf-8"))
    paths = {
        REPO_ROOT / record["artifact_relative_path"]
        for record in base_manifest["records"]
    }
    paths.update(
        {
            BASE_RELEASE_MANIFEST,
            BASE_RELEASE_MANIFEST_CSV,
            BASE_WORKSPACE_INVENTORY_JSON,
            BASE_WORKSPACE_INVENTORY_CSV,
            BASE_VALIDATION,
            BASE_RELEASE_ENVELOPE,
            SCRIPT_PATH,
            TEST_PATH,
            TEST_COMPAT_PATH,
            STATUS_MD_PATH,
            STATUS_JSON_PATH,
            RECONCILIATION_PATH,
            HARDENING_REPORT_PATH,
            CONFIG_PATH,
            PRIMARY_CSV_PATH,
            PRIMARY_PARQUET_PATH,
            COMMON_SUPPORT_PATH,
            FAMILY_REGISTRY_PATH,
            AVAILABILITY_AUDIT_PATH,
            CALIBRATION_AUDIT_PATH,
        }
    )
    missing = [relative(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("release files missing: " + " | ".join(sorted(missing)))
    if RELEASE_MANIFEST_PATH in paths or RELEASE_ENVELOPE_PATH in paths:
        raise RuntimeError("V8.1.1 control-plane self-reference")
    return sorted(paths, key=relative)


def build_release_manifest() -> dict[str, Any]:
    records = [
        {
            "artifact_relative_path": relative(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "artifact_class": artifact_class(path),
            "reproducibility_required": True,
            "manifest_version": MANIFEST_VERSION,
        }
        for path in release_content_paths()
    ]
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "release_status": RELEASE_STATUS,
        "manifest_version": MANIFEST_VERSION,
        "artifact_count": len(records),
        "total_release_bytes": sum(record["size_bytes"] for record in records),
        "records": records,
        "control_plane_exclusions": [
            relative(RELEASE_MANIFEST_PATH),
            relative(RELEASE_ENVELOPE_PATH),
            relative(VALIDATION_PATH),
        ],
        "control_plane_design": (
            "the release envelope authenticates the manifest and validation; "
            "neither self-hashes"
        ),
    }
    write_json(RELEASE_MANIFEST_PATH, payload)
    return payload


def _validate_manifest(manifest: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    seen: set[str] = set()
    for record in manifest["records"]:
        rel = str(record["artifact_relative_path"])
        if rel in seen:
            failures.append(f"duplicate manifest path: {rel}")
            continue
        seen.add(rel)
        path = REPO_ROOT / rel
        if not path.is_file():
            failures.append(f"missing release file: {rel}")
            continue
        if path.stat().st_size != int(record["size_bytes"]):
            failures.append(f"size mismatch: {rel}")
        if sha256_file(path) != str(record["sha256"]):
            failures.append(f"checksum mismatch: {rel}")
    if len(seen) != int(manifest["artifact_count"]):
        failures.append("artifact_count mismatch")
    if sum(int(record["size_bytes"]) for record in manifest["records"]) != int(
        manifest["total_release_bytes"]
    ):
        failures.append("total_release_bytes mismatch")
    return failures


def validate_release(
    *,
    require_envelope: bool,
    require_git_seal: bool,
    deep_predictions: bool,
) -> dict[str, Any]:
    failures: list[str] = []
    integrity = assert_base_immutable(deep_predictions=deep_predictions)
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    failures.extend(_validate_manifest(manifest))

    bootstrap = pd.read_parquet(PRIMARY_PARQUET_PATH)
    required_fields = {
        "required_market_count",
        "contributing_market_count",
        "contributing_markets",
        "excluded_markets",
        "all_required_markets_available",
        "macro_scope",
        "scope_validity_reason",
    }
    if not required_fields.issubset(bootstrap.columns):
        failures.append("bootstrap availability schema incomplete")
    block40 = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["block_length"].eq(ROBUSTNESS_BLOCK_LENGTH)
    ]
    five = block40[block40["macro_scope"].eq("FIVE_MARKET_MACRO")]
    available = block40[
        block40["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    ]
    if len(five) != 4 or five["valid"].astype(bool).any():
        failures.append("block-40 five-market rows are not all invalid")
    if not five["result_classification"].eq(
        "INSUFFICIENT_INFORMATION"
    ).all():
        failures.append("block-40 five-market classification incorrect")
    if len(available) != 4:
        failures.append("block-40 available-market row count incorrect")
    if not available["contributing_market_count"].eq(4).all():
        failures.append("block-40 available-market count incorrect")
    if not available["excluded_markets"].eq("IDX30").all():
        failures.append("IDX30 exclusion missing")
    if not available["confirmatory_or_exploratory"].eq("EXPLORATORY").all():
        failures.append("available-market inference is not exploratory")
    historical = available[
        available["model_id"].eq("historical_class_probability")
    ]
    if len(historical) != 1:
        failures.append("historical available-market result missing")
    else:
        expected = {
            "point_estimate": 0.00027648249533416733,
            "ci_lower": 0.00008693003484461146,
            "ci_upper": 0.0004634816708086555,
            "adjusted_p_value": 0.015198480151984802,
        }
        for column, value in expected.items():
            if not math.isclose(
                float(historical.iloc[0][column]),
                value,
                rel_tol=0,
                abs_tol=1e-15,
            ):
                failures.append(f"historical block-40 {column} changed")
        if historical.iloc[0]["result_classification"] != (
            "SUPPORTED_CURRENT_PIT_DIFFERENCE"
        ):
            failures.append("historical block-40 supported result lost")

    confirmatory = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["block_length"].eq(PRIMARY_BLOCK_LENGTH)
        & bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")
    ]
    if len(confirmatory) != 4:
        failures.append("confirmatory row count is not four")
    if not confirmatory["contributing_market_count"].eq(5).all():
        failures.append("confirmatory rows do not contain five markets")
    if not confirmatory["family_id"].eq(
        "CONFIRMATORY_MACRO_BRIER_BLOCK20"
    ).all():
        failures.append("confirmatory family drift")

    common = pd.read_csv(COMMON_SUPPORT_PATH)
    if not set(REQUIRED_CALIBRATION_FIELDS).issubset(common.columns):
        failures.append("common-support calibration schema incomplete")
    audit = pd.read_csv(CALIBRATION_AUDIT_PATH)
    if len(audit) != int(common["aggregation_level"].eq("date").sum()):
        failures.append("calibration audit/date-row reconciliation failed")
    family = pd.read_csv(FAMILY_REGISTRY_PATH)
    available_families = family[
        family["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    ]
    if available_families.empty:
        failures.append("available-market BH family missing")
    if not available_families[
        "confirmatory_or_exploratory"
    ].eq("EXPLORATORY").all():
        failures.append("available-market family not isolated as exploratory")

    envelope_result: dict[str, Any] = {"required": require_envelope}
    if require_envelope:
        if not RELEASE_ENVELOPE_PATH.is_file():
            failures.append("release envelope missing")
        else:
            envelope = json.loads(
                RELEASE_ENVELOPE_PATH.read_text(encoding="utf-8")
            )
            expected_hashes = {
                "release_manifest_sha256": sha256_file(RELEASE_MANIFEST_PATH),
                "validation_report_sha256": sha256_file(VALIDATION_PATH),
                "configuration_sha256": sha256_file(CONFIG_PATH),
            }
            for key, value in expected_hashes.items():
                if envelope.get(key) != value:
                    failures.append(f"envelope {key} mismatch")
            if envelope.get("git_tag") != FINAL_TAG:
                failures.append("envelope final tag mismatch")
            envelope_result = {
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
                    resolved_tag = git_output(
                        "rev-parse", f"{FINAL_TAG}^{{}}"
                    )
                    tag_type = git_output("cat-file", "-t", FINAL_TAG)
                    message = git_output(
                        "for-each-ref",
                        f"refs/tags/{FINAL_TAG}",
                        "--format=%(contents)",
                    )
                    if commit != envelope["git_commit_hash"]:
                        failures.append("content commit does not resolve exactly")
                    if tag_type != "tag":
                        failures.append("final tag is not annotated")
                    if resolved_tag != git_output("rev-parse", "HEAD"):
                        failures.append("final tag does not point to HEAD")
                    if RELEASE_STATUS not in message:
                        failures.append("final tag message mismatch")
                except (subprocess.CalledProcessError, KeyError) as exc:
                    failures.append(f"Git seal validation failed: {exc}")
    return {
        "experiment_id": EXPERIMENT_ID,
        "validation_status": "PASS" if not failures else "FAIL",
        "base_integrity": integrity,
        "release_manifest_records": int(manifest["artifact_count"]),
        "release_manifest_bytes": int(manifest["total_release_bytes"]),
        "bootstrap_rows": len(bootstrap),
        "calibration_audit_rows": len(audit),
        "envelope_validation": envelope_result,
        "failures": failures,
    }


def write_validation() -> dict[str, Any]:
    result = validate_release(
        require_envelope=False,
        require_git_seal=False,
        deep_predictions=True,
    )
    write_json(VALIDATION_PATH, result)
    if result["validation_status"] != "PASS":
        raise RuntimeError("V8.1.1 validation failed: " + " | ".join(result["failures"]))
    return result


def build_release_envelope(*, content_commit: str) -> dict[str, Any]:
    manifest = json.loads(RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_PATH.read_text(encoding="utf-8"))
    if validation["validation_status"] != "PASS":
        raise RuntimeError("cannot envelope a failing release")
    payload = {
        "experiment_id": EXPERIMENT_ID,
        "experiment_version": EXPERIMENT_VERSION,
        "release_status": RELEASE_STATUS,
        "release_manifest_path": relative(RELEASE_MANIFEST_PATH),
        "release_manifest_sha256": sha256_file(RELEASE_MANIFEST_PATH),
        "validation_report_path": relative(VALIDATION_PATH),
        "validation_report_sha256": sha256_file(VALIDATION_PATH),
        "configuration_path": relative(CONFIG_PATH),
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "git_commit_hash": content_commit,
        "git_commit_role": "V8.1.1_CONTENT_COMMIT_AUTHENTICATED_BY_ENVELOPE",
        "git_tag": FINAL_TAG,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_release_files": int(manifest["artifact_count"]),
        "total_release_bytes": int(manifest["total_release_bytes"]),
        "validation_status": validation["validation_status"],
        "self_reference_policy": (
            "envelope is excluded from the manifest it authenticates"
        ),
    }
    write_json(RELEASE_ENVELOPE_PATH, payload)
    return payload


def build_all() -> dict[str, Any]:
    integrity = assert_base_immutable(deep_predictions=True)
    source_bootstrap = pd.read_parquet(BASE_PRIMARY_PARQUET)
    patched_bootstrap, availability = patch_bootstrap_schema(source_bootstrap)
    registry = build_family_registry(patched_bootstrap)
    source_common = pd.read_csv(BASE_COMMON_SUPPORT)
    patched_common, calibration_audit = build_calibration_hardening(
        source_common
    )
    build_configuration()
    write_frame(PRIMARY_CSV_PATH, patched_bootstrap)
    write_frame(PRIMARY_PARQUET_PATH, patched_bootstrap)
    write_frame(COMMON_SUPPORT_PATH, patched_common)
    write_frame(FAMILY_REGISTRY_PATH, registry)
    write_frame(AVAILABILITY_AUDIT_PATH, availability)
    write_frame(CALIBRATION_AUDIT_PATH, calibration_audit)
    write_reports(
        integrity=integrity,
        bootstrap=patched_bootstrap,
        calibration_audit=calibration_audit,
    )
    write_reconciliation(
        integrity=integrity,
        source_bootstrap=source_bootstrap,
        patched_bootstrap=patched_bootstrap,
        source_common=source_common,
        patched_common=patched_common,
    )
    manifest = build_release_manifest()
    validation = write_validation()
    return {
        "status": RELEASE_STATUS,
        "bootstrap_rows": len(patched_bootstrap),
        "calibration_audit_rows": len(calibration_audit),
        "release_manifest_records": manifest["artifact_count"],
        "validation": validation["validation_status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--build", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--seal-envelope", action="store_true")
    parser.add_argument("--content-commit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.build:
        result = build_all()
    elif args.seal_envelope:
        if not args.content_commit:
            raise SystemExit("--seal-envelope requires --content-commit")
        result = build_release_envelope(content_commit=args.content_commit)
        result["status"] = result["validation_status"]
    else:
        result = validate_release(
            require_envelope=RELEASE_ENVELOPE_PATH.exists(),
            require_git_seal=(
                (REPO_ROOT / ".git").exists()
                and RELEASE_ENVELOPE_PATH.exists()
            ),
            deep_predictions=True,
        )
        result["status"] = result["validation_status"]
    print(json.dumps(json_safe(result), indent=2, sort_keys=True))
    return 0 if result["status"] in {"PASS", RELEASE_STATUS} else 1


if __name__ == "__main__":
    raise SystemExit(main())
