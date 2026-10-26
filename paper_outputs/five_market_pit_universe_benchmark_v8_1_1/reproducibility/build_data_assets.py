#!/usr/bin/env python3
"""Build publication data assets from the immutable V8.1.1 release.

This script is intentionally read-only with respect to the analytical release.
All writes are confined to:

    paper_outputs/five_market_pit_universe_benchmark_v8_1_1/

Run with the workspace Python environment that provides pandas, pyarrow,
matplotlib, seaborn, and openpyxl.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


WORKSPACE = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parents[1]
PIT = WORKSPACE / "research_data" / "point_in_time"
REPORTS = PIT / "reports"
PROCESSED = PIT / "processed"
QA = PIT / "qa"
MANIFESTS = PIT / "manifests"
CONFIG = PIT / "config"
RECONSTRUCTED = PIT / "reconstructed"
SOURCES = PIT / "sources"

TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
FIGURE_DATA = FIGURES / "figure_source_data"
AUDITS = OUTPUT / "audits"
OUT_MANIFESTS = OUTPUT / "manifests"

TITLE = (
    "Do Current-Constituent Universes Distort Equity Forecasting Evidence? "
    "A Point-in-Time Benchmark Across Five ASEAN Markets"
)
EXPERIMENT_ID = "ASEAN_FIVE_MARKET_PIT_CONTINUOUS_ROLLING_BENCHMARK_V8_1_1"
FINAL_TAG = "pit-five-market-v8.1.1-final"
FINAL_COMMIT = "8777babb77b0666604923d35d3f490462a040fd6"
RELEASE_STATUS = "FINAL_HARDENED_COMPLETE_FIVE_MARKET_INFERENCE_LABEL_PATCH"

MODEL_ORDER = [
    "historical_class_probability",
    "lightgbm",
    "logistic_regression",
    "random_forest",
]
MODEL_LABEL = {
    "historical_class_probability": "Historical probability",
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}
ARM_ORDER = [
    "POINT_IN_TIME",
    "CURRENT_2026_APPLIED_BACKWARD",
    "PIT_CURRENT_INTERSECTION_CONTROL",
]
ARM_LABEL = {
    "POINT_IN_TIME": "Point in time",
    "CURRENT_2026_APPLIED_BACKWARD": "Current 2026 backward",
    "PIT_CURRENT_INTERSECTION_CONTROL": "PIT-current intersection",
}
MARKET_ORDER = ["IDX30", "SET50", "VN30", "PSEI", "FBM_KLCI"]
MARKET_LABEL = {
    "IDX30": "Indonesia (IDX30)",
    "SET50": "Thailand (SET50)",
    "VN30": "Vietnam (VN30)",
    "PSEI": "Philippines (PSEI)",
    "FBM_KLCI": "Malaysia (FBM KLCI)",
}

INK = "#1B2733"
BLUE = "#2B6F8A"
BLUE_LIGHT = "#9CC6D8"
ORANGE = "#C46A36"
GOLD = "#B78A2B"
GREEN = "#52796F"
GREY = "#7C8793"
LIGHT_GREY = "#D8DEE5"
VERY_LIGHT = "#F3F5F7"


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, FIGURE_DATA, AUDITS, OUT_MANIFESTS):
        path.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def read_csv(name: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name, **kwargs)


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def verify_locked_release() -> dict[str, Any]:
    manifest_path = (
        MANIFESTS / "pit_multimarket_forecasting_v8_1_1_release_manifest.json"
    )
    envelope_path = MANIFESTS / "pit_v8_1_1_release_envelope.json"
    validation_path = QA / "pit_v8_1_1_artifact_validation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    for record in manifest["records"]:
        relative = record["artifact_relative_path"]
        path = WORKSPACE / Path(relative)
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        actual_hash = sha256(path) if exists else None
        records.append(
            {
                "artifact_relative_path": relative,
                "artifact_class": record["artifact_class"],
                "expected_size_bytes": int(record["size_bytes"]),
                "actual_size_bytes": actual_size,
                "expected_sha256": record["sha256"],
                "actual_sha256": actual_hash,
                "exists": exists,
                "size_match": exists and actual_size == int(record["size_bytes"]),
                "sha256_match": exists and actual_hash == record["sha256"],
            }
        )

    checks = pd.DataFrame(records)
    write_csv(checks, AUDITS / "LOCKED_RELEASE_CHECKSUM_RESULTS.csv")
    tag_commit = git_output("rev-list", "-n", "1", FINAL_TAG)
    head_commit = git_output("rev-parse", "HEAD")
    manifest_hash = sha256(manifest_path)
    validation_hash = sha256(validation_path)
    configuration_path = WORKSPACE / envelope["configuration_path"]
    configuration_hash = sha256(configuration_path)
    failures = checks[
        ~(checks["exists"] & checks["size_match"] & checks["sha256_match"])
    ]
    summary = {
        "generated_at_utc": utc_now(),
        "experiment_id": EXPERIMENT_ID,
        "release_status": RELEASE_STATUS,
        "final_tag": FINAL_TAG,
        "expected_final_commit": FINAL_COMMIT,
        "tag_commit": tag_commit,
        "head_commit_at_build": head_commit,
        "manifest_record_count": int(len(checks)),
        "manifest_expected_record_count": int(manifest["artifact_count"]),
        "manifest_expected_total_bytes": int(manifest["total_release_bytes"]),
        "manifest_actual_total_bytes": int(
            pd.to_numeric(checks["actual_size_bytes"], errors="coerce").sum()
        ),
        "manifest_sha256_expected": envelope["release_manifest_sha256"],
        "manifest_sha256_actual": manifest_hash,
        "validation_sha256_expected": envelope["validation_report_sha256"],
        "validation_sha256_actual": validation_hash,
        "configuration_sha256_expected": envelope["configuration_sha256"],
        "configuration_sha256_actual": configuration_hash,
        "release_validation_status": validation["validation_status"],
        "record_failure_count": int(len(failures)),
        "tag_matches_expected_commit": tag_commit == FINAL_COMMIT,
        "manifest_hash_matches": manifest_hash
        == envelope["release_manifest_sha256"],
        "validation_hash_matches": validation_hash
        == envelope["validation_report_sha256"],
        "configuration_hash_matches": configuration_hash
        == envelope["configuration_sha256"],
        "all_record_hashes_match": failures.empty,
        "status": "PASS"
        if (
            failures.empty
            and tag_commit == FINAL_COMMIT
            and manifest_hash == envelope["release_manifest_sha256"]
            and validation_hash == envelope["validation_report_sha256"]
            and configuration_hash == envelope["configuration_sha256"]
            and validation["validation_status"] == "PASS"
        )
        else "FAIL",
        "note": (
            "The V8.1.1 envelope authenticates the content commit and is itself "
            "added by the final tagged commit; this is the declared control-plane design."
        ),
    }
    (OUT_MANIFESTS / "locked_release_baseline.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Locked V8.1.1 release integrity baseline",
        "",
        f"- Status: **{summary['status']}**",
        f"- Experiment: `{EXPERIMENT_ID}`",
        f"- Release status: `{RELEASE_STATUS}`",
        f"- Tag: `{FINAL_TAG}`",
        f"- Tag commit: `{tag_commit}`",
        f"- Required final commit: `{FINAL_COMMIT}`",
        f"- Manifest records checked: {len(checks)}",
        f"- Record failures: {len(failures)}",
        f"- Manifest SHA-256 match: {summary['manifest_hash_matches']}",
        f"- Validation SHA-256 match: {summary['validation_hash_matches']}",
        f"- Configuration SHA-256 match: {summary['configuration_hash_matches']}",
        f"- Release validation artifact: `{validation['validation_status']}`",
        "",
        "The release manifest, envelope, configuration, validation record, and every "
        "manifest-listed artifact were read only. This build did not invoke any "
        "analytical runner or validation command capable of rewriting release files.",
    ]
    (AUDITS / "LOCKED_RELEASE_INTEGRITY_BASELINE.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if summary["status"] != "PASS":
        raise RuntimeError("Locked V8.1.1 release integrity verification failed")
    return summary


def build_table_01() -> pd.DataFrame:
    market = read_csv("forecasting_ready_market_inventory_v8.csv")
    intervals = read_csv("forecasting_ready_interval_inventory_v8.csv")
    history = (
        intervals.groupby(["market", "index_id"], as_index=False)
        .agg(
            membership_start=("interval_start", "min"),
            membership_end_exclusive=("interval_end_exclusive", "max"),
        )
    )
    frame = market.merge(history, on=["market", "index_id"], how="left")
    frame = frame[
        [
            "market",
            "index_id",
            "membership_start",
            "membership_end_exclusive",
            "pit_intervals",
            "data_ready_intervals",
            "valid_continuous_folds",
            "oos_trading_dates",
            "seed_collapsed_predictions",
            "valid_all_arm_model_cells",
            "invalid_cells",
        ]
    ].copy()
    frame.columns = [
        "Market",
        "Index",
        "Membership history start",
        "History end (exclusive)",
        "PIT intervals",
        "Ready intervals",
        "Valid folds",
        "OOS dates",
        "Collapsed predictions",
        "Valid arm-model cells",
        "Invalid cells",
    ]
    frame["Index"] = pd.Categorical(
        frame["Index"], categories=MARKET_ORDER, ordered=True
    )
    return frame.sort_values("Index").reset_index(drop=True)


def build_table_02() -> pd.DataFrame:
    rows = [
        (
            "Universe arms",
            "POINT_IN_TIME; CURRENT_2026_APPLIED_BACKWARD; PIT_CURRENT_INTERSECTION_CONTROL",
            "Each arm is trained separately; exact common support is a diagnostic, not a retraining design.",
        ),
        (
            "Target",
            "Next-session open-to-close direction",
            "target_close / target_open - 1; class 1 only when return > 0; maximum next-observation gap 14 calendar days.",
        ),
        (
            "Features",
            "Nine lagged raw-OHLC and same-market features",
            "Open-close, close-close, overnight gap, high-low range, 5-session volatility, momentum and downside rate, plus lagged market return and breadth.",
        ),
        (
            "Models",
            "Historical probability; logistic regression; random forest; LightGBM",
            "One information-set baseline, one regularized linear classifier, one bagged tree model, and one boosted tree model.",
        ),
        (
            "Seeds",
            "11; 23; 37; 53; 71",
            "Every collapsed prediction contains exactly five seeds.",
        ),
        (
            "Rolling origin",
            "Continuous expanding same-market history",
            "60 initial training dates; 20 calibration dates; 20 test dates; 20-date step; interval boundaries do not reset history.",
        ),
        (
            "Validity gates",
            "At least 2 folds, 40 OOS dates and 800 OOS predictions",
            "Both classes in train and test; matured labels; unique target keys; no overlap or leakage.",
        ),
        (
            "Metrics",
            "Primary Brier score; secondary log loss and balanced accuracy",
            "Calibration is diagnostic subject to estimability gates; prediction coverage is operational.",
        ),
        (
            "Aggregation",
            "Security → date → interval → market → five-market macro",
            "Equal weight at every stage; the primary macro gives each market equal weight.",
        ),
        (
            "Inference",
            "Paired moving-block bootstrap",
            "10,000 primary replications; seed 271828; block 20 confirmatory; blocks 10 and 40 exploratory; blocks confined to intervals.",
        ),
        (
            "Multiplicity",
            "Benjamini-Hochberg within declared families",
            "Confirmatory family contains four full-support five-market block-20 Brier comparisons; CI and adjusted-p joint gate.",
        ),
    ]
    return pd.DataFrame(rows, columns=["Element", "Locked specification", "Implementation"])


def build_table_03() -> pd.DataFrame:
    frame = read_csv("forecasting_ready_market_inventory_v8.csv")
    frame = frame[
        [
            "index_id",
            "valid_continuous_folds",
            "oos_trading_dates",
            "seed_collapsed_predictions",
            "valid_all_arm_model_cells",
            "invalid_cells",
        ]
    ].copy()
    frame.columns = [
        "Market",
        "Valid folds",
        "OOS dates",
        "Collapsed predictions",
        "Valid cells",
        "Invalid cells",
    ]
    frame["Market"] = pd.Categorical(
        frame["Market"], categories=MARKET_ORDER, ordered=True
    )
    return frame.sort_values("Market").reset_index(drop=True)


def build_table_04() -> pd.DataFrame:
    macro = read_csv("pit_multimarket_equal_market_macro_v8.csv")
    frame = macro[
        [
            "model_id",
            "universe_semantics",
            "brier_score",
            "log_loss",
            "balanced_accuracy",
            "prediction_count",
        ]
    ].copy()
    frame["Model"] = frame["model_id"].map(MODEL_LABEL)
    frame["Universe arm"] = frame["universe_semantics"].map(ARM_LABEL)
    frame["model_order"] = frame["model_id"].map(
        {name: i for i, name in enumerate(MODEL_ORDER)}
    )
    frame["arm_order"] = frame["universe_semantics"].map(
        {name: i for i, name in enumerate(ARM_ORDER)}
    )
    frame = frame.sort_values(["model_order", "arm_order"])
    return frame[
        [
            "Model",
            "Universe arm",
            "brier_score",
            "log_loss",
            "balanced_accuracy",
            "prediction_count",
        ]
    ].rename(
        columns={
            "brier_score": "Brier score",
            "log_loss": "Log loss",
            "balanced_accuracy": "Balanced accuracy",
            "prediction_count": "Collapsed predictions",
        }
    )


def confirmatory_rows() -> pd.DataFrame:
    primary = read_csv("pit_multimarket_primary_bootstrap_10000_v8_1_1.csv")
    frame = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["macro_scope"].eq("FIVE_MARKET_MACRO")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(20)
        & primary["valid"].astype(bool)
    ].copy()
    frame["Model"] = frame["model_id"].map(MODEL_LABEL)
    frame["model_order"] = frame["model_id"].map(
        {name: i for i, name in enumerate(MODEL_ORDER)}
    )
    return frame.sort_values("model_order").reset_index(drop=True)


def build_table_05() -> pd.DataFrame:
    frame = confirmatory_rows()
    return frame[
        [
            "Model",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "raw_p_value",
            "adjusted_p_value",
            "result_classification",
        ]
    ].rename(
        columns={
            "point_estimate": "Current minus PIT Brier",
            "ci_lower": "95% CI lower",
            "ci_upper": "95% CI upper",
            "raw_p_value": "Raw p",
            "adjusted_p_value": "BH-adjusted p",
            "result_classification": "Classification",
        }
    )


def support_macro_rows() -> pd.DataFrame:
    diagnostic = read_csv("pit_multimarket_support_composition_diagnostic_v8_1.csv")
    frame = diagnostic[
        diagnostic["aggregation_level"].eq("selected_market_macro")
    ].copy()
    frame["Model"] = frame["model_id"].map(MODEL_LABEL)
    frame["model_order"] = frame["model_id"].map(
        {name: i for i, name in enumerate(MODEL_ORDER)}
    )
    return frame.sort_values("model_order").reset_index(drop=True)


def build_table_06() -> pd.DataFrame:
    frame = support_macro_rows()
    return frame[
        [
            "Model",
            "delta_brier_score_current_minus_pit_full_support",
            "delta_brier_score_current_minus_pit_common_support",
            "descriptive_support_composition_component_brier_score",
            "common_support_count",
        ]
    ].rename(
        columns={
            "delta_brier_score_current_minus_pit_full_support": "Full-support delta",
            "delta_brier_score_current_minus_pit_common_support": "Common-support delta",
            "descriptive_support_composition_component_brier_score": "Arithmetic support/composition component",
            "common_support_count": "Matched pairs per model",
        }
    )


def build_table_07() -> pd.DataFrame:
    perf = read_csv("pit_multimarket_performance_by_market_v8.csv")
    base = perf[perf["model_id"].eq("historical_class_probability")].copy()
    pivot = base.pivot(
        index="index_id", columns="universe_semantics", values="prediction_coverage"
    ).reset_index()
    pivot["Current minus PIT"] = (
        pivot["CURRENT_2026_APPLIED_BACKWARD"] - pivot["POINT_IN_TIME"]
    )
    pivot["Scope"] = pivot["index_id"].map(MARKET_LABEL)
    macro = read_csv("pit_multimarket_equal_market_macro_v8.csv")
    macro = macro[macro["model_id"].eq("historical_class_probability")].set_index(
        "universe_semantics"
    )
    macro_row = {
        "index_id": "FIVE_MARKET_MACRO",
        "POINT_IN_TIME": macro.loc["POINT_IN_TIME", "prediction_coverage"],
        "CURRENT_2026_APPLIED_BACKWARD": macro.loc[
            "CURRENT_2026_APPLIED_BACKWARD", "prediction_coverage"
        ],
        "PIT_CURRENT_INTERSECTION_CONTROL": macro.loc[
            "PIT_CURRENT_INTERSECTION_CONTROL", "prediction_coverage"
        ],
        "Current minus PIT": macro.loc[
            "CURRENT_2026_APPLIED_BACKWARD", "prediction_coverage"
        ]
        - macro.loc["POINT_IN_TIME", "prediction_coverage"],
        "Scope": "Equal-market five-market macro",
    }
    pivot["order"] = pivot["index_id"].map(
        {name: i for i, name in enumerate(MARKET_ORDER)}
    )
    pivot = pivot.sort_values("order")
    pivot = pd.concat([pivot, pd.DataFrame([macro_row])], ignore_index=True)
    return pivot[
        [
            "Scope",
            "POINT_IN_TIME",
            "CURRENT_2026_APPLIED_BACKWARD",
            "PIT_CURRENT_INTERSECTION_CONTROL",
            "Current minus PIT",
        ]
    ].rename(
        columns={
            "POINT_IN_TIME": "PIT coverage",
            "CURRENT_2026_APPLIED_BACKWARD": "Current-backward coverage",
            "PIT_CURRENT_INTERSECTION_CONTROL": "Intersection-control coverage",
        }
    )


def _fmt_range(values: pd.Series) -> str:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return "not available"
    return f"[{numeric.min():+.8f}, {numeric.max():+.8f}]"


def build_table_08() -> pd.DataFrame:
    primary = read_csv("pit_multimarket_primary_bootstrap_10000_v8_1_1.csv")
    block10 = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["macro_scope"].eq("FIVE_MARKET_MACRO")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(10)
        & primary["valid"].astype(bool)
    ]
    block40_five = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["macro_scope"].eq("FIVE_MARKET_MACRO")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(40)
    ]
    block40_available = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(40)
        & primary["valid"].astype(bool)
    ]
    agg = read_csv("pit_multimarket_aggregation_robustness_v8_1.csv")
    agg = agg[agg["metric"].eq("brier_score")]
    lomo = read_csv("pit_multimarket_leave_one_market_out_v8_1.csv")
    lomo = lomo[lomo["metric"].eq("brier_score")]
    loio = read_csv("pit_multimarket_leave_one_interval_out_v8_1.csv")
    loio = loio[loio["metric"].eq("brier_score")]

    rows: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        b10 = block10[block10["model_id"].eq(model)].iloc[0]
        b40 = block40_available[block40_available["model_id"].eq(model)].iloc[0]
        rows.append(
            {
                "Analysis": "Block length 10",
                "Scope": "Five-market full support",
                "Model": MODEL_LABEL[model],
                "Estimate or range": f"{b10['point_estimate']:+.8f}",
                "Inference/status": b10["result_classification"],
                "Interpretive boundary": "Exploratory block-length sensitivity.",
            }
        )
        rows.append(
            {
                "Analysis": "Block length 40",
                "Scope": "Four-market available-market macro",
                "Model": MODEL_LABEL[model],
                "Estimate or range": f"{b40['point_estimate']:+.8f}",
                "Inference/status": b40["result_classification"],
                "Interpretive boundary": (
                    "IDX30 unavailable; not a five-market result. Random-forest "
                    "CI excludes zero but the BH joint gate is not passed."
                    if model == "random_forest"
                    else "IDX30 unavailable; not a five-market result."
                ),
            }
        )
        rows.append(
            {
                "Analysis": "Alternative aggregation",
                "Scope": "Reported full-support rules",
                "Model": MODEL_LABEL[model],
                "Estimate or range": _fmt_range(
                    agg[agg["model_id"].eq(model)][
                        "point_estimate_current_minus_pit"
                    ]
                ),
                "Inference/status": "DESCRIPTIVE_SENSITIVITY",
                "Interpretive boundary": "Point-estimate range; no family-wide significance claim.",
            }
        )
        rows.append(
            {
                "Analysis": "Leave one market out",
                "Scope": "Four markets per omission",
                "Model": MODEL_LABEL[model],
                "Estimate or range": _fmt_range(
                    lomo[lomo["model_id"].eq(model)][
                        "point_estimate_current_minus_pit"
                    ]
                ),
                "Inference/status": "DESCRIPTIVE_SENSITIVITY",
                "Interpretive boundary": "Country omissions do not identify country effects.",
            }
        )
        rows.append(
            {
                "Analysis": "Leave one interval out",
                "Scope": "One ready interval omitted",
                "Model": MODEL_LABEL[model],
                "Estimate or range": _fmt_range(
                    loio[loio["model_id"].eq(model)][
                        "point_estimate_current_minus_pit"
                    ]
                ),
                "Inference/status": "DESCRIPTIVE_SENSITIVITY",
                "Interpretive boundary": "Point-estimate range only.",
            }
        )
    rows.insert(
        0,
        {
            "Analysis": "Block length 40",
            "Scope": "Predefined five-market macro",
            "Model": "All four models",
            "Estimate or range": "not estimable",
            "Inference/status": (
                "INSUFFICIENT_INFORMATION"
                if len(block40_five) == 4
                and (~block40_five["valid"].astype(bool)).all()
                else "AUDIT_FAILURE"
            ),
            "Interpretive boundary": "IDX30 cannot support block 40; no five-market estimate exists.",
        },
    )
    return pd.DataFrame(rows)


def build_main_tables() -> dict[str, pd.DataFrame]:
    tables = {
        "table_01_market_scope.csv": build_table_01(),
        "table_02_protocol.csv": build_table_02(),
        "table_03_execution.csv": build_table_03(),
        "table_04_absolute_brier.csv": build_table_04(),
        "table_05_confirmatory_inference.csv": build_table_05(),
        "table_06_common_support.csv": build_table_06(),
        "table_07_prediction_coverage.csv": build_table_07(),
        "table_08_robustness.csv": build_table_08(),
    }
    for filename, frame in tables.items():
        write_csv(frame, TABLES / filename)
    return tables


def _save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=350, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def figure_01() -> pd.DataFrame:
    source = pd.DataFrame(
        [
            {
                "arm": "POINT_IN_TIME",
                "label": "Historical PIT membership",
                "membership_rule": "Membership effective on each historical date",
                "training": "Separately trained within same market and arm",
                "role": "Primary reference arm",
            },
            {
                "arm": "CURRENT_2026_APPLIED_BACKWARD",
                "label": "Current-2026 membership applied backward",
                "membership_rule": "Future 2026 roster determines historical security set",
                "training": "Separately trained within same market and arm",
                "role": "Primary comparison arm",
            },
            {
                "arm": "PIT_CURRENT_INTERSECTION_CONTROL",
                "label": "PIT-current intersection control",
                "membership_rule": "Historical PIT set intersected with current roster",
                "training": "Separately trained within same market and arm",
                "role": "Control arm",
            },
        ]
    )
    write_csv(source, FIGURE_DATA / "figure_01_research_design.csv")
    fig, ax = plt.subplots(figsize=(11, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.text(
        0.3,
        7.55,
        "Information-set construction and evaluation design",
        fontsize=16,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.3,
        7.12,
        "All arms use the same target, features, model registry, seeds, and rolling-origin rules.",
        fontsize=10.5,
        color=GREY,
    )
    colors = [BLUE, ORANGE, GREEN]
    y_values = [5.8, 3.9, 2.0]
    display_labels = [
        "Historical PIT\nmembership",
        "Current-2026 membership\napplied backward",
        "PIT-current\nintersection control",
    ]
    for row, display_label, color, y in zip(
        source.to_dict("records"), display_labels, colors, y_values
    ):
        left = FancyBboxPatch(
            (0.45, y - 0.55),
            3.35,
            1.1,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor=color,
            alpha=0.95,
        )
        ax.add_patch(left)
        ax.text(
            2.125,
            y,
            display_label,
            ha="center",
            va="center",
            color="white",
            fontsize=9.8,
            fontweight="bold",
            wrap=True,
        )
        arrow = FancyArrowPatch(
            (3.85, y),
            (4.55, y),
            arrowstyle="-|>",
            mutation_scale=16,
            color=GREY,
            linewidth=1.5,
        )
        ax.add_patch(arrow)
        mid = FancyBboxPatch(
            (4.6, y - 0.55),
            3.1,
            1.1,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=VERY_LIGHT,
            edgecolor=color,
            linewidth=1.4,
        )
        ax.add_patch(mid)
        ax.text(
            6.15,
            y,
            "Continuous same-market\nrolling-origin forecasts",
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
        )
        arrow2 = FancyArrowPatch(
            (7.75, y),
            (8.45, y),
            arrowstyle="-|>",
            mutation_scale=16,
            color=GREY,
            linewidth=1.5,
        )
        ax.add_patch(arrow2)
        right = FancyBboxPatch(
            (8.5, y - 0.55),
            3.0,
            1.1,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
        )
        ax.add_patch(right)
        ax.text(
            10.0,
            y,
            "Out-of-sample loss\nand coverage",
            ha="center",
            va="center",
            fontsize=10,
            color=INK,
        )
    ax.text(
        0.45,
        0.55,
        "Primary contrast: current minus PIT. Full support is confirmatory; exact common support is diagnostic; "
        "the intersection control is separately trained.",
        fontsize=10,
        color=INK,
        bbox={"facecolor": "#FFF7E8", "edgecolor": GOLD, "boxstyle": "round,pad=0.4"},
    )
    _save_figure(fig, "figure_01_research_design")
    return source


def figure_02() -> pd.DataFrame:
    source = read_csv(
        "forecasting_ready_interval_inventory_v8.csv",
        parse_dates=["interval_start", "interval_end_exclusive"],
    )
    source["eligible_for_forecasting"] = source["eligible_for_forecasting"].astype(bool)
    source = source[
        [
            "market",
            "index_id",
            "interval_id",
            "interval_start",
            "interval_end_exclusive",
            "trading_dates",
            "pit_security_count",
            "price_coverage",
            "target_pair_coverage",
            "eligible_for_forecasting",
            "blocking_reasons",
        ]
    ].copy()
    write_csv(source, FIGURE_DATA / "figure_02_interval_timeline.csv")
    fig, ax = plt.subplots(figsize=(12, 5.7))
    y_map = {index_id: len(MARKET_ORDER) - i for i, index_id in enumerate(MARKET_ORDER)}
    for row in source.itertuples():
        y = y_map[row.index_id]
        color = BLUE if row.eligible_for_forecasting else LIGHT_GREY
        alpha = 1.0 if row.eligible_for_forecasting else 0.75
        ax.plot(
            [row.interval_start, row.interval_end_exclusive],
            [y, y],
            color=color,
            alpha=alpha,
            linewidth=6 if row.eligible_for_forecasting else 3,
            solid_capstyle="butt",
        )
    ax.set_yticks(
        [y_map[x] for x in MARKET_ORDER],
        [MARKET_LABEL[x] for x in MARKET_ORDER],
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(
        "PIT membership intervals and forecasting-ready support",
        loc="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    ax.set_xlabel("Calendar time")
    ax.text(
        0.0,
        -0.21,
        "Blue = forecasting-ready interval; grey = reconstructed PIT interval that did not pass the locked forecasting gates.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=GREY,
    )
    _save_figure(fig, "figure_02_interval_timeline")
    return source


def figure_03() -> pd.DataFrame:
    frame = confirmatory_rows()
    source = frame[
        [
            "model_id",
            "Model",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "adjusted_p_value",
            "result_classification",
        ]
    ].copy()
    write_csv(source, FIGURE_DATA / "figure_03_confirmatory_forest.csv")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(source))[::-1]
    x = source["point_estimate"].to_numpy()
    lo = source["ci_lower"].to_numpy()
    hi = source["ci_upper"].to_numpy()
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([x - lo, hi - x]),
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        capsize=4,
        markersize=7,
        linewidth=1.8,
    )
    ax.axvline(0, color=INK, linewidth=1, linestyle="--")
    ax.set_yticks(y, source["Model"])
    ax.set_xlabel("Brier-score difference (current minus PIT)")
    ax.set_title(
        "Confirmatory five-market Brier-score differences (block length 20)",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(
        0,
        -0.2,
        "Negative values indicate lower loss for the current-universe arm; positive values indicate higher loss.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=GREY,
    )
    _save_figure(fig, "figure_03_confirmatory_forest")
    return source


def figure_04() -> pd.DataFrame:
    frame = support_macro_rows()
    source = frame[
        [
            "model_id",
            "Model",
            "delta_brier_score_current_minus_pit_full_support",
            "delta_brier_score_current_minus_pit_common_support",
            "descriptive_support_composition_component_brier_score",
            "common_support_count",
        ]
    ].copy()
    write_csv(source, FIGURE_DATA / "figure_04_common_support.csv")
    x = np.arange(len(source))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.bar(
        x - width / 2,
        source["delta_brier_score_current_minus_pit_full_support"],
        width,
        color=BLUE,
        label="Full support (primary)",
    )
    ax.bar(
        x + width / 2,
        source["delta_brier_score_current_minus_pit_common_support"],
        width,
        color=ORANGE,
        label="Exact common support (diagnostic)",
    )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x, source["Model"], rotation=12, ha="right")
    ax.set_ylabel("Brier-score difference (current minus PIT)")
    ax.set_title(
        "Full-support and exact common-support Brier comparisons",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.legend(frameon=False)
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(
        0,
        -0.25,
        "Negative values indicate lower loss for the current-universe arm. The gap between bars is descriptive, not causal.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=GREY,
    )
    _save_figure(fig, "figure_04_common_support")
    return source


def figure_05() -> pd.DataFrame:
    primary = read_csv("pit_multimarket_primary_bootstrap_10000_v8_1_1.csv")
    source = primary[
        primary["support_specification"].eq("FULL_SUPPORT")
        & primary["macro_scope"].eq("MARKET_LEVEL")
        & primary["metric"].eq("brier_score")
        & primary["block_length"].eq(20)
        & primary["valid"].astype(bool)
    ][
        [
            "index_id",
            "model_id",
            "point_estimate",
            "ci_lower",
            "ci_upper",
            "adjusted_p_value",
            "result_classification",
        ]
    ].copy()
    source["Market"] = source["index_id"].map(MARKET_LABEL)
    source["Model"] = source["model_id"].map(MODEL_LABEL)
    write_csv(source, FIGURE_DATA / "figure_05_market_model_heatmap.csv")
    matrix = (
        source.pivot(index="index_id", columns="model_id", values="point_estimate")
        .reindex(index=MARKET_ORDER, columns=MODEL_ORDER)
        .rename(index=MARKET_LABEL, columns=MODEL_LABEL)
    )
    limit = float(np.nanmax(np.abs(matrix.to_numpy())))
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="+.5f",
        cmap="vlag",
        center=0,
        vmin=-limit,
        vmax=limit,
        linewidths=0.7,
        linecolor="white",
        cbar_kws={"label": "Current minus PIT Brier score"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Market-level current-minus-PIT Brier-score differences",
        loc="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    ax.tick_params(axis="x", rotation=15)
    ax.tick_params(axis="y", rotation=0)
    ax.text(
        0,
        -0.18,
        "Negative values indicate lower loss for the current-universe arm; positive values indicate higher loss. "
        "Country patterns are descriptive.",
        transform=ax.transAxes,
        fontsize=9.2,
        color=GREY,
    )
    _save_figure(fig, "figure_05_market_model_heatmap")
    return source


def figure_06() -> pd.DataFrame:
    agg = read_csv("pit_multimarket_aggregation_robustness_v8_1.csv")
    agg = agg[agg["metric"].eq("brier_score")][
        [
            "aggregation_specification",
            "model_id",
            "point_estimate_current_minus_pit",
            "primary",
        ]
    ].copy()
    agg["family"] = "Aggregation"
    agg["specification"] = agg["aggregation_specification"]
    agg["excluded_market"] = ""
    lomo = read_csv("pit_multimarket_leave_one_market_out_v8_1.csv")
    lomo = lomo[lomo["metric"].eq("brier_score")][
        ["excluded_market", "model_id", "point_estimate_current_minus_pit"]
    ].copy()
    lomo["family"] = "Leave one market out"
    lomo["specification"] = "Exclude " + lomo["excluded_market"].astype(str)
    lomo["primary"] = False
    source = pd.concat(
        [
            agg[
                [
                    "family",
                    "specification",
                    "excluded_market",
                    "model_id",
                    "point_estimate_current_minus_pit",
                    "primary",
                ]
            ],
            lomo[
                [
                    "family",
                    "specification",
                    "excluded_market",
                    "model_id",
                    "point_estimate_current_minus_pit",
                    "primary",
                ]
            ],
        ],
        ignore_index=True,
    )
    source["Model"] = source["model_id"].map(MODEL_LABEL)
    write_csv(source, FIGURE_DATA / "figure_06_robustness.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), sharex=True)
    for ax, model in zip(axes.flat, MODEL_ORDER):
        data = source[source["model_id"].eq(model)].copy()
        agg_data = data[data["family"].eq("Aggregation")]
        lomo_data = data[data["family"].eq("Leave one market out")]
        ax.scatter(
            agg_data["point_estimate_current_minus_pit"],
            np.full(len(agg_data), 1.0),
            color=BLUE,
            alpha=0.8,
            label="Aggregation rules",
        )
        ax.scatter(
            lomo_data["point_estimate_current_minus_pit"],
            np.full(len(lomo_data), 0.0),
            color=ORANGE,
            alpha=0.8,
            marker="D",
            label="Leave-one-market-out",
        )
        primary = agg_data[agg_data["primary"].astype(bool)]
        if len(primary):
            ax.scatter(
                primary["point_estimate_current_minus_pit"],
                np.full(len(primary), 1.0),
                facecolors="white",
                edgecolors=INK,
                s=110,
                linewidths=1.5,
                zorder=4,
                label="Primary",
            )
        ax.axvline(0, color=INK, linewidth=1, linestyle="--")
        ax.set_yticks([0, 1], ["Leave-one-market-out", "Aggregation"])
        ax.set_title(MODEL_LABEL[model], loc="left", fontsize=11.5, fontweight="bold")
        ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(
        unique.values(),
        unique.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Brier-score differences across aggregation and market-omission specifications",
        x=0.08,
        y=1.03,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.supxlabel("Brier-score difference (current minus PIT)")
    fig.text(
        0.08,
        -0.01,
        "Negative values indicate lower loss for the current-universe arm. Points are sensitivity estimates; "
        "sign stability is not statistical support.",
        fontsize=9.3,
        color=GREY,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    _save_figure(fig, "figure_06_robustness")
    return source


def build_figures() -> dict[str, pd.DataFrame]:
    return {
        "figure_01": figure_01(),
        "figure_02": figure_02(),
        "figure_03": figure_03(),
        "figure_04": figure_04(),
        "figure_05": figure_05(),
        "figure_06": figure_06(),
    }


def build_supplementary_workbook(
    main_tables: dict[str, pd.DataFrame],
) -> None:
    market_sources = pd.DataFrame(
        [
            ("IDX30", "Historical membership evidence", "Official index-provider documents", "Accepted under the locked evidence hierarchy"),
            ("SET50", "Historical membership evidence", "Stock Exchange of Thailand archive and methodology", "Accepted under the locked evidence hierarchy"),
            ("VN30", "Historical membership evidence", "HOSE official exchange documents", "Accepted under the locked evidence hierarchy"),
            ("PSEI", "Historical membership evidence", "Philippine Stock Exchange documents", "Accepted with documented reconstruction limits"),
            ("FBM KLCI", "Historical membership evidence", "FTSE Russell, LSEG and Bursa records with tiered anchors", "PASS_WITH_DOCUMENTED_LIMITATIONS"),
            ("All markets", "Raw price semantics", "Provider-native raw OHLC from release-registered local and public-source lanes", "No adjusted close, volume dependency or imputation"),
        ],
        columns=["Index", "Evidence object", "Source class", "Status/constraint"],
    )
    fbm_validation = json.loads(
        (QA / "fbm_klci_membership_validation_v8.json").read_text(encoding="utf-8")
    )
    malaysia_summary = pd.DataFrame(
        [
            ("Accepted anchor", "2017-12-31"),
            ("Independent endpoint anchor", "2019-12-23"),
            (
                "Anchor-to-anchor exact reconstructed set equality",
                str(fbm_validation["anchor_to_anchor_exact_set_equality"]),
            ),
            (
                "Unknown membership dates in evaluated windows",
                fbm_validation["unknown_membership_dates_inside_evaluated_windows"],
            ),
            ("No current-constituent backcast", str(fbm_validation["no_current_constituent_backcast"])),
            ("Tier 1 sources", fbm_validation["source_tier_counts"]["1"]),
            ("Tier 2 sources", fbm_validation["source_tier_counts"]["2"]),
            ("Tier 3 sources", fbm_validation["source_tier_counts"]["3"]),
        ],
        columns=["Field", "Value"],
    )
    intervals = read_csv("forecasting_ready_interval_inventory_v8.csv")
    rejection = pd.read_parquet(QA / "pit_multimarket_observation_rejections_v8.parquet")
    rejection_summary = (
        rejection.groupby(
            ["index_id", "universe_semantics", "valid_for_model", "rejection_reason"],
            dropna=False,
            as_index=False,
        )
        .agg(rows=("security_id", "size"), securities=("security_id", "nunique"))
        .sort_values(["index_id", "universe_semantics", "rows"], ascending=[True, True, False])
    )
    folds = pd.read_parquet(PROCESSED / "pit_multimarket_fold_calendar_v8.parquet")
    primary = read_csv("pit_multimarket_primary_bootstrap_10000_v8_1_1.csv")
    common = read_csv("pit_multimarket_common_support_inference_v8_1_1.csv")
    common_aggregate = common[common["aggregation_level"].isin(["market", "selected_market_macro"])].copy()
    intersection = read_csv("pit_multimarket_equal_market_macro_v8.csv")
    intersection = intersection[
        intersection["universe_semantics"].eq("PIT_CURRENT_INTERSECTION_CONTROL")
    ]
    aggregation = read_csv("pit_multimarket_aggregation_robustness_v8_1.csv")
    lomo = read_csv("pit_multimarket_leave_one_market_out_v8_1.csv")
    loio = read_csv("pit_multimarket_leave_one_interval_out_v8_1.csv")
    seed = read_csv("pit_multimarket_seed_robustness_v8.csv")
    seed_summary = seed[seed["record_type"].eq("summary")].copy()
    sensitivity = read_csv("fbm_klci_5014_sensitivity_v8_1.csv")
    calibration = read_csv("calibration_estimability_audit_v8_1_1.csv")
    calibration_summary = pd.DataFrame(
        [
            ("Date-level rows", len(calibration)),
            ("Estimable", int(calibration["calibration_estimable"].astype(bool).sum())),
            ("Flagged", int((~calibration["calibration_estimable"].astype(bool)).sum())),
            ("Small-sample warnings", int(calibration["calibration_small_sample_warning"].astype(bool).sum())),
            ("Non-finite warnings", int(calibration["calibration_nonfinite_coefficient_warning"].astype(bool).sum())),
            ("Separation warnings", int(calibration["calibration_separation_warning"].astype(bool).sum())),
            ("Extreme-coefficient warnings", int(calibration["calibration_extreme_coefficient_warning"].astype(bool).sum())),
        ],
        columns=["Calibration audit field", "Count"],
    )
    families = read_csv("multiple_testing_family_registry_v8_1_1.csv")
    http_audit_path = QA / "http_retrieval_audit_v8_1.csv"
    http_audit = pd.read_csv(http_audit_path) if http_audit_path.exists() else pd.DataFrame()
    release_checks = pd.read_csv(AUDITS / "LOCKED_RELEASE_CHECKSUM_RESULTS.csv")
    claims_template = pd.DataFrame(
        [
            ("C01", "Confirmatory", "No supported five-market block-20 Brier difference for any model", "SUPPORTED"),
            ("C02", "Operational", "Current arm has about 3.45 percentage points lower macro coverage", "SUPPORTED"),
            ("C03", "Diagnostic", "Support and composition affect some numerical comparisons", "DESCRIPTIVE_ONLY"),
            ("C04", "Exploratory", "Historical-probability result is sensitive to block length and market availability", "EXPLORATORY"),
        ],
        columns=["Claim ID", "Evidence class", "Claim", "Status"],
    )

    sheets: list[tuple[str, pd.DataFrame]] = [
        ("S01_market_sources", market_sources),
        ("S02_membership_summary", main_tables["table_01_market_scope.csv"]),
        ("S03_malaysia_anchors", malaysia_summary),
        ("S04_interval_inventory", intervals),
        ("S05_rejection_audit", rejection_summary),
        ("S06_fold_calendar", folds),
        ("S07_absolute_performance", read_csv("pit_multimarket_performance_by_interval_v8.csv")),
        ("S08_bootstrap_results", primary),
        ("S09_common_support", common_aggregate),
        ("S10_intersection_control", intersection),
        ("S11_aggregation_rules", aggregation),
        ("S12_leave_market_out", lomo),
        ("S13_leave_interval_out", loio),
        ("S14_seed_robustness", seed_summary),
        ("S15_5014_sensitivity", sensitivity),
        ("S16_calibration_audit", calibration_summary),
        ("S17_testing_families", families),
        ("S18_http_quarantine", http_audit),
        ("S19_release_checks", release_checks),
        ("S20_claim_evidence", claims_template),
    ]
    path = TABLES / "supplementary_tables.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in sheets:
            frame.to_excel(writer, sheet_name=name[:31], index=False)
        workbook = writer.book
        for sheet in workbook.worksheets:
            sheet.freeze_panes = "A2"
            sheet.sheet_view.showGridLines = False
            sheet.auto_filter.ref = sheet.dimensions
            sheet.page_setup.orientation = (
                "landscape" if sheet.max_column > 7 else "portrait"
            )
            sheet.page_setup.fitToWidth = 1
            sheet.page_setup.fitToHeight = 0
            sheet.sheet_properties.pageSetUpPr.fitToPage = True
            sheet.print_title_rows = "1:1"
            for cell in sheet[1]:
                cell.font = cell.font.copy(bold=True)
                cell.alignment = cell.alignment.copy(wrap_text=True)
            for column_cells in sheet.columns:
                values = [str(c.value or "") for c in list(column_cells)[:500]]
                width = min(max(max((len(v) for v in values), default=8) + 2, 10), 42)
                sheet.column_dimensions[column_cells[0].column_letter].width = width


def build_source_map(
    tables: dict[str, pd.DataFrame],
    figures: dict[str, pd.DataFrame],
) -> None:
    rows = [
        ("Table 1", "research_data/point_in_time/reports/forecasting_ready_market_inventory_v8.csv; research_data/point_in_time/reports/forecasting_ready_interval_inventory_v8.csv", "Market scope, membership history and execution"),
        ("Table 2", "research_data/point_in_time/config/pit_multimarket_forecasting_v8.yaml; research_data/point_in_time/config/pit_multimarket_forecasting_v8_1.yaml", "Locked protocol"),
        ("Table 3", "research_data/point_in_time/reports/forecasting_ready_market_inventory_v8.csv", "Execution counts"),
        ("Table 4", "research_data/point_in_time/reports/pit_multimarket_equal_market_macro_v8.csv", "Absolute performance"),
        ("Table 5", "research_data/point_in_time/reports/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv", "Confirmatory inference"),
        ("Table 6", "research_data/point_in_time/reports/pit_multimarket_support_composition_diagnostic_v8_1.csv", "Full versus common support"),
        ("Table 7", "research_data/point_in_time/reports/pit_multimarket_performance_by_market_v8.csv; research_data/point_in_time/reports/pit_multimarket_equal_market_macro_v8.csv", "Operational coverage"),
        ("Table 8", "research_data/point_in_time/reports/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv; research_data/point_in_time/reports/pit_multimarket_aggregation_robustness_v8_1.csv; research_data/point_in_time/reports/pit_multimarket_leave_one_market_out_v8_1.csv; research_data/point_in_time/reports/pit_multimarket_leave_one_interval_out_v8_1.csv", "Robustness summary"),
        ("Figure 1", "research_data/point_in_time/config/pit_multimarket_forecasting_v8.yaml; research_data/point_in_time/config/pit_multimarket_forecasting_v8_1.yaml", "Research design"),
        ("Figure 2", "research_data/point_in_time/reports/forecasting_ready_interval_inventory_v8.csv", "Interval timeline"),
        ("Figure 3", "research_data/point_in_time/reports/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv", "Confirmatory forest"),
        ("Figure 4", "research_data/point_in_time/reports/pit_multimarket_support_composition_diagnostic_v8_1.csv", "Support comparison"),
        ("Figure 5", "research_data/point_in_time/reports/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv", "Market-model heatmap"),
        ("Figure 6", "research_data/point_in_time/reports/pit_multimarket_aggregation_robustness_v8_1.csv; research_data/point_in_time/reports/pit_multimarket_leave_one_market_out_v8_1.csv", "Robustness plot"),
    ]
    frame = pd.DataFrame(rows, columns=["Item", "Locked source artifact", "Use"])
    write_csv(frame, OUT_MANIFESTS / "table_figure_source_map.csv")


def main() -> None:
    ensure_dirs()
    integrity = verify_locked_release()
    tables = build_main_tables()
    figures = build_figures()
    build_supplementary_workbook(tables)
    build_source_map(tables, figures)
    payload = {
        "generated_at_utc": utc_now(),
        "title": TITLE,
        "experiment_id": EXPERIMENT_ID,
        "release_status": RELEASE_STATUS,
        "integrity_status": integrity["status"],
        "main_table_count": len(tables),
        "main_figure_count": len(figures),
        "supplementary_workbook_sheet_count": 20,
        "writes_confined_to": OUTPUT.relative_to(WORKSPACE).as_posix(),
        "locked_artifacts_modified": False,
    }
    (OUT_MANIFESTS / "data_asset_build_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
