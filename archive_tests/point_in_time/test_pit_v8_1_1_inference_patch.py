from __future__ import annotations

import json
import math
import subprocess

import pandas as pd
import pytest

from scripts.point_in_time import run_pit_v8_1_1_release as patch


def _require(path):
    if not path.exists():
        pytest.skip(f"V8.1.1 artifact not built yet: {path.name}")
    return path


@pytest.fixture
def bootstrap() -> pd.DataFrame:
    return pd.read_parquet(_require(patch.PRIMARY_PARQUET_PATH))


def _full_block40(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        frame["support_specification"].eq("FULL_SUPPORT")
        & frame["block_length"].eq(40)
    ]


def test_five_market_macro_requires_five_contributors(bootstrap):
    five = bootstrap[bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")]
    valid = five[five["valid"].astype(bool)]
    invalid = five[~five["valid"].astype(bool)]
    assert valid["contributing_market_count"].eq(5).all()
    assert invalid["contributing_market_count"].lt(5).all()
    assert valid["all_required_markets_available"].astype(bool).all()
    assert not invalid["all_required_markets_available"].astype(bool).any()


def test_block40_five_market_rows_are_insufficient(bootstrap):
    rows = _full_block40(bootstrap)
    rows = rows[rows["macro_scope"].eq("FIVE_MARKET_MACRO")]
    assert len(rows) == 4
    assert not rows["valid"].astype(bool).any()
    assert rows["result_classification"].eq(
        "INSUFFICIENT_INFORMATION"
    ).all()
    assert rows["reason"].eq(
        "one or more predefined markets unavailable"
    ).all()


def test_block40_available_rows_contain_exactly_four_markets(bootstrap):
    rows = _full_block40(bootstrap)
    rows = rows[rows["macro_scope"].eq("AVAILABLE_MARKET_MACRO")]
    assert len(rows) == 4
    assert rows["contributing_market_count"].eq(4).all()
    assert rows["contributing_markets"].eq(
        "SET50;VN30;PSEI;FBM_KLCI"
    ).all()


def test_block40_idx30_exclusion_is_explicit(bootstrap):
    rows = _full_block40(bootstrap)
    available = rows[rows["macro_scope"].eq("AVAILABLE_MARKET_MACRO")]
    assert available["excluded_markets"].eq("IDX30").all()
    audit = pd.read_csv(_require(patch.AVAILABILITY_AUDIT_PATH))
    selected = audit[
        audit["support_specification"].eq("FULL_SUPPORT")
        & audit["block_length"].eq(40)
    ]
    assert selected["excluded_markets"].eq("IDX30").all()
    assert selected["excluded_market_reasons"].str.contains(
        "insufficient block-length-40 information", regex=False
    ).all()


def test_available_market_family_is_separate_and_exploratory(bootstrap):
    rows = _full_block40(bootstrap)
    available = rows[rows["macro_scope"].eq("AVAILABLE_MARKET_MACRO")]
    assert available["confirmatory_or_exploratory"].eq("EXPLORATORY").all()
    assert not available["family_id"].eq(
        "CONFIRMATORY_MACRO_BRIER_BLOCK20"
    ).any()
    registry = pd.read_csv(_require(patch.FAMILY_REGISTRY_PATH))
    families = registry[
        registry["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
    ]
    assert len(families) == 2
    assert families["hypothesis_count"].eq(4).all()
    assert families["available_market_family_is_separate"].astype(bool).all()


def test_confirmatory_block20_rows_have_five_markets_and_are_unchanged(
    bootstrap,
):
    patched = bootstrap[
        bootstrap["support_specification"].eq("FULL_SUPPORT")
        & bootstrap["macro_scope"].eq("FIVE_MARKET_MACRO")
        & bootstrap["block_length"].eq(20)
    ].sort_values("model_id")
    source = pd.read_parquet(patch.BASE_PRIMARY_PARQUET)
    source = source[
        source["support_specification"].eq("FULL_SUPPORT")
        & source["inference_level"].eq("selected_five_market_macro")
        & source["block_length"].eq(20)
    ].sort_values("model_id")
    assert len(patched) == 4
    assert patched["contributing_market_count"].eq(5).all()
    columns = [
        "model_id",
        "point_estimate",
        "ci_lower",
        "ci_upper",
        "raw_p_value",
        "adjusted_p_value",
        "result_classification",
    ]
    pd.testing.assert_frame_equal(
        source[columns].reset_index(drop=True),
        patched[columns].reset_index(drop=True),
        check_dtype=False,
    )


def test_historical_probability_four_market_supported_result_is_retained(
    bootstrap,
):
    row = _full_block40(bootstrap)
    row = row[
        row["macro_scope"].eq("AVAILABLE_MARKET_MACRO")
        & row["model_id"].eq("historical_class_probability")
    ].iloc[0]
    assert math.isclose(
        row["point_estimate"],
        0.00027648249533416733,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        row["ci_lower"],
        0.00008693003484461146,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        row["ci_upper"],
        0.0004634816708086555,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        row["adjusted_p_value"],
        0.015198480151984802,
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert row["result_classification"] == (
        "SUPPORTED_CURRENT_PIT_DIFFERENCE"
    )


def test_calibration_instability_flags_are_deterministic():
    stable = {
        "sample_size": 40,
        "positive_count": 20,
        "negative_count": 20,
        "coefficients": [0.1, 0.2, 0.9, 1.1],
    }
    first = patch.calibration_flags(**stable)
    second = patch.calibration_flags(**stable)
    assert first == second
    assert first["calibration_estimable"]
    unstable = patch.calibration_flags(
        sample_size=20,
        positive_count=1,
        negative_count=19,
        coefficients=[float("nan"), 30.0, 2.0, 40.0],
    )
    assert not unstable["calibration_estimable"]
    assert unstable["calibration_separation_warning"]
    assert unstable["calibration_extreme_coefficient_warning"]
    assert unstable["calibration_nonfinite_coefficient_warning"]
    assert unstable["calibration_small_sample_warning"]


def test_calibration_audit_and_common_support_reconcile():
    common = pd.read_csv(_require(patch.COMMON_SUPPORT_PATH))
    audit = pd.read_csv(_require(patch.CALIBRATION_AUDIT_PATH))
    assert set(patch.REQUIRED_CALIBRATION_FIELDS) <= set(common.columns)
    assert len(audit) == common["aggregation_level"].eq("date").sum()
    assert audit["calibration_estimable"].dtype == bool
    flagged = ~audit["calibration_estimable"]
    assert flagged.any()
    assert audit.loc[
        flagged, "calibration_estimability_reason"
    ].notna().all()


def test_v8_1_predictions_and_base_release_are_immutable():
    result = patch.assert_base_immutable(deep_predictions=True)
    assert result["status"] == "PASS"
    assert result["seed_level_rows"] == 5_695_980
    assert result["collapsed_rows"] == 1_139_196
    assert result["v8_1_commit"] == patch.BASE_V81_COMMIT


def test_release_manifest_checksums_pass():
    manifest = json.loads(
        _require(patch.RELEASE_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert not patch._validate_manifest(manifest)
    validation = patch.validate_release(
        require_envelope=patch.RELEASE_ENVELOPE_PATH.exists(),
        require_git_seal=False,
        deep_predictions=False,
    )
    assert validation["validation_status"] == "PASS"


def test_clean_copy_validation_is_recorded():
    validation = json.loads(
        _require(patch.VALIDATION_PATH).read_text(encoding="utf-8")
    )
    clean = validation.get("clean_copy_validation")
    if clean is None:
        pytest.skip("clean-copy validation is recorded after content commit")
    assert clean["status"] == "PASS"
    assert clean["command_exit_code"] == 0
    assert clean["git_present"] is False


def test_final_annotated_tag_and_envelope():
    envelope = json.loads(
        _require(patch.RELEASE_ENVELOPE_PATH).read_text(encoding="utf-8")
    )
    commit = subprocess.run(
        ["git", "rev-parse", f"{envelope['git_commit_hash']}^{{commit}}"],
        cwd=patch.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == envelope["git_commit_hash"]
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", patch.FINAL_TAG],
        cwd=patch.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tag_type == "tag"
    resolved = subprocess.run(
        ["git", "rev-parse", f"{patch.FINAL_TAG}^{{}}"],
        cwd=patch.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=patch.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
