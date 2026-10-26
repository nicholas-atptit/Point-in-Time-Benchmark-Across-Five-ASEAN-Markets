from __future__ import annotations

import json
import subprocess
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from scripts.point_in_time import pit_multimarket_forecasting_v8 as v8
from scripts.point_in_time import run_pit_v8_1_release as release


def _require(path):
    if not path.exists():
        pytest.skip(f"release artifact not built yet: {path.name}")
    return path


def _tiny_predictions() -> pd.DataFrame:
    rows = []
    for semantics, probability in [(v8.PIT, 0.4), (v8.CURRENT, 0.6)]:
        rows.append(
            {
                "market": "Malaysia",
                "index_id": "FBM_KLCI",
                "interval_id": "I01",
                "fold_id": "F01",
                "model_id": "logistic_regression",
                "security_id": "FBM_KLCI::0001",
                "prediction_date": date(2020, 1, 3),
                "target_date": date(2020, 1, 6),
                "universe_semantics": semantics,
                "actual_class": 1,
                "prediction_probability": probability,
                "predicted_class": int(probability >= 0.5),
                "label_maturity_cutoff": date(2020, 1, 2),
            }
        )
    return pd.DataFrame(rows)


def test_v7_and_v8_immutability():
    result = release.assert_v8_immutable(deep_predictions=False)
    assert result["status"] == "PASS"
    assert result["v7_integrity"]["status"] == "PASS"
    assert result["v8_manifest_records"] == 76


def test_prediction_invariants():
    result = release.assert_v8_immutable(deep_predictions=True)
    checks = result["prediction_checks"]
    assert checks["seed_rows"] == 5_695_980
    assert checks["collapsed_rows"] == 1_139_196
    assert checks["unique_seed_keys"]
    assert checks["unique_collapsed_keys"]
    assert checks["collapsed_seed_counts"] == [5]


def test_release_manifest_reconciles():
    path = _require(release.RELEASE_MANIFEST_JSON)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert not release._validate_manifest_records(manifest)
    assert manifest["manifest_role"] == (
        "RELEASE_CONTENT_MANIFEST_NOT_WORKSPACE_INVENTORY"
    )


def test_workspace_inventory_reconciles():
    manifest = json.loads(
        _require(release.RELEASE_MANIFEST_JSON).read_text(encoding="utf-8")
    )
    inventory = json.loads(
        _require(release.WORKSPACE_INVENTORY_JSON).read_text(encoding="utf-8")
    )
    assert not release._validate_workspace_inventory(inventory, manifest)
    assert inventory["checkpoint_policy"] == (
        "REMOVED_AFTER_VALIDATED_INCORPORATION"
    )


def test_checkpoint_inventory_and_deletion_policy():
    payload = json.loads(
        _require(release.CHECKPOINT_JSON_PATH).read_text(encoding="utf-8")
    )
    records = pd.read_csv(_require(release.CHECKPOINT_CSV_PATH))
    assert payload["checkpoint_files"] == 165
    assert payload["completion_markers"] == 165
    assert payload["component_records"] == 9_900
    assert payload["incorporation_validation"] == "PASS"
    assert records["incorporated_into_final_output"].astype(bool).all()
    assert records["safe_to_delete"].astype(bool).all()
    assert not release.CHECKPOINT_ROOT.exists()


def test_two_level_manifest_envelope():
    path = _require(release.RELEASE_ENVELOPE_PATH)
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["release_manifest_sha256"] == release.sha256_file(
        release.RELEASE_MANIFEST_JSON
    )
    assert envelope["workspace_inventory_sha256"] == release.sha256_file(
        release.WORKSPACE_INVENTORY_JSON
    )
    assert release.relative(release.RELEASE_ENVELOPE_PATH) not in {
        row["artifact_relative_path"]
        for row in json.loads(
            release.RELEASE_MANIFEST_JSON.read_text(encoding="utf-8")
        )["records"]
    }


def test_source_document_and_extraction_hashes_are_separate():
    registry = pd.read_csv(_require(release.SOURCE_REGISTRY_CSV))
    extraction = registry[registry["extraction_local_path"].notna()]
    assert len(extraction) == 8
    assert extraction["extraction_sha256"].notna().all()
    anchor = registry[registry["source_id"].eq("SRC_ANCHOR_2017_12_31")].iloc[0]
    assert pd.isna(anchor["source_document_sha256"])
    assert pd.notna(anchor["extraction_sha256"])


def test_source_mime_is_content_validated():
    registry = pd.read_csv(_require(release.SOURCE_REGISTRY_CSV))
    documents = registry[registry["source_document_local_path"].notna()]
    assert set(documents["source_document_mime_type"]) == {
        "application/pdf",
        "text/html",
    }
    for row in documents.to_dict("records"):
        path = release.REPO_ROOT / row["source_document_local_path"]
        assert release.detect_mime(path) == row["source_document_mime_type"]
        assert not release.is_http_challenge(path)


def test_cloudflare_and_failed_responses_are_quarantined():
    audit = pd.read_csv(_require(release.HTTP_AUDIT_PATH))
    assert len(audit) == 9
    assert audit["challenge_detected"].astype(bool).all()
    assert not audit["used_as_evidence"].astype(bool).any()
    assert all("http_challenge_quarantine_v8_1" in item for item in audit["quarantine_path"])


def test_no_challenge_response_is_evidence():
    audit = pd.read_csv(_require(release.HTTP_AUDIT_PATH))
    registry = pd.read_csv(_require(release.SOURCE_REGISTRY_CSV))
    evidence_paths = set(
        registry["source_document_local_path"].dropna().astype(str)
    ) | set(registry["extraction_local_path"].dropna().astype(str))
    assert not evidence_paths.intersection(set(audit["quarantine_path"].astype(str)))
    roles = " ".join(registry["evidence_role"].astype(str)).lower()
    assert "cloudflare" not in roles


def test_2017_anchor_provenance_is_accurate():
    registry = pd.read_csv(_require(release.SOURCE_REGISTRY_CSV))
    anchor = registry[registry["source_id"].eq("SRC_ANCHOR_2017_12_31")].iloc[0]
    assert anchor["evidentiary_status"] == (
        "ACCEPTED_TIER2_FACTUAL_EXTRACTION_WITH_REMOTE_DOCUMENT_REFERENCE"
    )
    assert anchor["source_document_retrieval_status"] == (
        "REMOTE_REFERENCE_ONLY_DOCUMENT_NOT_CACHED"
    )


def test_june_2018_remains_explicit_tier3():
    registry = pd.read_csv(_require(release.SOURCE_REGISTRY_CSV))
    row = registry[registry["source_id"].eq("SRC_REVIEW_2018_06")].iloc[0]
    assert int(row["source_tier"]) == 3
    assert row["evidentiary_status"] == (
        "ACCEPTED_TIER3_EXACT_ENDPOINT_CROSS_VALIDATED"
    )
    assert "not represented as a primary source" in row["notes"]


def test_5014_recovery_gap_is_explicit():
    audit = pd.read_csv(_require(release.RECOVERY_CSV_PATH))
    assert audit["recovery_status"].eq("UNRECOVERED_EXPLICIT_GAP").all()
    assert audit["raw_ohlc_rows"].eq(0).all()
    assert audit["joined_active_member_days"].eq(0).all()
    assert audit["requested_active_member_days"].eq(493).all()
    assert not audit["eligibility_threshold_changed"].astype(bool).any()


def test_5014_sensitivity_has_all_specifications_and_metrics():
    frame = pd.read_csv(_require(release.SENSITIVITY_CSV_PATH))
    assert set(frame["specification"]) == {
        "A_LOCKED_PRIMARY",
        "B_EXCLUDE_5014_ALL_ARMS",
        "C_EXCLUDE_5014_ACTIVE_INTERVALS",
    }
    for metric in ["brier_score", "log_loss", "balanced_accuracy", "prediction_coverage"]:
        assert metric in frame.columns or f"delta_{metric}_current_minus_pit" in frame.columns
    assert "brier_rank" in frame.columns


def test_common_support_exact_matching_and_target_equality():
    matched = release.match_common_support_predictions(_tiny_predictions())
    assert len(matched) == 1
    assert matched["actual_class_pit"].eq(matched["actual_class_current"]).all()
    assert not matched["common_support_key"].duplicated().any()
    assert (
        pd.to_datetime(matched["label_maturity_cutoff_pit"])
        < pd.to_datetime(matched["prediction_date"])
    ).all()


def test_common_support_is_subset_and_deterministic():
    source = _tiny_predictions()
    first = release.match_common_support_predictions(source)
    second = release.match_common_support_predictions(source.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(first, second)
    pit_keys = set(
        map(tuple, source[source["universe_semantics"].eq(v8.PIT)][release.MATCH_KEY].to_numpy())
    )
    current_keys = set(
        map(tuple, source[source["universe_semantics"].eq(v8.CURRENT)][release.MATCH_KEY].to_numpy())
    )
    matched_keys = set(map(tuple, first[release.MATCH_KEY].to_numpy()))
    assert matched_keys <= pit_keys
    assert matched_keys <= current_keys


def test_common_support_rejects_target_mismatch():
    source = _tiny_predictions()
    source.loc[source["universe_semantics"].eq(v8.CURRENT), "actual_class"] = 0
    with pytest.raises(RuntimeError, match="target mismatch"):
        release.match_common_support_predictions(source)


def test_support_composition_diagnostic_is_descriptive():
    frame = pd.read_csv(_require(release.SUPPORT_DIAGNOSTIC_CSV_PATH))
    columns = [
        column
        for column in frame.columns
        if column.startswith("descriptive_support_composition_component_")
    ]
    assert columns
    assert frame["interpretation"].str.contains("not a causal decomposition").all()


def test_primary_bootstrap_reproducibility_and_size():
    frame = pd.read_parquet(_require(release.PRIMARY_BOOTSTRAP_PARQUET_PATH))
    primary = frame[
        frame["support_specification"].eq("FULL_SUPPORT")
        & frame["inference_level"].eq("selected_five_market_macro")
        & frame["block_length"].eq(20)
    ]
    assert len(primary) == 4
    assert primary["bootstrap_samples"].eq(10_000).all()
    assert primary["bootstrap_seed"].eq(271_828).all()
    assert primary["valid"].astype(bool).all()


def test_finite_sample_p_value_uses_plus_one():
    result = release._finite_sample_summary(
        1.0, np.asarray([1.0, 1.5, 0.5, 1.25])
    )
    # Centered replicates are [0, .5, -.5, .25], so zero are as extreme as 1.
    assert result["finite_sample_extreme_count"] == 0
    assert result["raw_p_value"] == pytest.approx(1 / 5)


def test_bh_family_membership_is_exact():
    rows = []
    for model, p_value in zip(sorted(v8.MODEL_SPECS), [0.01, 0.02, 0.03, 0.04]):
        rows.append(
            {
                "support_specification": "FULL_SUPPORT",
                "inference_level": "selected_five_market_macro",
                "metric": "brier_score",
                "block_length": 20,
                "model_id": model,
                "valid": True,
                "raw_p_value": p_value,
                "ci_lower": 0.01,
                "ci_upper": 0.02,
            }
        )
    adjusted = release.apply_bh_families(pd.DataFrame(rows))
    assert adjusted["family_id"].eq("CONFIRMATORY_MACRO_BRIER_BLOCK20").all()
    assert adjusted["confirmatory_or_exploratory"].eq("CONFIRMATORY").all()
    assert adjusted["adjusted_p_value"].notna().all()


def test_moving_block_bootstrap_respects_interval_and_is_reproducible():
    rows = []
    start = date(2020, 1, 1)
    for interval, offset in [("I01", 0), ("I02", 1)]:
        for idx in range(100):
            rows.append(
                {
                    "interval_id": interval,
                    "index_id": "FBM_KLCI",
                    "prediction_date": start + timedelta(days=idx),
                    "delta_brier_score_current_minus_pit": (idx % 7) / 100 + offset,
                }
            )
    frame = pd.DataFrame(rows)
    first = release._hierarchical_bootstrap_replicates(
        frame,
        "delta_brier_score_current_minus_pit",
        block_length=20,
        samples=100,
        seed_parts=[1, 2, 3],
    )
    second = release._hierarchical_bootstrap_replicates(
        frame,
        "delta_brier_score_current_minus_pit",
        block_length=20,
        samples=100,
        seed_parts=[1, 2, 3],
    )
    assert first[2] is None
    for key in first[0]:
        np.testing.assert_array_equal(first[0][key], second[0][key])


def test_aggregation_robustness_and_leave_out_outputs():
    aggregation = pd.read_csv(_require(release.AGGREGATION_PATH))
    assert {
        "PRIMARY_EQUAL_MARKET",
        "EQUAL_OOS_DATE",
        "EQUAL_INTERVAL",
        "SECURITY_DATE_WEIGHTING",
        "EXCLUDE_INTERVALS_LT_80_OOS_DATES",
        "EXCLUDE_5014_ACTIVE_INTERVALS",
    } <= set(aggregation["aggregation_specification"])
    assert aggregation.loc[
        aggregation["aggregation_specification"].eq("PRIMARY_EQUAL_MARKET"),
        "primary",
    ].astype(bool).all()
    assert len(pd.read_csv(_require(release.LOMO_PATH))) > 0
    assert len(pd.read_csv(_require(release.LOIO_PATH))) > 0


def test_clean_copy_release_validation_recorded():
    validation = json.loads(
        _require(release.ARTIFACT_VALIDATION_PATH).read_text(encoding="utf-8")
    )
    clean = validation.get("clean_copy_validation")
    if clean is None:
        pytest.skip("clean-copy validation is recorded after the pre-seal suite")
    assert clean["status"] == "PASS"
    assert clean["command_exit_code"] == 0


def test_no_absolute_local_paths_or_secrets():
    manifest = json.loads(
        _require(release.RELEASE_MANIFEST_JSON).read_text(encoding="utf-8")
    )
    result = release._portable_security_checks(manifest)
    assert result["status"] == "PASS"
    assert not result["absolute_local_path_hits"]
    assert not result["secret_pattern_hits"]


def test_git_commit_and_annotated_tag_are_recorded():
    envelope = json.loads(
        _require(release.RELEASE_ENVELOPE_PATH).read_text(encoding="utf-8")
    )
    commit = subprocess.run(
        ["git", "rev-parse", f"{envelope['git_commit_hash']}^{{commit}}"],
        cwd=release.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert commit == envelope["git_commit_hash"]
    tag_type = subprocess.run(
        ["git", "cat-file", "-t", release.FINAL_TAG],
        cwd=release.REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tag_type == "tag"
