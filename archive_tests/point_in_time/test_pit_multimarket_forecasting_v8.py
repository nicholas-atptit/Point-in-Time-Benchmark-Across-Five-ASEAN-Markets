from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.point_in_time import pit_multimarket_forecasting_v8 as v8


ROOT = Path(__file__).resolve().parents[2]
PIT_ROOT = ROOT / "research_data" / "point_in_time"
RECONSTRUCTED = PIT_ROOT / "reconstructed"
SOURCE_ROOT = PIT_ROOT / "sources" / "fbm_klci_v8"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fbm_interval_sets() -> dict[str, set[str]]:
    intervals = pd.read_parquet(v8.FBM_INTERVAL_PATH)
    return {
        str(start): set(group["ticker_on_date"].astype(str))
        for start, group in intervals.groupby("interval_start", sort=True)
    }


def test_v8_version_scope_and_frozen_v7_protocol() -> None:
    config = yaml.safe_load(v8.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == v8.EXPERIMENT_ID
    assert config["study_label"] == v8.STUDY_LABEL
    assert config["experiment_version"] == "v8"
    assert config["membership_version"] == "pit-membership-v8"
    assert config["data_version"] == "pit-data-v8"
    assert config["snapshot_version"] == "pit-universe-v8"
    assert config["price_version"] == "pit-data-v8"
    assert config["manifest_version"] == "manifest-portability-v8.0.0"
    assert config["predefined_market_scope"] == list(v8.MARKETS)
    assert config["feature_columns"] == list(v8.FEATURE_COLUMNS)
    assert config["seeds"] == [11, 23, 37, 53, 71]
    assert set(config["model_registry"]) == set(v8.MODEL_SPECS)
    assert config["v7_reference"]["protocol_change"] == "none"


def test_v7_remains_checksum_valid_and_frozen() -> None:
    result = v8.validate_v7_immutability()
    assert result["status"] == "PASS"
    assert result["v7_manifest_records"] == 34
    assert result["v7_status"] == "PARTIAL_DATA_BLOCKED"
    assert result["failures"] == []


def test_source_registry_is_complete_and_all_local_hashes_validate() -> None:
    registry = pd.read_csv(SOURCE_ROOT / "fbm_klci_source_registry_v8.csv")
    required = {
        "source_id",
        "publisher",
        "source_tier",
        "document_title",
        "document_type",
        "publication_date",
        "as_of_date",
        "effective_date",
        "retrieval_date",
        "source_url",
        "archive_url",
        "local_cache_path",
        "sha256",
        "redistribution_status",
        "evidence_role",
        "anchor_or_event",
        "entering_names",
        "exiting_names",
        "source_quality_status",
        "notes",
    }
    assert required <= set(registry.columns)
    assert not registry["source_id"].duplicated().any()
    assert len(registry) >= 20
    for record in registry.to_dict("records"):
        local = str(record.get("local_cache_path") or "")
        if local and local.lower() != "nan":
            path = ROOT / local
            assert path.is_file(), path
            assert _sha256(path) == str(record["sha256"])


def test_full_anchors_are_exactly_30_and_cross_validated() -> None:
    anchors = pd.read_csv(
        RECONSTRUCTED / "fbm_klci_anchor_snapshots_v8.csv",
        dtype={"bursa_code": str},
    )
    assert set(anchors["anchor_date"].astype(str)) == {"2017-12-31", "2019-12-23"}
    assert anchors.groupby("anchor_id")["stable_security_id"].nunique().eq(30).all()
    assert anchors.groupby("anchor_id")["bursa_code"].nunique().eq(30).all()
    assert anchors["anchor_validation_status"].eq(
        "ACCEPTED_EXACT_30_CROSS_VALIDATED"
    ).all()
    forbidden = anchors["security_name_canonical"].str.lower().str.contains(
        r"\bcash\b|\betf unit\b|\bmid 70\b", regex=True
    )
    assert not forbidden.any()


def test_exactly_30_fbm_members_per_interval_and_validated_date() -> None:
    intervals = pd.read_parquet(v8.FBM_INTERVAL_PATH)
    daily = pd.read_parquet(v8.FBM_DAILY_PATH)
    assert intervals["interval_id"].nunique() == 13
    assert intervals.groupby("interval_id")["security_id"].nunique().eq(30).all()
    assert daily.groupby("membership_date")["security_id"].nunique().eq(30).all()
    assert not daily.duplicated(["membership_date", "security_id"]).any()


def test_scheduled_2018_reviews_apply_exact_additions_and_deletions() -> None:
    sets = _fbm_interval_sets()
    before_june = sets["2018-01-01"]
    after_june = sets["2018-06-18"]
    assert {"5014", "5168", "7277"} <= after_june
    assert {"1015", "4677", "6399"}.isdisjoint(after_june)
    assert after_june == (
        before_june - {"1015", "4677", "6399"}
    ) | {"5014", "5168", "7277"}
    after_december = sets["2018-12-24"]
    assert after_december == (
        after_june - {"5235SS", "4863"}
    ) | {"7113", "1015"}


def test_effective_date_boundaries_prevent_future_membership_leakage() -> None:
    intervals = pd.read_parquet(v8.FBM_INTERVAL_PATH)
    intervals["interval_start"] = pd.to_datetime(intervals["interval_start"])
    intervals["interval_end_exclusive"] = pd.to_datetime(
        intervals["interval_end_exclusive"]
    )
    june_pre = intervals[intervals["interval_start"].eq("2018-01-01")]
    june_post = intervals[intervals["interval_start"].eq("2018-06-18")]
    assert june_pre["interval_end_exclusive"].dt.date.unique().tolist() == [
        pd.Timestamp("2018-06-18").date()
    ]
    assert "5014" not in set(june_pre["ticker_on_date"].astype(str))
    assert "5014" in set(june_post["ticker_on_date"].astype(str))
    december_pre = intervals[intervals["interval_start"].eq("2018-06-18")]
    december_post = intervals[intervals["interval_start"].eq("2018-12-24")]
    assert "7113" not in set(december_pre["ticker_on_date"].astype(str))
    assert "7113" in set(december_post["ticker_on_date"].astype(str))


def test_extraordinary_events_and_structural_continuity_are_explicit() -> None:
    events = pd.read_csv(RECONSTRUCTED / "fbm_klci_membership_events_v8.csv")
    extraordinary = events[
        events["scheduled_or_extraordinary"].eq("extraordinary")
    ]
    assert len(extraordinary) >= 5
    assert {
        "demerger_temporary_inclusion",
        "demerger_rule_6_2_6_demotion",
        "merger_structural_continuation",
        "legal_name_change",
        "delisting_after_index_exit",
    } <= set(extraordinary["event_type"])
    assert extraordinary["source_id_primary"].notna().all()


def test_stable_ids_renames_and_stapled_security_are_deterministic() -> None:
    crosswalk = pd.read_csv(
        v8.FBM_CROSSWALK_PATH, dtype={"bursa_code": str}
    )
    crosswalk["bursa_code"] = crosswalk["bursa_code"].str.zfill(4)
    assert not crosswalk["stable_security_id"].duplicated().any()
    for record in crosswalk.to_dict("records"):
        assert record["stable_security_id"] == v8.v7.v6.security_id(
            "FBM_KLCI", record["bursa_code"]
        )
    celcom = crosswalk[crosswalk["bursa_code"].eq("6947")].iloc[0]
    assert celcom["rename_effective_date"] == "2023-02-24"
    assert "DiGi" in celcom["historical_company_names"]
    guthrie = crosswalk[crosswalk["bursa_code"].eq("5285")].iloc[0]
    assert guthrie["rename_effective_date"] == "2024-05-28"
    stapled = crosswalk[crosswalk["bursa_code"].eq("5235SS")]
    assert len(stapled) == 1
    assert bool(stapled.iloc[0]["stapled_security_flag"])
    anchors = pd.read_csv(
        RECONSTRUCTED / "fbm_klci_anchor_snapshots_v8.csv",
        dtype={"bursa_code": str},
    )
    assert (anchors["bursa_code"].eq("5235SS")).sum() == 1


def test_anchor_to_anchor_chain_equality_and_source_provenance_pass() -> None:
    validation = json.loads(v8.FBM_VALIDATION_PATH.read_text(encoding="utf-8"))
    assert validation["status"] in {"PASS", "PASS_WITH_DOCUMENTED_LIMITATIONS"}
    assert validation["anchor_to_anchor_exact_set_equality"] is True
    assert validation["unknown_membership_dates_inside_evaluated_windows"] == 0
    assert validation["no_current_constituent_backcast"] is True
    assert validation["no_future_event_applied_early"] is True
    assert validation["source_checksums_valid"] is True
    assert validation["failures"] == []
    trace = pd.read_csv(
        RECONSTRUCTED / "fbm_klci_reconstruction_trace_v8.csv"
    )
    exact = trace["validation_result"].eq("PASS_EXACT_SET_EQUALITY")
    assert exact.any()
    assert trace["validation_result"].str.startswith("PASS").all()


def test_current_arm_is_isolated_from_pit_membership_evidence() -> None:
    members = pd.read_parquet(v8.MEMBERSHIP_DAILY_PATH)
    fbm = members[members["index_id"].eq("FBM_KLCI")].copy()
    fbm["membership_date"] = pd.to_datetime(fbm["membership_date"])
    early = fbm[fbm["membership_date"].eq(fbm["membership_date"].min())]
    pit_codes = set(
        early.loc[early["universe_semantics"].eq(v8.PIT), "ticker_on_date"].astype(str)
    )
    current_codes = set(
        early.loc[
            early["universe_semantics"].eq(v8.CURRENT), "ticker_on_date"
        ].astype(str)
    )
    assert len(pit_codes) == 30
    assert len(current_codes) == 30
    assert pit_codes != current_codes
    assert early.loc[
        early["universe_semantics"].eq(v8.PIT), "membership_version"
    ].eq("pit-membership-v8").all()
    assert "CURRENT_2026_CONFIG" not in " ".join(
        early.loc[
            early["universe_semantics"].eq(v8.PIT), "membership_chain_id"
        ].astype(str)
    )


def test_raw_price_membership_join_is_unadjusted_and_clears_locked_gates() -> None:
    raw = pd.read_parquet(v8.RECOVERED_RAW_PATH)
    malaysia = raw[raw["market"].eq("Malaysia")]
    assert len(malaysia) >= 73_697
    assert not malaysia.duplicated(["security_id", "date"]).any()
    assert malaysia["price_semantics"].eq("RAW_OHLC").all()
    assert {"open", "high", "low", "close"} <= set(malaysia.columns)
    assert "adj_close" not in {column.lower() for column in malaysia.columns}
    assert "volume" not in {column.lower() for column in malaysia.columns}
    audit = pd.read_csv(v8.PRICE_AUDIT_CSV_PATH)
    assert not audit["adjusted_close_used"].astype(bool).any()
    assert not audit["volume_used"].astype(bool).any()
    assert not audit["forward_fill_used"].astype(bool).any()
    assert not audit["imputation_used"].astype(bool).any()
    interval = pd.read_csv(v8.INTERVAL_INVENTORY_PATH)
    fbm = interval[interval["index_id"].eq("FBM_KLCI")]
    assert len(fbm) == 13
    assert fbm["eligible_for_forecasting"].astype(bool).all()
    assert (fbm["price_coverage"] >= 0.95).all()
    assert (fbm["feature_lookback_coverage"] >= 0.95).all()
    assert (fbm["target_pair_coverage"] >= 0.95).all()


def test_targets_labels_and_continuous_folds_are_strictly_ordered() -> None:
    features = pd.read_parquet(v8.FEATURE_PATH)
    assert (
        pd.to_datetime(features["target_date"])
        > pd.to_datetime(features["prediction_date"])
    ).all()
    assert features[v8.FEATURE_COLUMNS].notna().all().all()
    collapsed = pd.read_parquet(v8.COLLAPSED_PATH)
    assert (
        pd.to_datetime(collapsed["label_maturity_cutoff"])
        < pd.to_datetime(collapsed["prediction_date"])
    ).all()
    folds = pd.read_parquet(v8.FOLD_PATH)
    valid = folds[folds["valid_fold"].astype(bool)].copy()
    assert valid["continuous_market_history"].astype(bool).all()
    assert (valid["training_dates"] >= 60).all()
    assert (valid["calibration_dates"] == 20).all()
    assert (valid["test_dates"] == 20).all()
    assert (
        pd.to_datetime(valid["training_end"])
        < pd.to_datetime(valid["calibration_start"])
    ).all()
    assert (
        pd.to_datetime(valid["calibration_end"])
        < pd.to_datetime(valid["test_start"])
    ).all()
    assert valid[valid["index_id"].eq("FBM_KLCI")]["fold_id"].nunique() >= 2


def test_all_market_arm_model_cells_pass_declared_validity_gates() -> None:
    cells = pd.read_parquet(v8.ALL_CELLS_PATH)
    assert len(cells) == 5 * 3 * 4
    assert cells["valid"].astype(bool).all()
    assert (cells["scheduled_valid_folds"] >= 2).all()
    assert (cells["prediction_folds"] >= 2).all()
    assert (cells["oos_trading_dates"] >= 40).all()
    assert (cells["oos_security_date_predictions"] >= 800).all()
    assert cells["target_class_count"].eq(2).all()
    invalid = pd.read_csv(PIT_ROOT / "reports" / "pit_multimarket_invalid_cells_v8.csv")
    assert invalid.empty


def test_seed_collapse_is_unique_complete_and_exact_on_sample() -> None:
    seed = pd.read_parquet(v8.SEED_PATH)
    collapsed = pd.read_parquet(v8.COLLAPSED_PATH)
    assert len(seed) == len(v8.SEEDS) * len(collapsed)
    assert set(seed["seed"].astype(int)) == set(v8.SEEDS)
    assert collapsed["seed_count"].eq(len(v8.SEEDS)).all()
    assert collapsed["seed"].eq("COLLAPSED").all()
    assert collapsed["prediction_probability"].between(0, 1).all()
    key = [
        "interval_id",
        "universe_semantics",
        "fold_id",
        "security_id",
        "prediction_date",
        "model_id",
    ]
    assert not collapsed.duplicated(key).any()
    sample = collapsed[key].sample(n=128, random_state=271828)
    selected_seed = seed.merge(sample, on=key, how="inner", validate="many_to_one")
    means = selected_seed.groupby(key)["prediction_probability"].mean().sort_index()
    observed = (
        collapsed.merge(sample, on=key, how="inner", validate="one_to_one")
        .set_index(key)["prediction_probability"]
        .sort_index()
    )
    assert np.allclose(means, observed)


def test_bootstrap_is_reproducible_and_bh_families_are_metric_specific() -> None:
    vector = np.linspace(-0.01, 0.02, 80)
    first = v8.v7.analytics.moving_block_means(
        vector, samples=500, block_length=20, seed=271828
    )
    second = v8.v7.analytics.moving_block_means(
        vector, samples=500, block_length=20, seed=271828
    )
    assert np.array_equal(first, second)
    inference = pd.read_csv(
        PIT_ROOT / "reports" / "pit_multimarket_bootstrap_inference_v8.csv"
    )
    expected = (
        inference["inference_level"].astype(str)
        + "|"
        + inference["metric"].astype(str)
        + "|block_"
        + inference["block_length"].astype(str)
    )
    assert inference["bh_family"].eq(expected).all()
    assert inference.groupby("bh_family")["metric"].nunique().eq(1).all()
    assert inference.groupby("bh_family")["block_length"].nunique().eq(1).all()
    assert set(inference["block_length"].astype(int)) <= {10, 20, 40}


def test_prediction_coverage_is_separate_from_predictive_loss() -> None:
    inference = pd.read_csv(
        PIT_ROOT / "reports" / "pit_multimarket_bootstrap_inference_v8.csv"
    )
    assert {"brier_score", "prediction_coverage"} <= set(inference["metric"])
    assert not (
        inference[inference["metric"].eq("prediction_coverage")]["bh_family"]
        .isin(
            inference[inference["metric"].eq("brier_score")]["bh_family"]
        )
        .any()
    )
    status_text = v8.STATUS_MD_PATH.read_text(encoding="utf-8")
    assert "Primary Brier inference" in status_text
    assert "Operational prediction coverage" in status_text
    assert "not treated as forecasting-performance evidence" in status_text


def test_five_market_macro_uses_exactly_five_valid_markets() -> None:
    macro = pd.read_csv(
        PIT_ROOT / "reports" / "pit_multimarket_equal_market_macro_v8.csv"
    )
    assert len(macro) == 3 * 4
    assert macro["model_valid_markets"].eq(5).all()
    market = pd.read_csv(
        PIT_ROOT / "reports" / "pit_multimarket_performance_by_market_v8.csv"
    )
    assert set(market["index_id"]) == set(v8.MARKETS)


def test_manifest_is_portable_complete_and_checksum_valid() -> None:
    manifest = json.loads(v8.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == v8.EXPERIMENT_ID
    assert manifest["manifest_version"] == v8.MANIFEST_VERSION
    assert manifest["artifact_count"] == len(manifest["records"])
    assert manifest["artifact_count"] >= 60
    for record in manifest["records"]:
        relative = Path(record["artifact_relative_path"])
        assert not relative.is_absolute()
        path = ROOT / relative
        assert path.is_file(), path
        assert _sha256(path) == record["checksum"]
    validation = json.loads(v8.ARTIFACT_VALIDATION_PATH.read_text(encoding="utf-8"))
    assert validation["status"] == "PASS"
    assert validation["repository_relative_keys"] is True
    assert validation["checksum_failures"] == []
