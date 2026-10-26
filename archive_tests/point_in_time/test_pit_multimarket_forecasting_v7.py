from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scripts.point_in_time import pit_multimarket_analytics_v6 as analytics
from scripts.point_in_time import pit_multimarket_forecasting_v7 as v7


ROOT = Path(__file__).resolve().parents[2]
PIT = ROOT / "research_data" / "point_in_time"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_v7_scope_and_frozen_protocol_registry() -> None:
    config = yaml.safe_load(v7.CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["experiment_id"] == v7.EXPERIMENT_ID
    assert config["study_label"] == v7.STUDY_LABEL
    assert config["experiment_version"] == "v7"
    assert config["predefined_market_scope"] == [
        "IDX30",
        "SET50",
        "VN30",
        "PSEI",
        "FBM_KLCI",
    ]
    assert config["feature_columns"] == list(v7.FEATURE_COLUMNS)
    assert config["seeds"] == [11, 23, 37, 53, 71]
    assert set(config["model_registry"]) == {
        "historical_class_probability",
        "logistic_regression",
        "random_forest",
        "lightgbm",
    }
    assert config["minimum_validity_gates"] == {
        "valid_folds_per_market_arm_model": 2,
        "oos_trading_dates_per_market_arm_model": 40,
        "oos_security_date_predictions_per_market_arm_model": 800,
        "both_target_classes_train": True,
        "both_target_classes_test": True,
    }


def test_v6_and_locked_v4_artifacts_remain_immutable() -> None:
    result = v7.validate_v6_immutability()
    assert result["v6_manifest_records"] == 51
    assert result["v6_manifest_checksums_valid"] is True
    assert result["v4_locked_records"] == 87
    assert result["v4_locked_checksums_valid"] is True
    assert result["failures"] == []


def test_psei_conflict_windows_are_gaps_and_full_list_resets_are_bounded() -> None:
    membership = pd.read_parquet(v7.MEMBERSHIP_PATH)
    psei = membership[membership["index_id"].eq("PSEI")].copy()
    assert psei["interval_id"].nunique() >= 2
    assert psei.groupby("interval_id")["security_id"].nunique().eq(30).all()
    assert pd.to_datetime(psei["interval_start"]).min().date().isoformat() == "2017-03-13"
    first_chain_end = pd.to_datetime(
        psei.loc[
            psei["membership_chain_id"].str.contains("2017H1"),
            "interval_end_exclusive",
        ]
    ).max()
    second_chain_start = pd.to_datetime(
        psei.loc[
            psei["membership_chain_id"].str.contains("2023EXTRA2"),
            "interval_start",
        ]
    ).min()
    assert first_chain_end.date().isoformat() == "2023-09-26"
    assert second_chain_start.date().isoformat() == "2023-10-04"


def test_dynamic_pit_membership_and_deliberate_current_backfill_are_isolated() -> None:
    members = pd.read_parquet(v7.MEMBERSHIP_DAILY_PATH)
    members["membership_date"] = pd.to_datetime(members["membership_date"])
    psei = members[members["index_id"].eq("PSEI")]
    plus_pit = psei[
        psei["ticker_on_date"].eq("PLUS")
        & psei["universe_semantics"].eq(v7.PIT)
    ]
    assert plus_pit["membership_date"].min().date().isoformat() == "2025-08-18"
    plus_current = psei[
        psei["ticker_on_date"].eq("PLUS")
        & psei["universe_semantics"].eq(v7.CURRENT)
    ]
    assert plus_current["membership_date"].min() < plus_pit["membership_date"].min()
    bloom_pit = psei[
        psei["ticker_on_date"].eq("BLOOM")
        & psei["universe_semantics"].eq(v7.PIT)
    ]
    assert not (
        bloom_pit["membership_date"] >= pd.Timestamp("2025-08-18")
    ).any()


def test_future_membership_does_not_change_earlier_pit_observations() -> None:
    observations = pd.read_parquet(v7.OBSERVATION_PATH)
    observations["prediction_date"] = pd.to_datetime(observations["prediction_date"])
    psei_pit = observations[
        observations["index_id"].eq("PSEI")
        & observations["universe_semantics"].eq(v7.PIT)
    ]
    assert psei_pit[
        psei_pit["ticker_on_date"].eq("PLUS")
        & (psei_pit["prediction_date"] < pd.Timestamp("2025-08-18"))
    ].empty
    assert psei_pit[
        psei_pit["ticker_on_date"].eq("RCR")
        & (psei_pit["prediction_date"] < pd.Timestamp("2026-02-02"))
    ].empty


def test_delisted_or_removed_security_remains_usable_historically() -> None:
    observations = pd.read_parquet(v7.OBSERVATION_PATH)
    observations["prediction_date"] = pd.to_datetime(observations["prediction_date"])
    pcor = observations[
        observations["index_id"].eq("PSEI")
        & observations["universe_semantics"].eq(v7.PIT)
        & observations["ticker_on_date"].eq("PCOR")
    ]
    assert len(pcor) > 0
    assert pcor["prediction_date"].max() < pd.Timestamp("2019-02-18")


def test_continuous_folds_use_prior_market_history_without_overlap() -> None:
    folds = pd.read_parquet(v7.FOLD_PATH)
    valid = folds[folds["valid_fold"].astype(bool)].copy()
    for column in [
        "training_start",
        "training_end",
        "calibration_start",
        "calibration_end",
        "test_start",
        "test_end",
    ]:
        valid[column] = pd.to_datetime(valid[column])
    assert valid["continuous_market_history"].astype(bool).all()
    assert (valid["training_start"] <= valid["training_end"]).all()
    assert (valid["training_end"] < valid["calibration_start"]).all()
    assert (valid["calibration_end"] < valid["test_start"]).all()
    assert (valid["test_start"] <= valid["test_end"]).all()
    assert (valid["training_dates"] >= 60).all()
    assert (valid["calibration_dates"] == 20).all()
    assert (valid["test_dates"] == 20).all()
    idx = valid[valid["index_id"].eq("IDX30")]
    interval = pd.read_csv(v7.INTERVAL_INVENTORY_PATH)
    starts = interval.set_index("interval_id")["interval_start"]
    assert any(
        row.training_start.date() < pd.Timestamp(starts[row.interval_id]).date()
        for row in idx.itertuples()
    )
    for _, group in valid.groupby("index_id"):
        ordered = group.sort_values("test_start")
        assert (
            ordered["test_start"].iloc[1:].to_numpy()
            > ordered["test_end"].iloc[:-1].to_numpy()
        ).all()


def test_features_targets_and_labels_are_strictly_ordered() -> None:
    features = pd.read_parquet(v7.FEATURE_PATH)
    collapsed = pd.read_parquet(v7.COLLAPSED_PATH)
    assert (
        pd.to_datetime(features["target_date"])
        > pd.to_datetime(features["prediction_date"])
    ).all()
    assert (
        pd.to_datetime(features["feature_available_date"])
        == pd.to_datetime(features["prediction_date"])
    ).all()
    assert features[v7.FEATURE_COLUMNS].notna().all().all()
    assert (
        pd.to_datetime(collapsed["label_maturity_cutoff"])
        < pd.to_datetime(collapsed["prediction_date"])
    ).all()
    assert (
        pd.to_datetime(collapsed["target_date"])
        > pd.to_datetime(collapsed["prediction_date"])
    ).all()


def test_no_security_date_target_appears_in_train_and_test() -> None:
    features = pd.read_parquet(v7.FEATURE_PATH)
    predictions = pd.read_parquet(v7.COLLAPSED_PATH)
    features["target_date"] = pd.to_datetime(features["target_date"])
    predictions["prediction_date"] = pd.to_datetime(predictions["prediction_date"])
    predictions["label_maturity_cutoff"] = pd.to_datetime(
        predictions["label_maturity_cutoff"]
    )
    for (fold_id, semantics), test in predictions.groupby(
        ["fold_id", "universe_semantics"], sort=False
    ):
        cutoff = test["label_maturity_cutoff"].iloc[0]
        index_id = test["index_id"].iloc[0]
        train = features[
            features["index_id"].eq(index_id)
            & features["universe_semantics"].eq(semantics)
            & (features["target_date"] <= cutoff)
        ]
        train_keys = set(
            train[["security_id", "prediction_date"]]
            .assign(prediction_date=lambda x: pd.to_datetime(x["prediction_date"]))
            .itertuples(index=False, name=None)
        )
        test_keys = set(
            test[["security_id", "prediction_date"]].itertuples(
                index=False, name=None
            )
        )
        assert train_keys.isdisjoint(test_keys), fold_id


def test_psei_recovery_and_mapping_rows_are_structurally_valid() -> None:
    raw = pd.read_parquet(v7.RECOVERED_RAW_PATH)
    recovered = raw[
        raw["provider"].eq("repository_carlo01092_investagrams_archive")
    ]
    assert len(recovered) > 0
    valid = v7.valid_ohlc_mask(recovered)
    assert valid.all()
    assert not recovered.duplicated(["market", "security_id", "date"]).any()
    ac = recovered[recovered["ticker_used"].eq("AC")]
    assert len(ac) > 0
    assert ac["security_id"].eq(v7.v6.security_id("PSEI", "AC")).all()
    assert recovered["price_semantics"].eq("RAW_OHLC").all()


def test_psei_and_fbm_current_symbol_mappings_are_retained() -> None:
    frames = v7.current_universe_frames()
    raw = pd.read_parquet(v7.RECOVERED_RAW_PATH)
    assert len(frames["PSEI"]) == 30
    assert len(frames["FBM_KLCI"]) == 30
    malaysia_ids = {
        v7.v6.security_id("FBM_KLCI", ticker)
        for ticker in frames["FBM_KLCI"]["local_symbol"].astype(str)
    }
    raw_malaysia_ids = set(
        raw.loc[raw["market"].eq("Malaysia"), "security_id"].astype(str)
    )
    assert malaysia_ids <= raw_malaysia_ids


def test_idx30_becomes_valid_under_continuous_history() -> None:
    cells = pd.read_parquet(v7.ALL_CELLS_PATH)
    idx = cells[cells["index_id"].eq("IDX30")]
    assert len(idx) == 12
    assert idx["valid"].astype(bool).all()
    assert (idx["scheduled_valid_folds"] >= 2).all()
    assert (idx["oos_trading_dates"] >= 40).all()
    assert (idx["oos_security_date_predictions"] >= 800).all()


def test_market_cell_gates_and_invalid_retention() -> None:
    cells = pd.read_parquet(v7.ALL_CELLS_PATH)
    invalid = pd.read_csv(PIT / "reports" / "pit_multimarket_invalid_cells_v7.csv")
    assert len(cells) == 5 * 3 * 4
    valid = cells[cells["valid"].astype(bool)]
    assert (valid["scheduled_valid_folds"] >= 2).all()
    assert (valid["prediction_folds"] >= 2).all()
    assert (valid["oos_trading_dates"] >= 40).all()
    assert (valid["oos_security_date_predictions"] >= 800).all()
    assert (valid["target_class_count"] == 2).all()
    assert len(invalid) == int((~cells["valid"].astype(bool)).sum())
    assert set(invalid["index_id"]) >= {"FBM_KLCI"}


def test_seed_collapse_keys_probabilities_and_counts() -> None:
    seed = pd.read_parquet(v7.SEED_PATH)
    collapsed = pd.read_parquet(v7.COLLAPSED_PATH)
    assert set(seed["seed"].astype(int)) == set(v7.SEEDS)
    assert collapsed["seed_count"].eq(len(v7.SEEDS)).all()
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
    means = seed.groupby(key, sort=True)["prediction_probability"].mean()
    observed = collapsed.set_index(key)["prediction_probability"]
    assert np.allclose(means.sort_index(), observed.sort_index())


def test_bootstrap_and_bh_are_reproducible() -> None:
    vector = np.linspace(-0.01, 0.02, 80)
    first = analytics.moving_block_means(
        vector, samples=500, block_length=20, seed=271828
    )
    second = analytics.moving_block_means(
        vector, samples=500, block_length=20, seed=271828
    )
    assert np.array_equal(first, second)
    adjusted = analytics.benjamini_hochberg([0.04, 0.001, 0.02, np.nan])
    assert np.isnan(adjusted[-1])
    assert np.all((adjusted[:3] >= np.array([0.04, 0.001, 0.02])))
    inference = pd.read_csv(
        PIT / "reports" / "pit_multimarket_bootstrap_inference_v7.csv"
    )
    if len(inference):
        valid = inference[inference["valid"].astype(str).str.lower().eq("true")]
        tested = valid[valid["raw_p_value"].notna()]
        assert len(tested)
        assert tested["adjusted_p_value"].notna().all()
    assert set(inference["block_length"].dropna().astype(int)) <= {10, 20, 40}


def test_manifest_is_repository_relative_and_all_checksums_validate() -> None:
    manifest = json.loads(v7.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["experiment_id"] == v7.EXPERIMENT_ID
    assert manifest["manifest_version"] == v7.MANIFEST_VERSION
    assert manifest["artifact_count"] == len(manifest["records"])
    assert manifest["artifact_count"] >= 30
    for record in manifest["records"]:
        relative = Path(record["artifact_relative_path"])
        assert not relative.is_absolute()
        path = ROOT / relative
        assert path.is_file()
        assert _sha256(path) == record["checksum"]


def test_required_v7_outputs_and_status_are_consistent() -> None:
    for path in v7.required_artifact_paths():
        assert path.is_file(), path
    status = json.loads(v7.STATUS_JSON_PATH.read_text(encoding="utf-8"))
    inventory = pd.read_csv(v7.INVENTORY_PATH)
    assert status["experiment_id"] == v7.EXPERIMENT_ID
    assert set(inventory["index_id"]) == set(v7.MARKETS)
    assert status["status"] in {
        "COMPLETE_FIVE_MARKET",
        "PARTIAL_DATA_BLOCKED",
        "PARTIAL_PROTOCOL_BLOCKED",
        "PARTIAL_DATA_AND_PROTOCOL_BLOCKED",
        "FAILED_INTEGRITY",
    }
    assert status["v6_integrity"]["v6_manifest_checksums_valid"] is True
    assert status["v6_integrity"]["v4_locked_checksums_valid"] is True
