# Data dictionary

The full field-level schema and measured null counts are in `audit/data_schema.csv`. Missing labels and invalid folds remain missing; missing values are not zero observations.

## pit_historical_raw_ohlc_recovered_v8.parquet
Rows: 604,422; fields: 14; key: `['market', 'security_id', 'date']`; duplicates: 0.
Observed range: 1985-01-03 to 2026-07-01. This is file coverage, not a claim that all dates are forecast-ready.

## pit_membership_daily_all_arms_v8.parquet
Rows: 897,141; fields: 9; key: `['index_id', 'universe_semantics', 'security_id', 'membership_date']`; duplicates: 0.
Observed range: 2016-01-25 to 2026-07-01. This is file coverage, not a claim that all dates are forecast-ready.

## pit_membership_intervals_v8.parquet
Rows: 3,060; fields: 15; key: `['interval_id', 'security_id']`; duplicates: 0.
Observed range: 2016-01-25 to 2026-05-04. This is file coverage, not a claim that all dates are forecast-ready.

## pit_multimarket_feature_panel_v8.parquet
Rows: 819,427; fields: 36; key: `['index_id', 'universe_semantics', 'security_id', 'prediction_date']`; duplicates: 0.
Observed range: 2016-01-25 to 2026-06-30. This is file coverage, not a claim that all dates are forecast-ready.

## pit_multimarket_fold_calendar_v8.parquet
Rows: 239; fields: 17; key: `['fold_id']`; duplicates: 0.
Observed range: None to None. This is file coverage, not a claim that all dates are forecast-ready.

## pit_multimarket_observation_panel_v8.parquet
Rows: 820,557; fields: 22; key: `['index_id', 'universe_semantics', 'security_id', 'prediction_date']`; duplicates: 0.
Observed range: 2016-01-25 to 2026-06-30. This is file coverage, not a claim that all dates are forecast-ready.

## pit_multimarket_predictions_seed_collapsed_v8.parquet
Rows: 1,139,196; fields: 28; key: `['interval_id', 'universe_semantics', 'fold_id', 'security_id', 'prediction_date', 'target_date', 'model_id', 'seed']`; duplicates: 0.
Observed range: 2017-07-11 to 2026-06-29. This is file coverage, not a claim that all dates are forecast-ready.

## pit_multimarket_predictions_seed_level_v8.parquet
Rows: 5,695,980; fields: 27; key: `['interval_id', 'universe_semantics', 'fold_id', 'security_id', 'prediction_date', 'target_date', 'model_id', 'seed']`; duplicates: 0.
Observed range: 2017-07-11 to 2026-06-29. This is file coverage, not a claim that all dates are forecast-ready.

## fbm_klci_membership_daily_v8.parquet
Rows: 62,460; fields: 9; key: `['index_id', 'universe_semantics', 'security_id', 'membership_date']`; duplicates: 0.
Observed range: 2018-01-02 to 2026-07-01. This is file coverage, not a claim that all dates are forecast-ready.

## fbm_klci_membership_intervals_v8.parquet
Rows: 390; fields: 16; key: `['interval_id', 'security_id']`; duplicates: 0.
Observed range: 2018-01-01 to 2025-06-23. This is file coverage, not a claim that all dates are forecast-ready.
