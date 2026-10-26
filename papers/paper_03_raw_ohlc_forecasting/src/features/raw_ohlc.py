"""Leakage-audited helpers for common raw-OHLC experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


RAW_OHLC_COLUMNS = ("open_raw", "high_raw", "low_raw", "close_raw")


def raw_ohlc_valid_mask(frame: pd.DataFrame) -> pd.Series:
    """Return rows with finite, strictly positive, structurally valid raw OHLC."""
    values = frame.loc[:, RAW_OHLC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=frame.index)
    valid &= values.gt(0.0).all(axis=1)
    valid &= values["high_raw"] >= values[["open_raw", "low_raw", "close_raw"]].max(axis=1)
    valid &= values["low_raw"] <= values[["open_raw", "high_raw", "close_raw"]].min(axis=1)
    if "is_missing_bar" in frame:
        valid &= ~frame["is_missing_bar"].fillna(False).astype(bool)
    return valid


def build_lagged_raw_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build features available by the close of session ``t``.

    Features are lagged through the prior observed session.  The label is the
    next observed session's open-to-close direction, so the current row is the
    forecast timestamp and the future row is used only for ``y_true``.
    """
    required = {"market", "symbol_canonical", "date", *RAW_OHLC_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw-OHLC feature input missing columns: {sorted(missing)}")
    data = frame.sort_values(["market", "symbol_canonical", "date"]).copy()
    for column in RAW_OHLC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    key = [data["market"], data["symbol_canonical"]]
    grouped = data.groupby(["market", "symbol_canonical"], sort=False, group_keys=False)
    open_close = data["close_raw"] / data["open_raw"] - 1.0
    close_close = grouped["close_raw"].pct_change(fill_method=None)
    overnight = data["open_raw"] / grouped["close_raw"].shift(1) - 1.0
    intraday_range = (data["high_raw"] - data["low_raw"]) / data["open_raw"]
    features = pd.DataFrame(index=data.index)
    features["lagged_open_close_return"] = open_close.groupby(key, sort=False).shift(1)
    features["lagged_close_close_return"] = close_close.groupby(key, sort=False).shift(1)
    features["lagged_overnight_gap"] = overnight.groupby(key, sort=False).shift(1)
    features["lagged_high_low_range"] = intraday_range.groupby(key, sort=False).shift(1)
    features["lagged_rolling_volatility_5"] = close_close.groupby(key, sort=False).transform(lambda values: values.rolling(5, min_periods=5).std().shift(1))
    features["lagged_momentum_5"] = data["close_raw"].groupby(key, sort=False).transform(lambda values: values.pct_change(5, fill_method=None).shift(1))
    features["lagged_downside_rate_5"] = (open_close < 0).astype(float).groupby(key, sort=False).transform(lambda values: values.rolling(5, min_periods=5).mean().shift(1))

    daily = data.assign(_close_close=close_close, _open_close=open_close).groupby(["market", "date"], sort=False).agg(
        _market_return=("_close_close", "mean"),
        _market_breadth=("_open_close", lambda values: float((values > 0).mean())),
    ).reset_index()
    daily["lagged_market_return"] = daily.groupby("market", sort=False)["_market_return"].shift(1)
    daily["lagged_market_breadth"] = daily.groupby("market", sort=False)["_market_breadth"].shift(1)
    market_features = data[["market", "date"]].merge(daily[["market", "date", "lagged_market_return", "lagged_market_breadth"]], on=["market", "date"], how="left", sort=False)
    features["lagged_market_return"] = market_features["lagged_market_return"].to_numpy()
    features["lagged_market_breadth"] = market_features["lagged_market_breadth"].to_numpy()

    future_open = grouped["open_raw"].shift(-1)
    future_close = grouped["close_raw"].shift(-1)
    target_return = future_close / future_open - 1.0
    data["target_date"] = grouped["date"].shift(-1)
    data["target_open_close_return"] = target_return
    data["y_true"] = np.where(target_return.notna(), (target_return > 0).astype("int8"), np.nan)
    data["current_overnight_gap"] = overnight
    data["current_intraday_range"] = intraday_range
    feature_columns = list(features.columns)
    for column in feature_columns:
        data[column] = features[column].to_numpy()
    if any("volume" in column or "adj" in column.lower() for column in feature_columns):
        raise AssertionError("raw-OHLC registry contains a prohibited volume/adjusted feature")
    return data, feature_columns


def build_market_state_observables(frame: pd.DataFrame) -> pd.DataFrame:
    """Return market-level observables available at the close of each date.

    Thresholds are deliberately not assigned here.  The experiment runner fits
    state thresholds on each training fold and applies them to its test rows.
    """
    data = frame.sort_values(["market", "symbol_canonical", "date"]).copy()
    for column in RAW_OHLC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    grouped = data.groupby(["market", "symbol_canonical"], sort=False, group_keys=False)
    data["_close_close_t"] = grouped["close_raw"].pct_change(fill_method=None)
    data["_open_close_t"] = data["close_raw"] / data["open_raw"] - 1.0
    daily = data.groupby(["market", "date"], as_index=False).agg(
        market_return_t=("_close_close_t", "mean"),
        market_breadth_t=("_open_close_t", lambda values: float((values > 0).mean())),
        downside_intensity_t=("_open_close_t", lambda values: float(np.minimum(values.to_numpy(dtype=float), 0.0).mean())),
    )
    daily = daily.sort_values(["market", "date"])
    daily["market_volatility_20_t"] = daily.groupby("market", sort=False)["market_return_t"].transform(
        lambda values: values.rolling(20, min_periods=10).std()
    )
    daily["market_trend_20_t"] = daily.groupby("market", sort=False)["market_return_t"].transform(
        lambda values: values.rolling(20, min_periods=10).mean()
    )
    daily["market_proxy_t"] = daily.groupby("market", sort=False)["market_return_t"].transform(
        lambda values: (1.0 + values.fillna(0.0)).cumprod()
    )
    daily["market_drawdown_t"] = daily["market_proxy_t"] / daily.groupby("market", sort=False)["market_proxy_t"].transform(
        lambda values: values.rolling(60, min_periods=20).max()
    ) - 1.0
    return daily[[
        "market", "date", "market_return_t", "market_breadth_t", "downside_intensity_t",
        "market_volatility_20_t", "market_trend_20_t", "market_drawdown_t",
    ]]


def known_action_dates(path: str | None) -> set[tuple[str, str, str]]:
    """Read only action/anomaly labels; never consume adjusted prices."""
    if not path:
        return set()
    audit = pd.read_csv(path, dtype=str, keep_default_na=False)
    if audit.empty:
        return set()
    selected = audit[audit["anomaly_type"].str.contains("dividend|stock_split", case=False, regex=True)]
    return {(str(row.market), str(row.symbol_canonical), str(row.date)) for row in selected.itertuples(index=False)}
