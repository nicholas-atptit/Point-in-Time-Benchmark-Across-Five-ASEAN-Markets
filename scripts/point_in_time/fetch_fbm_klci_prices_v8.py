#!/usr/bin/env python3
"""Acquire and audit raw Yahoo chart OHLC for historical FBM KLCI members.

This helper is intentionally narrow:

* it requests only Bursa stock-code ``.KL`` symbols needed by the V8
  reconstruction;
* it preserves the provider response alongside a normalized raw-OHLC CSV;
* it never reads or emits adjusted close or volume as model inputs;
* it performs no interpolation, forward fill, or imputation.

The Yahoo chart endpoint is a fallback public source already used elsewhere in
the repository.  V8 keeps the response and request metadata so the acquisition
can be independently audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
SOURCE_ROOT = (
    REPO_ROOT / "research_data" / "point_in_time" / "sources" / "fbm_klci_v8"
)
PRICE_ROOT = SOURCE_ROOT / "prices"
MANIFEST_CSV = SOURCE_ROOT / "fbm_klci_price_acquisition_v8.csv"
MANIFEST_JSON = SOURCE_ROOT / "fbm_klci_price_acquisition_v8.json"

DEFAULT_CODES = [
    "0166",
    "1015",
    "3034",
    "3182",
    "4715",
    "5014",
    "5168",
    "5235SS",
    "5246",
    "6399",
    "7106",
    "7113",
    "7277",
]
START = date(2017, 11, 1)
END = date(2026, 7, 1)
PROVIDER = "Yahoo Finance chart API"
ENDPOINT = "https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
USER_AGENT = "Mozilla/5.0 (compatible; ASEAN-PIT-V8-research/1.0)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unix_seconds(value: date) -> int:
    return int(
        datetime.combine(value, datetime_time.min, tzinfo=timezone.utc).timestamp()
    )


def request_url(symbol: str, start: date, end: date) -> str:
    # Yahoo period2 is exclusive.  The caller supplies an inclusive research end.
    period2 = unix_seconds(end) + 86_400
    query = urllib.parse.urlencode(
        {
            "period1": unix_seconds(start),
            "period2": period2,
            "interval": "1d",
            "events": "history",
        }
    )
    return f"{ENDPOINT.format(symbol=urllib.parse.quote(symbol))}?{query}"


def valid_ohlc_mask(frame: pd.DataFrame) -> pd.Series:
    numeric = frame[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    return (
        numeric.notna().all(axis=1)
        & (numeric > 0).all(axis=1)
        & (numeric["high"] >= numeric[["open", "low", "close"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "high", "close"]].min(axis=1))
    )


def parse_chart(payload: dict[str, Any], symbol: str) -> pd.DataFrame:
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo chart error for {symbol}: {error}")
    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"Yahoo returned no chart result for {symbol}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = ((result.get("indicators") or {}).get("quote") or [])
    if not quotes:
        raise RuntimeError(f"Yahoo returned no raw quote block for {symbol}")
    quote = quotes[0]
    lengths = [
        len(timestamps),
        *[len(quote.get(field) or []) for field in ("open", "high", "low", "close")],
    ]
    row_count = min(lengths)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                timestamps[:row_count], unit="s", utc=True, errors="coerce"
            ).date,
            "open": (quote.get("open") or [])[:row_count],
            "high": (quote.get("high") or [])[:row_count],
            "low": (quote.get("low") or [])[:row_count],
            "close": (quote.get("close") or [])[:row_count],
        }
    )
    frame["provider_symbol"] = symbol
    return frame


def fetch_one(
    code: str,
    *,
    start: date,
    end: date,
    refresh: bool,
) -> dict[str, Any]:
    symbol = f"{code}.KL"
    raw_path = PRICE_ROOT / f"yahoo_chart_{symbol}_{start}_{end}.json"
    csv_path = PRICE_ROOT / f"raw_ohlc_{symbol}_{start}_{end}.csv"
    url = request_url(symbol, start, end)
    retrieval_timestamp = datetime.now(timezone.utc).isoformat()
    status = "FETCHED"
    error_message = ""

    if raw_path.is_file() and not refresh:
        raw_bytes = raw_path.read_bytes()
        status = "REUSED_CACHED_RESPONSE"
    else:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            return {
                "bursa_code": code,
                "provider_symbol": symbol,
                "request_start": start.isoformat(),
                "request_end_inclusive": end.isoformat(),
                "request_url": url,
                "retrieval_timestamp": retrieval_timestamp,
                "status": f"HTTP_{exc.code}",
                "raw_response_path": "",
                "normalized_path": "",
                "raw_sha256": "",
                "normalized_sha256": "",
                "source_rows": 0,
                "accepted_rows": 0,
                "duplicate_dates": 0,
                "invalid_ohlc_rows": 0,
                "first_date": "",
                "last_date": "",
                "price_semantics": "RAW_OHLC_ONLY",
                "adjusted_close_used": False,
                "imputation_used": False,
                "error_message": str(exc),
            }
        raw_path.write_bytes(raw_bytes)

    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
        frame = parse_chart(payload, symbol)
        source_rows = len(frame)
        duplicate_dates = int(frame.duplicated(["date"], keep=False).sum())
        frame = frame.drop_duplicates(
            ["date", "open", "high", "low", "close"], keep="first"
        )
        valid = valid_ohlc_mask(frame)
        invalid_rows = int((~valid).sum())
        frame = frame[valid].sort_values("date", kind="stable").reset_index(drop=True)
        if frame["date"].duplicated().any():
            conflicting = frame.loc[frame["date"].duplicated(False), "date"].unique()
            raise RuntimeError(
                f"conflicting duplicate Yahoo dates for {symbol}: {conflicting[:5]}"
            )
        frame.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    except Exception as exc:
        error_message = str(exc)
        status = "PARSE_OR_VALIDATION_FAILED"
        frame = pd.DataFrame()
        source_rows = 0
        duplicate_dates = 0
        invalid_rows = 0

    return {
        "bursa_code": code,
        "provider_symbol": symbol,
        "request_start": start.isoformat(),
        "request_end_inclusive": end.isoformat(),
        "request_url": url,
        "retrieval_timestamp": retrieval_timestamp,
        "status": status,
        "raw_response_path": (
            raw_path.relative_to(REPO_ROOT).as_posix() if raw_path.is_file() else ""
        ),
        "normalized_path": (
            csv_path.relative_to(REPO_ROOT).as_posix() if csv_path.is_file() else ""
        ),
        "raw_sha256": sha256_file(raw_path) if raw_path.is_file() else "",
        "normalized_sha256": sha256_file(csv_path) if csv_path.is_file() else "",
        "source_rows": source_rows,
        "accepted_rows": len(frame),
        "duplicate_dates": duplicate_dates,
        "invalid_ohlc_rows": invalid_rows,
        "first_date": str(frame["date"].min()) if not frame.empty else "",
        "last_date": str(frame["date"].max()) if not frame.empty else "",
        "price_semantics": "RAW_OHLC_ONLY",
        "adjusted_close_used": False,
        "imputation_used": False,
        "error_message": error_message,
    }


def run(
    codes: Sequence[str],
    *,
    start: date = START,
    end: date = END,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    PRICE_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for position, code in enumerate(codes):
        rows.append(fetch_one(code, start=start, end=end, refresh=refresh))
        if position + 1 < len(codes):
            time.sleep(0.25)
    frame = pd.DataFrame(rows)
    frame.to_csv(MANIFEST_CSV, index=False, encoding="utf-8", lineterminator="\n")
    payload = {
        "acquisition_id": "FBM_KLCI_V8_HISTORICAL_RAW_OHLC",
        "provider": PROVIDER,
        "endpoint": ENDPOINT,
        "requested_codes": list(codes),
        "request_start": start.isoformat(),
        "request_end_inclusive": end.isoformat(),
        "adjusted_close_used": False,
        "volume_used_as_feature": False,
        "forward_fill_used": False,
        "imputation_used": False,
        "records": rows,
    }
    MANIFEST_JSON.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", action="append", dest="codes")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = run(args.codes or DEFAULT_CODES, refresh=args.refresh)
    failures = [row for row in rows if not str(row["status"]).startswith(("FETCHED", "REUSED"))]
    print(
        json.dumps(
            {
                "symbols": len(rows),
                "accepted_symbols": len(rows) - len(failures),
                "accepted_rows": sum(int(row["accepted_rows"]) for row in rows),
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
