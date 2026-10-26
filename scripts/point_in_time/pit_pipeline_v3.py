#!/usr/bin/env python3
"""Additive v3 PIT execution layer.

The v2 source-acquisition artifacts are the sole membership evidence input to
this module. v3 never turns a snapshot-only event into an interval. It adds
raw-price acquisition records, deterministic identity-review records, strict
interval eligibility, and labelled PIT/current panels without mutating any v1
or v2 file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(SCRIPT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(SCRIPT_ROOT))
import pit_pipeline_v2 as v2  # noqa: E402

PIPELINE_VERSION = "3.0.0"
MEMBERSHIP_VERSION = "pit-membership-v3"
SNAPSHOT_VERSION = "pit-universe-v3"
DATA_VERSION = "pit-data-v3"
TARGET_START = date(2016, 1, 1)
TARGET_END = date(2026, 6, 1)
PIT_SEMANTICS = "POINT_IN_TIME"
CURRENT_SEMANTICS = "CURRENT_2026_APPLIED_BACKWARD"
UTC = timezone.utc
RAW_SEMANTICS = "RAW_OHLC"

INTERVAL_COLUMNS = [
    "index_id", "market", "security_id", "ticker_on_effective_date",
    "effective_from", "effective_to", "event_type", "announcement_date",
    "source_chain", "anchor_source_id", "reconstruction_method",
    "event_chain_complete", "identity_resolved", "confidence",
    "resolution_status", "blocking_reason", "membership_version",
]
AUDIT_COLUMNS = [
    "security_id", "published_name", "historical_ticker", "membership_start",
    "membership_end", "event_chain_complete", "identity_resolved",
    "local_raw_price_available", "active_member_days", "covered_member_days",
    "coverage_ratio", "blocking_reason",
]
NAME_ONLY_COLUMNS = [
    "candidate_id", "market", "published_name", "review_period", "source_id",
    "candidate_security_id", "candidate_ticker", "match_method",
    "identifier_evidence", "date_consistency", "exchange_consistency",
    "corporate_action_evidence", "confidence", "resolution_status",
    "manual_review_required", "notes",
]
RAW_LOG_COLUMNS = [
    "market", "security_id", "ticker_used", "exchange_suffix", "provider",
    "request_start", "request_end", "retrieval_timestamp", "retrieval_status",
    "rows_received", "date_min", "date_max", "membership_days_required",
    "membership_days_covered", "coverage_ratio", "failure_code",
    "fallback_attempted", "raw_file_path", "checksum",
]


def now_utc() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def as_date(value: Any) -> date | None:
    text = iso(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    payload = "|".join("" if value is None else str(value) for value in parts)
    return f"{prefix}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length]}"


def ensure_dirs(root: Path) -> None:
    for relative in ("processed", "reconciliation", "reports", "snapshots", "qa", "manifests", "raw_prices"):
        (root / relative).mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_parquet(path: Path, frame: pd.DataFrame, columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame.reindex(columns=list(columns))
    frame.to_parquet(path, index=False, engine="pyarrow")


def read_parquet(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return iso(value)
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def artifact_manifest(root: Path, path: Path, purpose: str, record_count: int | None = None) -> Path:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        # The durable status reports live at repository-level reports/ rather
        # than inside research_data/point_in_time. Keep their manifest in the
        # v3 manifest directory while preserving the explicit relative path.
        relative = os.path.relpath(path, root).replace(os.sep, "/")
    manifest_path = root / "manifests" / f"v3__{_safe_filename(relative.replace('/', '__'))}.manifest.json"
    write_json(manifest_path, {
        "pipeline": "point_in_time_v3", "pipeline_version": PIPELINE_VERSION,
        "path": relative, "purpose": purpose, "format": path.suffix.lstrip("."),
        "sha256": sha256_file(path), "record_count": record_count, "created_at": now_utc(),
    })
    return manifest_path


def current_tickers(repo_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "VN30": repo_root / "config" / "vn30_constituents_2026_06.csv",
        "SET50": repo_root / "config" / "benchmark_universe" / "thailand_set50_2026_06.csv",
        "IDX30": repo_root / "config" / "benchmark_universe" / "indonesia_idx30_2026_06.csv",
        "FBM_KLCI": repo_root / "config" / "benchmark_universe" / "malaysia_klci_2026_06.csv",
        "PSEI": repo_root / "config" / "benchmark_universe" / "philippines_psei_2026_06.csv",
    }
    result: dict[str, pd.DataFrame] = {}
    for index_id, path in paths.items():
        if not path.is_file():
            result[index_id] = pd.DataFrame(columns=["local_symbol", "company_name"])
            continue
        frame = pd.read_csv(path, dtype=str).fillna("")
        if "local_symbol" not in frame.columns:
            frame["local_symbol"] = frame.iloc[:, 0].astype(str)
        if "company_name" not in frame.columns:
            frame["company_name"] = ""
        result[index_id] = frame
    return result


def _source_period_map(root: Path) -> dict[str, str]:
    source = read_parquet(root, "processed/source_evidence_v2.parquet")
    result: dict[str, str] = {}
    for row in source.to_dict("records"):
        source_id, period = str(row.get("source_id") or ""), str(row.get("review_period") or "")
        if source_id and period and period not in {"ARCHIVE", "None", "nan"}:
            result[source_id] = period
    return result


def _source_quality_map(root: Path) -> dict[str, str]:
    source = read_parquet(root, "processed/source_evidence_v2.parquet")
    return {str(row.get("source_id")): str(row.get("source_quality") or "") for row in source.to_dict("records") if row.get("source_id")}


def build_name_only_resolution(root: Path) -> pd.DataFrame:
    candidates = read_parquet(root, "reconciliation/entity_resolution_candidates_v2.parquet")
    events = read_parquet(root, "interim/membership_events_extracted_v2.parquet")
    unresolved = candidates[candidates.get("decision", pd.Series(dtype=str)).astype(str).eq("UNRESOLVED_NAME_ONLY_OFFICIAL_EVENT")].copy()
    event_map = {str(row.get("source_record")): row for row in events.to_dict("records")}
    rows: list[dict[str, Any]] = []
    for record in unresolved.to_dict("records"):
        event = event_map.get(str(record.get("source_record")), {})
        rows.append({
            "candidate_id": record.get("candidate_id"), "market": record.get("market"),
            "published_name": record.get("security_name_as_published"), "review_period": event.get("review_period"),
            "source_id": event.get("source_id"), "candidate_security_id": None, "candidate_ticker": None,
            "match_method": "MANUAL_UNRESOLVED_NAME_ONLY",
            "identifier_evidence": "Retained event source publishes a name but no official exchange code, ISIN, or dated ticker; the current June-2026 comparator was not used as historical identity evidence.",
            "date_consistency": bool(record.get("date_consistency")), "exchange_consistency": bool(record.get("exchange_consistency")),
            "corporate_action_evidence": f"{event.get('event_type') or 'membership'} event retained from source {event.get('source_id') or 'unknown'}; no dated corporate-action identity link was retained.",
            "confidence": 0.0, "resolution_status": "UNRESOLVED_MANUAL_REVIEW", "manual_review_required": True,
            "notes": "No automatic fuzzy-name match. The record remains in the unresolved queue and blocks only the affected FBM KLCI interval; no membership interval was dropped or inferred.",
        })
    return pd.DataFrame(rows, columns=NAME_ONLY_COLUMNS).sort_values(["market", "review_period", "published_name", "candidate_id"], kind="stable").reset_index(drop=True)


def _ticker_for(security_id: str, on_date: date, ticker_rows: pd.DataFrame) -> str | None:
    if ticker_rows.empty:
        return None
    starts = pd.to_datetime(ticker_rows["valid_from"], errors="coerce").dt.date
    ends = pd.to_datetime(ticker_rows["valid_to"], errors="coerce").dt.date
    rows = ticker_rows[ticker_rows["security_id"].astype(str).eq(str(security_id)) & (starts <= on_date) & (ends >= on_date)]
    return (str(rows.iloc[0].get("ticker") or "") or None) if not rows.empty else None


def build_membership_v3(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    old = read_parquet(root, "processed/index_membership_v2.parquet")
    tickers = read_parquet(root, "processed/ticker_history_v2.parquet")
    securities = read_parquet(root, "processed/security_master_v2.parquet")
    reconciliation = read_parquet(root, "reconciliation/reconciliation_log_v2.parquet")
    period_by_source = _source_period_map(root)
    conflict_by_source: dict[str, list[str]] = defaultdict(list)
    conflict_by_period: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in reconciliation.to_dict("records"):
        if str(row.get("severity") or "") != "HIGH" or str(row.get("status") or "") != "OPEN":
            continue
        index_id, details, source_id = str(row.get("index_id") or ""), str(row.get("details") or ""), str(row.get("source_id") or "")
        if source_id:
            conflict_by_source[source_id].append(details)
        for period in re.findall(r"\b20\d{2}[-A-Za-z0-9]+\b", details):
            conflict_by_period[(index_id, period)].append(details)
    rows: list[dict[str, Any]] = []
    for record in old.to_dict("records"):
        index_id, security_id = str(record.get("index_id") or ""), str(record.get("security_id") or "")
        start, end, source_id = as_date(record.get("effective_from")), as_date(record.get("effective_to")), str(record.get("source_id") or "")
        ticker, period, reasons = _ticker_for(security_id, start or TARGET_START, tickers), period_by_source.get(source_id, ""), []
        reasons.extend(conflict_by_source.get(source_id, []))
        reasons.extend(conflict_by_period.get((index_id, period), []))
        identity_resolved = str(record.get("resolution_status") or "").startswith("RESOLVED")
        if not bool(record.get("event_chain_complete")):
            reasons.append("v2 interval is not event-chain complete")
        if not identity_resolved:
            reasons.append("security identity is unresolved")
        rows.append({
            "index_id": index_id, "market": record.get("market"), "security_id": security_id,
            "ticker_on_effective_date": ticker, "effective_from": start, "effective_to": end,
            "event_type": record.get("event_type"), "announcement_date": as_date(record.get("announcement_date")),
            "source_chain": f"official:{source_id}" if source_id else None, "anchor_source_id": record.get("anchor_source_id") or source_id,
            "reconstruction_method": record.get("reconstruction_method"), "event_chain_complete": bool(record.get("event_chain_complete")),
            "identity_resolved": identity_resolved, "confidence": float(record.get("confidence") or 0.0),
            "resolution_status": record.get("resolution_status"), "blocking_reason": " | ".join(dict.fromkeys(reasons)) or None,
            "membership_version": MEMBERSHIP_VERSION,
        })
    frame = pd.DataFrame(rows, columns=INTERVAL_COLUMNS)
    return (frame.sort_values(["index_id", "effective_from", "security_id"], kind="stable").reset_index(drop=True) if not frame.empty else frame), tickers, securities, reconciliation


def _local_price_candidates(repo_root: Path, market: str, ticker: str) -> list[Path]:
    market_dir = {"Vietnam": "vietnam", "Thailand": "thailand", "Indonesia": "indonesia", "Malaysia": "malaysia", "Philippines": "philippines"}.get(market, market.lower())
    candidates: list[Path] = []
    yahoo_dir = repo_root / "data" / "cache" / "yahoo_daily_2016" / market_dir
    candidates.extend([yahoo_dir / f"{ticker}__{ticker}.csv", *sorted(yahoo_dir.glob(f"{_safe_filename(ticker)}__*.csv"))])
    if market == "Vietnam":
        candidates.append(repo_root / "data" / "cache" / "vnstock_daily_2016" / f"{ticker}.csv")
    raw_dir = repo_root / "data" / "raw" / "asean_eod" / market_dir
    for provider_dir in ("local_existing", "yahoo", "stooq"):
        candidates.extend(sorted((raw_dir / provider_dir).glob(f"{_safe_filename(ticker)}*.csv")))
    unique: list[Path] = []
    for path in candidates:
        if path.is_file() and path not in unique:
            unique.append(path)
    return unique


def _normalize_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    lower = {str(column).strip().lower(): column for column in frame.columns}
    date_col = next((lower[name] for name in ("date", "datetime", "timestamp", "time") if name in lower), None)
    cols = {name: lower.get(name) for name in ("open", "high", "low", "close")}
    if date_col is None or any(value is None for value in cols.values()):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    result = pd.DataFrame({"date": pd.to_datetime(frame[date_col], errors="coerce", utc=True).dt.date, **{name: pd.to_numeric(frame[column], errors="coerce") for name, column in cols.items()}})
    result = result.dropna(subset=["date", "open", "high", "low", "close"], how="any")
    return result.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _read_local_ohlc(path: Path) -> pd.DataFrame:
    try:
        return _normalize_ohlc(pd.read_csv(path))
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])


def _yahoo_url(symbol: str, start: date, end: date) -> str:
    period1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    period2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    return f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol, safe='')}?period1={period1}&period2={period2}&interval=1d&events=history"


def _fetch_yahoo(symbol: str, start: date, end: date, timeout: int, retries: int) -> tuple[str, bytes | None, pd.DataFrame, str]:
    url = _yahoo_url(symbol, start, end)
    last_error = ""
    for attempt in range(retries + 1):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; VSEF PIT raw-OHLC acquisition)", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            parsed = json.loads(payload.decode("utf-8"))
            chart = parsed.get("chart", {})
            if chart.get("error"):
                return "failed", payload, pd.DataFrame(), str(chart["error"])
            result = (chart.get("result") or [None])[0]
            timestamps = result.get("timestamp") or [] if result else []
            quote = ((result.get("indicators") or {}).get("quote") or [{}])[0] if result else {}
            if not timestamps or not quote:
                return "empty", payload, pd.DataFrame(), "no daily quote rows"
            frame = _normalize_ohlc(pd.DataFrame({"date": pd.to_datetime(timestamps, unit="s", utc=True).date, "open": quote.get("open", []), "high": quote.get("high", []), "low": quote.get("low", []), "close": quote.get("close", [])}))
            return ("success" if not frame.empty else "empty"), payload, frame, "" if not frame.empty else "no valid raw OHLC rows"
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP_{exc.code}"
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                return ("rate_limited" if exc.code == 429 else "failed"), None, pd.DataFrame(), last_error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}:{exc}"
            if attempt >= retries:
                return "provider_unavailable", None, pd.DataFrame(), last_error
        time.sleep(min(8.0, 1.5 * (2**attempt)))
    return "provider_unavailable", None, pd.DataFrame(), last_error or "retry_exhausted"


def _required_dates(group: pd.DataFrame) -> set[date]:
    required: set[date] = set()
    for row in group.to_dict("records"):
        start, end = as_date(row.get("required_price_start")) or as_date(row.get("membership_start")), as_date(row.get("required_price_end")) or as_date(row.get("membership_end"))
        if not start or not end:
            continue
        required.update(start + timedelta(days=offset) for offset in range((end - start).days + 1))
    return required


def _queue_requirements(root: Path) -> dict[str, dict[str, Any]]:
    queue = pd.read_csv(root / "reports" / "price_download_queue_by_provider.csv", dtype=str).fillna("")
    grouped: dict[str, dict[str, Any]] = {}
    for security_id, group in queue.groupby("security_id", sort=True):
        first = group.iloc[0].to_dict()
        dates = _required_dates(group)
        tickers, suffixes = sorted({str(value).strip() for value in group["historical_ticker"] if str(value).strip()}), sorted({str(value).strip() for value in group["exchange_suffix"] if str(value).strip()})
        if not tickers:
            continue
        start, end = min(dates) if dates else as_date(first.get("required_price_start")), max(dates) if dates else as_date(first.get("required_price_end"))
        grouped[str(security_id)] = {"market": first.get("market"), "security_id": str(security_id), "ticker": tickers[0], "exchange_suffix": suffixes[0] if suffixes else "", "request_start": start - timedelta(days=30) if start else None, "request_end": end, "required_dates": dates, "queue_rows": group}
    return grouped


def _raw_record_rows(frame: pd.DataFrame, *, market: str, security_id: str, ticker: str, suffix: str, provider: str, raw_path: str, retrieval_timestamp: datetime, checksum: str) -> list[dict[str, Any]]:
    return [{"market": market, "security_id": security_id, "ticker_used": ticker, "exchange_suffix": suffix, "date": as_date(row.get("date")), "open": float(row["open"]), "high": float(row["high"]), "low": float(row["low"]), "close": float(row["close"]), "provider": provider, "price_semantics": RAW_SEMANTICS, "raw_file_path": raw_path, "retrieval_timestamp": retrieval_timestamp, "checksum": checksum} for row in frame.to_dict("records") if as_date(row.get("date"))]


def _log_attempt(req: Mapping[str, Any], provider: str, status: str, frame: pd.DataFrame, *, raw_path: str | None = None, checksum: str | None = None, failure_code: str | None = None, fallback_attempted: bool = False, error: str = "") -> dict[str, Any]:
    required = set(req.get("required_dates") or set())
    available = {as_date(value) for value in frame.get("date", pd.Series(dtype=object)).tolist()} if not frame.empty else set()
    available = {value for value in available if value}
    covered = len(required.intersection(available))
    dates = sorted(available)
    return {"market": req.get("market"), "security_id": req.get("security_id"), "ticker_used": req.get("ticker"), "exchange_suffix": req.get("exchange_suffix"), "provider": provider, "request_start": iso(req.get("request_start")), "request_end": iso(req.get("request_end")), "retrieval_timestamp": now_utc(), "retrieval_status": status, "rows_received": int(len(frame)), "date_min": iso(min(dates)) if dates else None, "date_max": iso(max(dates)) if dates else None, "membership_days_required": len(required), "membership_days_covered": covered, "coverage_ratio": round(covered / len(required), 6) if required else 0.0, "failure_code": failure_code or error or None, "fallback_attempted": bool(fallback_attempted), "raw_file_path": raw_path, "checksum": checksum}


def acquire_raw_prices(repo_root: Path, root: Path, *, sleep_seconds: float = 1.0, timeout_seconds: int = 30, retries: int = 2) -> dict[str, Any]:
    """Process the existing queue and retain raw local/provider responses."""
    ensure_dirs(root)
    requirements = _queue_requirements(root)
    raw_rows: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    raw_dir = root / "raw_prices"
    for position, req in enumerate(requirements.values(), start=1):
        market, security_id, ticker, suffix = str(req.get("market") or ""), str(req.get("security_id") or ""), str(req.get("ticker") or ""), str(req.get("exchange_suffix") or "")
        request_start, request_end = req.get("request_start") or TARGET_START, req.get("request_end") or TARGET_END
        local_frame, local_path = pd.DataFrame(), None
        for candidate in _local_price_candidates(repo_root, market, ticker):
            candidate_frame = _read_local_ohlc(candidate)
            if not candidate_frame.empty:
                local_path, local_frame = candidate, candidate_frame
                break
        if local_path is not None:
            year_dir = raw_dir / _safe_filename(market) / str(request_start.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            local_copy = year_dir / f"local_existing_{_safe_filename(ticker)}_{local_path.name}"
            if not local_copy.exists() or sha256_file(local_copy) != sha256_file(local_path):
                shutil.copy2(local_path, local_copy)
            checksum = sha256_file(local_copy)
            provider = "local_vnstock" if "vnstock" in str(local_path) else "local_existing"
            raw_rows.extend(_raw_record_rows(local_frame, market=market, security_id=security_id, ticker=ticker, suffix=suffix, provider=provider, raw_path=str(local_copy.relative_to(root)), retrieval_timestamp=now_utc(), checksum=checksum))
            complete = set(req["required_dates"]).issubset(set(local_frame["date"]))
            logs.append(_log_attempt(req, provider, "success" if complete else "partial", local_frame, raw_path=str(local_copy.relative_to(root)), checksum=checksum, failure_code=None if complete else "LOCAL_DATE_GAP"))
        else:
            logs.append(_log_attempt(req, "local_existing", "missing", pd.DataFrame(), failure_code="LOCAL_RAW_OHLC_NOT_FOUND"))
        logs.append(_log_attempt(req, "EODHD", "not_attempted", pd.DataFrame(), failure_code="EODHD_CREDENTIALS_NOT_CONFIGURED", fallback_attempted=True))
        yahoo_symbol = f"{ticker}{suffix}" if suffix and not ticker.endswith(suffix) else ticker
        status, payload, yahoo_frame, error = _fetch_yahoo(yahoo_symbol, request_start, request_end, timeout_seconds, retries)
        raw_path, checksum = None, None
        if payload:
            year_dir = raw_dir / _safe_filename(market) / str(request_start.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            target = year_dir / f"yahoo_{_safe_filename(yahoo_symbol)}_{request_start.isoformat()}_{request_end.isoformat()}.json"
            target.write_bytes(payload)
            raw_path, checksum = str(target.relative_to(root)), sha256_bytes(payload)
            raw_rows.extend(_raw_record_rows(yahoo_frame, market=market, security_id=security_id, ticker=ticker, suffix=suffix, provider="yahoo", raw_path=raw_path, retrieval_timestamp=now_utc(), checksum=checksum))
        logs.append(_log_attempt(req, "yahoo", status, yahoo_frame, raw_path=raw_path, checksum=checksum, failure_code=None if status in {"success", "partial"} else error, fallback_attempted=True))
        if status not in {"success", "partial"} and local_frame.empty:
            unresolved_rows.append({"market": market, "security_id": security_id, "ticker_used": ticker, "exchange_suffix": suffix, "request_start": iso(request_start), "request_end": iso(request_end), "preferred_provider": "EODHD", "fallback_provider": "Yahoo Finance", "retrieval_status": status, "failure_reason": error or "No raw OHLC source returned rows; adjusted data not substituted."})
        if position < len(requirements):
            time.sleep(max(0.0, sleep_seconds))
    raw_frame = pd.DataFrame(raw_rows)
    if not raw_frame.empty:
        raw_frame = raw_frame.drop_duplicates(["security_id", "date", "provider"], keep="last").sort_values(["market", "security_id", "date", "provider"], kind="stable").reset_index(drop=True)
    else:
        raw_frame = pd.DataFrame(columns=["market", "security_id", "ticker_used", "exchange_suffix", "date", "open", "high", "low", "close", "provider", "price_semantics", "raw_file_path", "retrieval_timestamp", "checksum"])
    raw_path = root / "processed" / "historical_raw_ohlc_v3.parquet"
    write_parquet(raw_path, raw_frame)
    log_path, unresolved_path = root / "reports" / "raw_ohlc_acquisition_log_v3.csv", root / "reports" / "raw_ohlc_unresolved_v3.csv"
    write_csv(log_path, logs, RAW_LOG_COLUMNS)
    write_csv(unresolved_path, unresolved_rows, ["market", "security_id", "ticker_used", "exchange_suffix", "request_start", "request_end", "preferred_provider", "fallback_provider", "retrieval_status", "failure_reason"])
    artifact_manifest(root, raw_path, "Normalized raw OHLC rows retained from local files and provider responses", len(raw_frame))
    artifact_manifest(root, log_path, "Per-provider raw-OHLC acquisition attempts and checksums", len(logs))
    artifact_manifest(root, unresolved_path, "Securities with no usable raw OHLC after declared provider attempts", len(unresolved_rows))
    raw_files = [path for path in raw_dir.rglob("*") if path.is_file()]
    write_json(root / "manifests" / "v3__raw_prices.manifest.json", {"pipeline": "point_in_time_v3", "pipeline_version": PIPELINE_VERSION, "root": "raw_prices", "retrieved_at": now_utc(), "files": [{"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in sorted(raw_files)]})
    return {"queued_securities": len(requirements), "raw_rows": len(raw_frame), "successful_provider_attempts": int(sum(str(row.get("retrieval_status")) in {"success", "partial"} and str(row.get("provider")) == "yahoo" for row in logs)), "failed_downloads": len(unresolved_rows), "raw_path": str(raw_path)}


def build_unqueueable_report(root: Path) -> pd.DataFrame:
    queue = pd.read_csv(root / "reports" / "price_download_queue_by_provider.csv", dtype=str).fillna("")
    securities = read_parquet(root, "processed/security_master_v2.parquet")
    queued_ids = set(queue["security_id"].astype(str)) if "security_id" in queue else set()
    rows = [{"security_id": str(record.get("security_id")), "market": record.get("market"), "reason": "missing_verified_observation_date", "source_ids": None, "investigation": "Security exists in the v2 identity table but no dated membership observation or explicit interval endpoint was retained; a required price range cannot be invented.", "verified_ticker": record.get("official_exchange_code"), "verified_date_start": None, "verified_date_end": None, "resolution_status": "BLOCKED_UNQUEUEABLE", "blocking_reason": "No verified observation/effective date; not placed in provider queue."} for record in securities.to_dict("records") if str(record.get("security_id") or "") and str(record.get("security_id")) not in queued_ids]
    frame = pd.DataFrame(rows)
    path = root / "reports" / "unqueueable_security_resolution_v3.csv"
    columns = ["security_id", "market", "reason", "source_ids", "investigation", "verified_ticker", "verified_date_start", "verified_date_end", "resolution_status", "blocking_reason"]
    write_csv(path, rows, columns)
    artifact_manifest(root, path, "Review of identity records without a verified observation date", len(rows))
    return frame


def _raw_by_security_date(root: Path) -> pd.DataFrame:
    frame = read_parquet(root, "processed/historical_raw_ohlc_v3.parquet")
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.date
    frame = frame.dropna(subset=["security_id", "date", "open", "high", "low", "close"])
    frame["_priority"] = frame["provider"].astype(str).map({"local_existing": 0, "local_vnstock": 1, "yahoo": 2}).fillna(99)
    return frame.sort_values(["security_id", "date", "_priority", "provider"], kind="stable").drop_duplicates(["security_id", "date"], keep="first").drop(columns=["_priority"])


def _interval_price_stats(
    interval: Mapping[str, Any],
    memberships: pd.DataFrame,
    raw: pd.DataFrame,
    market_dates: Mapping[str, set[date]] | None = None,
    security_dates: Mapping[str, set[date]] | None = None,
) -> dict[str, Any]:
    start, end = as_date(interval.get("interval_start")), as_date(interval.get("interval_end"))
    if memberships.empty or raw.empty or not start or not end:
        return {"required": 0, "covered": 0, "ratio": 0.0, "member_stats": {}, "dates": set()}
    members = memberships[memberships["index_id"].astype(str).eq(str(interval.get("index_id"))) & (pd.to_datetime(memberships["effective_from"], errors="coerce").dt.date == start) & (pd.to_datetime(memberships["effective_to"], errors="coerce").dt.date == end)]
    if members.empty:
        return {"required": 0, "covered": 0, "ratio": 0.0, "member_stats": {}, "dates": set()}
    market = str(members.iloc[0].get("market") or "")
    if market_dates is None:
        market_dates = defaultdict(set)
        for record in raw.to_dict("records"):
            on_date = as_date(record.get("date"))
            if on_date:
                market_dates[str(record.get("market") or "")].add(on_date)
    if security_dates is None:
        security_dates = defaultdict(set)
        for record in raw.to_dict("records"):
            on_date = as_date(record.get("date"))
            if on_date:
                security_dates[str(record.get("security_id") or "")].add(on_date)
    dates = {value for value in market_dates.get(market, set()) if start <= value < end}
    member_stats, covered_total = {}, 0
    for record in members.to_dict("records"):
        security_id = str(record.get("security_id"))
        observed = {value for value in security_dates.get(security_id, set()) if start <= value < end}
        covered = len(observed.intersection(dates))
        covered_total += covered
        member_stats[security_id] = {"ticker": record.get("ticker_on_effective_date"), "required": len(dates), "covered": covered, "ratio": covered / len(dates) if dates else 0.0, "local": not raw.loc[raw["security_id"].astype(str).eq(security_id) & raw["provider"].astype(str).isin(["local_existing", "local_vnstock"])].empty}
    required = len(dates) * len(members)
    return {"required": required, "covered": covered_total, "ratio": covered_total / required if required else 0.0, "member_stats": member_stats, "dates": dates}


def build_eligibility(root: Path, memberships: pd.DataFrame, raw: pd.DataFrame) -> tuple[pd.DataFrame, set[str], dict[str, dict[str, Any]]]:
    calendars = pd.read_csv(root / "reports" / "index_review_calendar.csv", dtype=str).fillna("")
    period_map, quality_map = _source_period_map(root), _source_quality_map(root)
    reconciliation = read_parquet(root, "reconciliation/reconciliation_log_v2.parquet")
    conflict_by_source: defaultdict[str, list[str]] = defaultdict(list)
    conflict_by_period: defaultdict[tuple[str, str], list[str]] = defaultdict(list)
    for record in reconciliation.to_dict("records"):
        if str(record.get("severity") or "") != "HIGH" or str(record.get("status") or "") != "OPEN":
            continue
        source_id, index_id, details = str(record.get("source_id") or ""), str(record.get("index_id") or ""), str(record.get("details") or "")
        if source_id:
            conflict_by_source[source_id].append(details)
        for period in re.findall(r"\b20\d{2}[-A-Za-z0-9]+\b", details):
            conflict_by_period[(index_id, period)].append(details)
    rows, eligible_ids, stats_by_id = [], set(), {}
    market_dates: defaultdict[str, set[date]] = defaultdict(set)
    security_dates: defaultdict[str, set[date]] = defaultdict(set)
    for record in raw.to_dict("records"):
        on_date = as_date(record.get("date"))
        if on_date:
            market_dates[str(record.get("market") or "")].add(on_date)
            security_dates[str(record.get("security_id") or "")].add(on_date)
    groups = memberships.groupby(["index_id", "effective_from", "effective_to"], dropna=False, sort=True) if not memberships.empty else []
    for (index_id, start, end), group in groups:
        start_d, end_d = as_date(start), as_date(end)
        source_ids = sorted({str(value) for value in group["anchor_source_id"].dropna().tolist() if str(value)})
        source_id, period = (source_ids[0] if source_ids else ""), period_map.get(source_ids[0], "") if source_ids else ""
        source_official = bool(source_id and str(quality_map.get(source_id, "")).startswith("OFFICIAL"))
        stats = _interval_price_stats({"index_id": index_id, "interval_start": start_d, "interval_end": end_d}, memberships, raw, market_dates, security_dates)
        identity_rate = float(group["identity_resolved"].astype(bool).mean()) if len(group) else 0.0
        event_complete, observed_count = bool(group["event_chain_complete"].astype(bool).all()), int(group["security_id"].nunique())
        conflicts = conflict_by_source.get(source_id, []) + conflict_by_period.get((str(index_id), period), [])
        blockers = []
        if not source_official: blockers.append("source coverage is not accepted official evidence")
        if not event_complete: blockers.append("event-chain incomplete")
        if identity_rate < 0.95: blockers.append(f"identity resolution rate {identity_rate:.3f} < 0.95")
        if conflicts: blockers.append(f"{len(set(conflicts))} unresolved high-severity conflict(s)")
        if stats["ratio"] < 0.95: blockers.append(f"active-member-day raw-OHLC coverage {stats['ratio']:.3f} < 0.95")
        expected = int(v2.INDEX_CONFIG[str(index_id)]["constituent_count"])
        if observed_count != expected: blockers.append(f"observed constituent count {observed_count} != expected {expected}")
        eligible, interval_id = not blockers, stable_id("ELG", index_id, start_d, end_d, source_id)
        row = {"market": group.iloc[0].get("market"), "index_id": index_id, "interval_start": start_d, "interval_end": end_d, "source_coverage": 1.0 if source_official else 0.0, "event_chain_complete": event_complete, "identity_resolution_rate": round(identity_rate, 6), "active_member_day_price_coverage": round(float(stats["ratio"]), 6), "expected_constituent_count": expected, "observed_constituent_count": observed_count, "high_severity_conflicts": len(set(conflicts)), "eligible": bool(eligible), "blocking_reasons": " | ".join(dict.fromkeys(blockers)) or None, "eligibility_interval_id": interval_id, "review_period": period, "source_id": source_id}
        rows.append(row); stats_by_id[interval_id] = {**stats, "row": row, "source_id": source_id}
        if eligible: eligible_ids.add(interval_id)
    for index_id, cfg in v2.INDEX_CONFIG.items():
        if not memberships[memberships["index_id"].astype(str).eq(index_id)].empty:
            continue
        market_cal = calendars[calendars["index_id"].astype(str).eq(index_id)]
        expected, official = len(market_cal), int(market_cal["coverage_status"].astype(str).str.startswith("ACCEPTED_OFFICIAL").sum()) if not market_cal.empty else 0
        conflict_count = int(len(reconciliation[reconciliation["index_id"].astype(str).eq(index_id) & reconciliation["severity"].astype(str).eq("HIGH") & reconciliation["status"].astype(str).eq("OPEN")])) if not reconciliation.empty else 0
        entity_rate = 0.0 if index_id == "FBM_KLCI" else 1.0
        blockers = ["no evidence-backed membership interval reconstructed"]
        if official < expected: blockers.append(f"accepted official review evidence {official}/{expected}")
        if entity_rate < 0.95: blockers.append(f"identity resolution rate {entity_rate:.3f} < 0.95")
        if conflict_count: blockers.append(f"{conflict_count} unresolved high-severity conflict(s)")
        interval_id = stable_id("ELG", index_id, TARGET_START, TARGET_END, "NO_INTERVAL")
        row = {"market": cfg["market"], "index_id": index_id, "interval_start": TARGET_START, "interval_end": TARGET_END, "source_coverage": round(official / expected, 6) if expected else 0.0, "event_chain_complete": False, "identity_resolution_rate": entity_rate, "active_member_day_price_coverage": 0.0, "expected_constituent_count": cfg["constituent_count"], "observed_constituent_count": 0, "high_severity_conflicts": conflict_count, "eligible": False, "blocking_reasons": " | ".join(blockers), "eligibility_interval_id": interval_id, "review_period": "TARGET_SCOPE_NO_INTERVAL", "source_id": None}
        rows.append(row); stats_by_id[interval_id] = {"required": 0, "covered": 0, "ratio": 0.0, "member_stats": {}, "dates": set(), "row": row, "source_id": ""}
    frame = pd.DataFrame(rows).sort_values(["market", "interval_start", "index_id"], kind="stable").reset_index(drop=True)
    return frame, eligible_ids, stats_by_id


def build_coverage_report(root: Path, stats_by_id: Mapping[str, Mapping[str, Any]], raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_keys = {
        (str(security_id), as_date(on_date))
        for security_id, on_date in zip(raw.get("security_id", pd.Series(dtype=str)), raw.get("date", pd.Series(dtype=object)))
        if as_date(on_date)
    }
    for interval_id, stats in stats_by_id.items():
        row, member_stats = dict(stats.get("row") or {}), stats.get("member_stats") or {}
        if not row.get("index_id") or not row.get("interval_start"):
            continue
        market_required, market_covered, market_ratio = int(stats.get("required") or 0), int(stats.get("covered") or 0), float(stats.get("ratio") or 0.0)
        minimum = min((float(value.get("ratio") or 0.0) for value in member_stats.values()), default=0.0)
        ranked = sorted(member_stats.items(), key=lambda item: (float(item[1].get("ratio") or 0.0), item[0]))
        for rank, (security_id, value) in enumerate(ranked, start=1):
            rows.append({"grain": "security", "market": row.get("market"), "index_id": row.get("index_id"), "interval_id": interval_id, "interval_start": row.get("interval_start"), "interval_end": row.get("interval_end"), "security_id": security_id, "historical_ticker": value.get("ticker"), "required_active_member_days": value.get("required"), "covered_active_member_days": value.get("covered"), "coverage_ratio": value.get("ratio"), "market_required_active_member_days": market_required, "market_covered_active_member_days": market_covered, "market_coverage_ratio": market_ratio, "constituent_weighted_coverage": market_ratio, "minimum_security_coverage": minimum, "largest_gap_rank": rank, "price_source_status": "LOCAL_RAW_OHLC" if value.get("local") else ("RAW_PROVIDER_OR_LOCAL" if value.get("covered") else "MISSING_RAW_OHLC")})
        for on_date in sorted(stats.get("dates") or set()):
            covered_count = sum((str(security_id), on_date) in raw_keys for security_id in member_stats)
            rows.append({"grain": "date", "market": row.get("market"), "index_id": row.get("index_id"), "interval_id": interval_id, "interval_start": row.get("interval_start"), "interval_end": row.get("interval_end"), "security_id": None, "historical_ticker": None, "required_active_member_days": len(member_stats), "covered_active_member_days": covered_count, "coverage_ratio": covered_count / len(member_stats) if member_stats else 0.0, "market_required_active_member_days": market_required, "market_covered_active_member_days": market_covered, "market_coverage_ratio": market_ratio, "constituent_weighted_coverage": market_ratio, "minimum_security_coverage": minimum, "largest_gap_rank": None, "price_source_status": "DATE_LEVEL"})
        rows.append({"grain": "market_interval", "market": row.get("market"), "index_id": row.get("index_id"), "interval_id": interval_id, "interval_start": row.get("interval_start"), "interval_end": row.get("interval_end"), "security_id": None, "historical_ticker": None, "required_active_member_days": market_required, "covered_active_member_days": market_covered, "coverage_ratio": market_ratio, "market_required_active_member_days": market_required, "market_covered_active_member_days": market_covered, "market_coverage_ratio": market_ratio, "constituent_weighted_coverage": market_ratio, "minimum_security_coverage": minimum, "largest_gap_rank": None, "price_source_status": "MARKET_INTERVAL"})
    frame = pd.DataFrame(rows)
    path = root / "reports" / "active_member_day_price_coverage_v3.parquet"; write_parquet(path, frame); artifact_manifest(root, path, "Active-member-day raw-OHLC coverage at security, date, and interval grains", len(frame))
    return frame


def build_set50_audit(root: Path, memberships: pd.DataFrame, stats_by_id: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    interval_ids = {(as_date((stats.get("row") or {}).get("interval_start")), as_date((stats.get("row") or {}).get("interval_end"))): interval_id for interval_id, stats in stats_by_id.items() if str((stats.get("row") or {}).get("index_id")) == "SET50"}
    rows = []
    for record in memberships[memberships["index_id"].astype(str).eq("SET50")].to_dict("records"):
        start, end = as_date(record.get("effective_from")), as_date(record.get("effective_to")); stats = (stats_by_id.get(interval_ids.get((start, end)), {}).get("member_stats", {}) if (start, end) in interval_ids else {}).get(str(record.get("security_id")), {})
        blockers = []
        if not record.get("event_chain_complete"): blockers.append("event-chain incomplete")
        if not record.get("identity_resolved"): blockers.append("identity unresolved")
        if not stats.get("covered"): blockers.append("no raw OHLC rows for active member days")
        elif float(stats.get("ratio") or 0.0) < 0.95: blockers.append(f"raw-OHLC coverage {float(stats.get('ratio') or 0.0):.3f} < 0.95")
        if record.get("blocking_reason"): blockers.append(str(record.get("blocking_reason")))
        rows.append({"security_id": record.get("security_id"), "published_name": None, "historical_ticker": record.get("ticker_on_effective_date"), "membership_start": start, "membership_end": end - timedelta(days=1) if end else None, "event_chain_complete": bool(record.get("event_chain_complete")), "identity_resolved": bool(record.get("identity_resolved")), "local_raw_price_available": bool(stats.get("local")), "active_member_days": int(stats.get("required") or 0), "covered_member_days": int(stats.get("covered") or 0), "coverage_ratio": float(stats.get("ratio") or 0.0), "blocking_reason": " | ".join(dict.fromkeys(blockers)) or None})
    frame = pd.DataFrame(rows, columns=AUDIT_COLUMNS); path = root / "reports" / "set50_completion_audit_v3.csv"; write_csv(path, rows, AUDIT_COLUMNS); artifact_manifest(root, path, "SET50 pilot interval/security completion audit", len(frame)); return frame


def build_snapshots(root: Path, memberships: pd.DataFrame, eligibility: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    tickers = read_parquet(root, "processed/ticker_history_v2.parquet"); rows: list[dict[str, Any]] = []
    for interval in eligibility[eligibility["eligible"].astype(bool)].to_dict("records") if not eligibility.empty else []:
        index_id, interval_id = str(interval.get("index_id")), str(interval.get("eligibility_interval_id")); start, end = as_date(interval.get("interval_start")), as_date(interval.get("interval_end"))
        if not start or not end: continue
        members = memberships[memberships["index_id"].astype(str).eq(index_id) & (pd.to_datetime(memberships["effective_from"], errors="coerce").dt.date == start) & (pd.to_datetime(memberships["effective_to"], errors="coerce").dt.date == end)]
        if members.empty: continue
        market = str(members.iloc[0].get("market") or "")
        dates = sorted({as_date(value) for value in raw.loc[raw["market"].astype(str).eq(market) & (pd.to_datetime(raw["date"], errors="coerce").dt.date >= start) & (pd.to_datetime(raw["date"], errors="coerce").dt.date < end), "date"].tolist() if as_date(value)})
        for on_date in dates:
            for record in members.to_dict("records"):
                ticker = _ticker_for(str(record.get("security_id")), on_date, tickers)
                if ticker:
                    rows.append({"date": on_date, "market": market, "index_id": index_id, "security_id": record.get("security_id"), "ticker_on_date": ticker, "membership_effective_from": start, "membership_effective_to": end, "source_chain": record.get("source_chain"), "confidence": record.get("confidence"), "eligibility_interval_id": interval_id, "snapshot_version": SNAPSHOT_VERSION})
    frame = pd.DataFrame(rows)
    if not frame.empty: frame = frame.drop_duplicates(["date", "market", "index_id", "security_id"], keep="first").sort_values(["date", "index_id", "security_id"], kind="stable").reset_index(drop=True)
    path = root / "snapshots" / "pit_universe_daily_v3.parquet"; set50_path = root / "snapshots" / "set50_pit_universe_daily_v3.parquet"
    write_parquet(path, frame); write_parquet(set50_path, frame[frame["index_id"].astype(str).eq("SET50")] if not frame.empty else frame)
    artifact_manifest(root, path, "Daily PIT snapshots restricted to eligible source/identity/price intervals", len(frame)); artifact_manifest(root, set50_path, "SET50 pilot daily PIT snapshots", int(len(frame[frame["index_id"].astype(str).eq("SET50")])) if not frame.empty else 0)
    return frame


def build_price_panels(root: Path, snapshots: pd.DataFrame, raw: pd.DataFrame, repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if snapshots.empty or raw.empty: pit = pd.DataFrame()
    else:
        joined = snapshots.merge(raw, on=["market", "security_id", "date"], how="left").dropna(subset=["open", "high", "low", "close"])
        pit = pd.DataFrame({"date": joined["date"], "market": joined["market"], "index_id": joined["index_id"], "security_id": joined["security_id"], "ticker_on_date": joined["ticker_on_date"], "open": joined["open"].astype(float), "high": joined["high"].astype(float), "low": joined["low"].astype(float), "close": joined["close"].astype(float), "membership_effective_from": joined["membership_effective_from"], "membership_effective_to": joined["membership_effective_to"], "price_provider": joined["provider"], "price_semantics": RAW_SEMANTICS, "membership_source_chain": joined["source_chain"], "eligibility_interval_id": joined["eligibility_interval_id"], "data_version": DATA_VERSION, "universe_semantics": PIT_SEMANTICS}).drop_duplicates(["date", "market", "index_id", "security_id"], keep="first")
    pit_path = root / "processed" / "pit_raw_ohlc_panel_v3.parquet"; write_parquet(pit_path, pit); artifact_manifest(root, pit_path, "PIT membership joined to raw OHLC only on active constituent dates", len(pit))
    current, matched_rows = current_tickers(repo_root), []
    if not snapshots.empty:
        for interval_id, group in snapshots.groupby("eligibility_interval_id", sort=True):
            index_id, market = str(group.iloc[0]["index_id"]), str(group.iloc[0]["market"]); current_set = set(current.get(index_id, pd.DataFrame()).get("local_symbol", pd.Series(dtype=str)).astype(str)); dates = sorted({as_date(value) for value in group["date"].tolist() if as_date(value)})
            if not current_set: continue
            start, end = as_date(group.iloc[0]["membership_effective_from"]), as_date(group.iloc[0]["membership_effective_to"])
            for ticker in sorted(current_set):
                ticker_raw = raw[(raw["market"].astype(str).eq(market)) & raw["ticker_used"].astype(str).eq(ticker) & raw["date"].isin(dates)]
                matched_rows.extend({"date": as_date(record.get("date")), "market": market, "index_id": index_id, "security_id": stable_id("CURRENT", index_id, ticker), "ticker_on_date": ticker, "open": record.get("open"), "high": record.get("high"), "low": record.get("low"), "close": record.get("close"), "membership_effective_from": start, "membership_effective_to": end, "price_provider": record.get("provider"), "price_semantics": RAW_SEMANTICS, "membership_source_chain": "CURRENT_2026_CONFIG_ONLY_NOT_PIT_EVIDENCE", "eligibility_interval_id": interval_id, "data_version": DATA_VERSION, "universe_semantics": CURRENT_SEMANTICS} for record in ticker_raw.to_dict("records"))
    current_panel = pd.DataFrame(matched_rows)
    if not current_panel.empty: current_panel = current_panel.drop_duplicates(["date", "market", "index_id", "security_id"], keep="first").sort_values(["date", "index_id", "security_id"], kind="stable").reset_index(drop=True)
    current_path = root / "processed" / "current_universe_matched_panel_v3.parquet"; write_parquet(current_path, current_panel); artifact_manifest(root, current_path, "Current June-2026 constituents matched on the same eligible dates and raw-OHLC semantics", len(current_panel)); return pit, current_panel


def build_comparison(root: Path, snapshots: pd.DataFrame, pit: pd.DataFrame, current_panel: pd.DataFrame, repo_root: Path) -> pd.DataFrame:
    current, rows = current_tickers(repo_root), []
    for (on_date, index_id), group in snapshots.groupby(["date", "index_id"], sort=True) if not snapshots.empty else []:
        pit_tickers, current_set = set(group["ticker_on_date"].astype(str)), set(current.get(str(index_id), pd.DataFrame()).get("local_symbol", pd.Series(dtype=str)).astype(str)); intersection, union = pit_tickers.intersection(current_set), pit_tickers.union(current_set)
        pit_rows = int(len(pit[(pit["date"] == on_date) & pit["index_id"].astype(str).eq(str(index_id))])) if not pit.empty else 0; current_rows = int(len(current_panel[(current_panel["date"] == on_date) & current_panel["index_id"].astype(str).eq(str(index_id))])) if not current_panel.empty else 0
        rows.append({"date": as_date(on_date), "market": v2.INDEX_CONFIG[str(index_id)]["market"], "index_id": index_id, "pit_constituent_count": len(pit_tickers), "current_universe_constituent_count": len(current_set), "intersection": len(intersection), "union": len(union), "jaccard_overlap": len(intersection) / len(union) if union else 0.0, "current_members_not_yet_active_historically": len(current_set.difference(pit_tickers)), "historical_members_absent_from_june_2026": len(pit_tickers.difference(current_set)), "constituent_turnover": 0, "pit_raw_ohlc_rows": pit_rows, "matched_current_universe_rows": current_rows, "missing_active_member_days": max(0, len(pit_tickers) - pit_rows), "eligibility_interval_id": group.iloc[0].get("eligibility_interval_id"), "pit_universe_semantics": PIT_SEMANTICS, "current_universe_semantics": CURRENT_SEMANTICS})
    columns = ["date", "market", "index_id", "pit_constituent_count", "current_universe_constituent_count", "intersection", "union", "jaccard_overlap", "current_members_not_yet_active_historically", "historical_members_absent_from_june_2026", "constituent_turnover", "pit_raw_ohlc_rows", "matched_current_universe_rows", "missing_active_member_days", "eligibility_interval_id", "pit_universe_semantics", "current_universe_semantics"]
    overlap_path, descriptive_path = root / "reports" / "current_vs_pit_overlap_by_date_v3.csv", root / "reports" / "current_vs_pit_descriptive_v3.parquet"; write_csv(overlap_path, rows, columns); frame = pd.DataFrame(rows, columns=columns); write_parquet(descriptive_path, frame); artifact_manifest(root, overlap_path, "Date-level descriptive current versus PIT overlap", len(frame)); artifact_manifest(root, descriptive_path, "Date-level descriptive current versus PIT overlap in Parquet form", len(frame)); return frame


def build_forecasting_readiness(root: Path, eligibility: pd.DataFrame, pit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in eligibility.to_dict("records"):
        interval_id, cov, chain, identity = str(record.get("eligibility_interval_id")), float(record.get("active_member_day_price_coverage") or 0.0), bool(record.get("event_chain_complete")), float(record.get("identity_resolution_rate") or 0.0); panel_rows = int(len(pit[pit["eligibility_interval_id"].astype(str).eq(interval_id)])) if not pit.empty else 0
        blockers = ([] if chain else ["membership event chain incomplete"]) + ([] if identity >= 0.95 else ["high-severity identity mapping unresolved"]) + ([] if cov >= 0.95 else ["active-member-day raw-OHLC coverage below locked threshold"])
        if bool(record.get("eligible")) and panel_rows == 0: blockers.append("no PIT raw-OHLC panel rows")
        rows.append({"market": record.get("market"), "index_id": record.get("index_id"), "interval_start": record.get("interval_start"), "interval_end": record.get("interval_end"), "eligibility_interval_id": interval_id, "membership_event_chain_complete": chain, "identity_resolution_rate": identity, "membership_interval_validation": chain and int(record.get("observed_constituent_count") or 0) == int(record.get("expected_constituent_count") or 0), "active_member_day_price_coverage": cov, "feature_lookback_available": bool(record.get("eligible")) and chain, "target_dates_valid": bool(record.get("eligible")) and panel_rows > 0, "ticker_delisting_boundary_clear": bool(record.get("eligible")) and chain, "eligible_for_forecasting": bool(record.get("eligible")) and not blockers, "blocking_reasons": " | ".join(dict.fromkeys(blockers + ([str(record.get("blocking_reasons"))] if record.get("blocking_reasons") else []))) or None})
    frame = pd.DataFrame(rows); path = root / "qa" / "forecasting_readiness_v3.csv"; write_csv(path, rows, list(frame.columns) if not frame.empty else ["market", "index_id", "interval_start", "interval_end", "eligibility_interval_id", "membership_event_chain_complete", "identity_resolution_rate", "membership_interval_validation", "active_member_day_price_coverage", "feature_lookback_available", "target_dates_valid", "ticker_delisting_boundary_clear", "eligible_for_forecasting", "blocking_reasons"]); artifact_manifest(root, path, "Forecasting-readiness gate without running forecasting models", len(frame)); return frame


def build_klci_unresolved(root: Path) -> pd.DataFrame:
    calendar, search, rows = pd.read_csv(root / "reports" / "index_review_calendar.csv", dtype=str).fillna(""), pd.read_csv(root / "reports" / "source_search_log.csv", dtype=str).fillna(""), []
    for period in ["2016-12", "2018-06", "2018-12"]:
        cal, logs = calendar[(calendar["index_id"] == "FBM_KLCI") & (calendar["review_period"] == period)], search[(search["index_id"] == "FBM_KLCI") & (search["review_period"] == period)]
        rows.append({"review_period": period, "sources_checked": "; ".join(sorted(set(logs["source_checked"].astype(str)))), "official_source_found": bool(not cal.empty and cal.iloc[0].get("coverage_status") == "ACCEPTED_OFFICIAL_OR_CHECKSUMMED_SOURCE"), "secondary_source_found": bool(not cal.empty and cal.iloc[0].get("coverage_status") == "SECONDARY_CORROBORATION_ONLY"), "membership_event_complete": False, "accepted_source_ids": ";".join(sorted(set(logs["accepted_source_id"].astype(str).replace("nan", "")))), "remaining_gap": "No accepted official membership notice with a complete dated event chain; secondary corroboration is retained but cannot replace official evidence.", "eligibility_effect": "BLOCKS_DATES_COVERING_REVIEW_PERIOD; no interpolation or carry-through"})
    frame = pd.DataFrame(rows); path = root / "reports" / "klci_unresolved_periods_v3.csv"; write_csv(path, rows, ["review_period", "sources_checked", "official_source_found", "secondary_source_found", "membership_event_complete", "accepted_source_ids", "remaining_gap", "eligibility_effect"]); artifact_manifest(root, path, "Explicit FBM KLCI unresolved-review-period search reconciliation", len(frame)); return frame


def _copy_identity_v3(root: Path) -> None:
    for old_relative, new_relative in [("processed/security_master_v2.parquet", "processed/security_master_v3.parquet"), ("processed/ticker_history_v2.parquet", "processed/ticker_history_v3.parquet"), ("reconciliation/entity_resolution_candidates_v2.parquet", "reconciliation/entity_resolution_candidates_v3.parquet"), ("reconciliation/reconciliation_log_v2.parquet", "reconciliation/reconciliation_log_v3.parquet")]:
        old, new = root / old_relative, root / new_relative; frame = pd.read_parquet(old) if old.is_file() else pd.DataFrame(); write_parquet(new, frame); artifact_manifest(root, new, f"Versioned v3 copy of {old_relative}", len(frame))


def _membership_validation(memberships: pd.DataFrame) -> list[str]:
    failures: list[str] = []
    for row in memberships.to_dict("records"):
        start, end = as_date(row.get("effective_from")), as_date(row.get("effective_to"))
        if not start or not end or start >= end: failures.append("non_positive_interval")
        if not row.get("event_chain_complete"): failures.append("incomplete_event_chain_in_interval_table")
        if not str(row.get("ticker_on_effective_date") or ""): failures.append("missing_effective_ticker")
    for (index_id, security_id), group in memberships.groupby(["index_id", "security_id"], sort=False):
        ordered = group.sort_values("effective_from"); previous_end = None
        for record in ordered.to_dict("records"):
            start, end = as_date(record.get("effective_from")), as_date(record.get("effective_to"))
            if previous_end and start and start < previous_end: failures.append(f"overlap:{index_id}:{security_id}")
            previous_end = end if end and (previous_end is None or end > previous_end) else previous_end
    return failures


def _status_payload(root: Path, memberships: pd.DataFrame, name_only: pd.DataFrame, eligibility: pd.DataFrame, raw: pd.DataFrame, pit: pd.DataFrame, current_panel: pd.DataFrame, overlap: pd.DataFrame, readiness: pd.DataFrame, acquisition: Mapping[str, Any], tests: Mapping[str, Any] | None = None) -> dict[str, Any]:
    calendar, events, queue = pd.read_csv(root / "reports" / "index_review_calendar.csv", dtype=str).fillna(""), read_parquet(root, "interim/membership_events_extracted_v2.parquet"), pd.read_csv(root / "reports" / "price_download_queue_by_provider.csv", dtype=str).fillna("")
    security_master = read_parquet(root, "processed/security_master_v2.parquet")
    queued_ids = set(queue.loc[queue["download_status"].astype(str).eq("QUEUED"), "security_id"].astype(str)) if not queue.empty else set()
    local_ids = set(
        security_id
        for security_id, group in queue.groupby("security_id", sort=False)
        if group["download_status"].astype(str).eq("LOCAL_RAW_OHLC_COVERED").all()
    ) if not queue.empty else set()
    raw_log = pd.read_csv(root / "reports" / "raw_ohlc_acquisition_log_v3.csv", dtype=str).fillna("") if (root / "reports" / "raw_ohlc_acquisition_log_v3.csv").is_file() else pd.DataFrame()
    downloaded_ids = set(raw_log.loc[raw_log["provider"].astype(str).eq("yahoo") & raw_log["retrieval_status"].astype(str).isin(["success", "partial"]), "security_id"].astype(str)) if not raw_log.empty else set()
    failed_ids = set(pd.read_csv(root / "reports" / "raw_ohlc_unresolved_v3.csv", dtype=str).fillna("")["security_id"].astype(str)) if (root / "reports" / "raw_ohlc_unresolved_v3.csv").is_file() else set()
    full_eligible = eligibility[eligibility["eligible"].astype(bool)] if not eligibility.empty else pd.DataFrame(); blocked = eligibility[~eligibility["eligible"].astype(bool)] if not eligibility.empty else pd.DataFrame(); partial_indices = sorted(set(full_eligible["index_id"].astype(str))) if not full_eligible.empty else []
    unresolved_periods = calendar[calendar["coverage_status"].astype(str).isin(["SECONDARY_CORROBORATION_ONLY", "SEARCHED_NO_ACCEPTED_SOURCE"])]
    status = "COMPLETE" if len(partial_indices) == len(v2.INDEX_CONFIG) and not len(name_only) and not len(blocked) and not acquisition.get("failed_downloads") else ("PARTIAL" if partial_indices else "BLOCKED")
    by_market_coverage = {str(market): float(pd.to_numeric(group["active_member_day_price_coverage"], errors="coerce").fillna(0.0).max()) for market, group in eligibility.groupby("market")} if not eligibility.empty else {}
    return {"status": status, "pipeline_version": PIPELINE_VERSION, "target_period": {"start": TARGET_START, "end": TARGET_END}, "source_and_event_chains": {"events_extracted": int(len(events)), "events_validated": int(len(events[events.get("manual_review_required", pd.Series(dtype=bool)).astype(bool).eq(False)])) if not events.empty else 0, "membership_intervals": int(len(memberships)), "complete_event_chain_intervals": int(len(memberships.groupby(["index_id", "effective_from", "effective_to"]))) if not memberships.empty else 0, "incomplete_event_chain_markets": sorted(set(v2.INDEX_CONFIG) - set(memberships["index_id"].astype(str)) if not memberships.empty else set(v2.INDEX_CONFIG)), "unresolved_review_periods": sorted(set(unresolved_periods["index_id"].astype(str) + ":" + unresolved_periods["review_period"].astype(str))) if not unresolved_periods.empty else []}, "identity": {"historical_securities": int(len(security_master)), "resolved_identities": int(len(security_master)), "unresolved_identities": int(len(name_only)), "resolution_methods": {"exact_official_exchange_code": int(len(security_master)), "manual_unresolved_name_only": int(len(name_only))}}, "raw_price_coverage": {"securities_required": int(len(security_master)), "securities_locally_available": int(len(local_ids)), "securities_queued_for_download": int(len(queued_ids)), "securities_successfully_downloaded": int(len(downloaded_ids.intersection(queued_ids))), "failed_downloads": int(len(failed_ids.intersection(queued_ids))), "unqueueable_securities": int(max(0, len(security_master) - len(queue["security_id"].unique()) if not queue.empty else len(security_master))), "raw_rows": int(len(raw)), "active_member_day_coverage_by_market": by_market_coverage}, "eligibility": {"full_period_eligible_markets": [], "partial_period_eligible_markets": partial_indices, "eligible_date_ranges": [{"market": row.get("market"), "index_id": row.get("index_id"), "start": row.get("interval_start"), "end_exclusive": row.get("interval_end")} for row in full_eligible.to_dict("records")], "blocked_intervals": int(len(blocked)), "blockers": {str(row.get("index_id")): str(row.get("blocking_reasons")) for row in blocked.to_dict("records")}}, "data_output": {"pit_snapshot_rows": int(len(pd.read_parquet(root / "snapshots" / "pit_universe_daily_v3.parquet"))) if (root / "snapshots" / "pit_universe_daily_v3.parquet").is_file() else 0, "pit_raw_ohlc_rows": int(len(pit)), "matched_current_universe_rows": int(len(current_panel)), "descriptive_comparison_rows": int(len(overlap))}, "tests": dict(tests or {"passed": None, "failed": None, "status": "NOT_RECORDED"}), "readiness": {"forecasting_ready_intervals": int(readiness["eligible_for_forecasting"].astype(bool).sum()) if not readiness.empty else 0}}


def write_status_report(root: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    report_dir = root.parent.parent / "reports" / "data_execution"; report_dir.mkdir(parents=True, exist_ok=True); json_path, md_path = report_dir / "PIT_COMPLETION_STATUS_V3.json", report_dir / "PIT_COMPLETION_STATUS_V3.md"; write_json(json_path, payload)
    source, ident, price, elig, output, tests = payload.get("source_and_event_chains", {}), payload.get("identity", {}), payload.get("raw_price_coverage", {}), payload.get("eligibility", {}), payload.get("data_output", {}), payload.get("tests", {})
    lines = ["# PIT completion status v3", "", f"Status: **{payload.get('status')}**", "", "This is an execution and data-quality status report. It contains no manuscript conclusions or forecasting results.", "", "## SOURCE AND EVENT CHAINS", "", f"- Events extracted: {source.get('events_extracted', 0)}; events validated: {source.get('events_validated', 0)}.", f"- Membership intervals: {source.get('membership_intervals', 0)}; complete interval chains: {source.get('complete_event_chain_intervals', 0)}.", f"- Markets without reconstructed intervals: {', '.join(source.get('incomplete_event_chain_markets', [])) or 'none'}.", f"- Unresolved review periods: {', '.join(source.get('unresolved_review_periods', [])) or 'none'}.", "", "## IDENTITY", "", f"- Historical securities required: {ident.get('historical_securities', 0)}; resolved: {ident.get('resolved_identities', 0)}; unresolved: {ident.get('unresolved_identities', 0)}.", f"- Resolution methods: {json.dumps(ident.get('resolution_methods', {}), sort_keys=True)}.", "", "## RAW PRICE COVERAGE", "", f"- Securities required: {price.get('securities_required', 0)}; local raw-OHLC available: {price.get('securities_locally_available', 0)}; queued: {price.get('securities_queued_for_download', 0)}; successfully downloaded: {price.get('securities_successfully_downloaded', 0)}; failed downloads: {price.get('failed_downloads', 0)}; unqueueable: {price.get('unqueueable_securities', 0)}.", f"- Active-member-day coverage by market/interval: {json.dumps(price.get('active_member_day_coverage_by_market', {}), sort_keys=True)}.", "", "## ELIGIBILITY", "", f"- Full-period eligible markets: {', '.join(elig.get('full_period_eligible_markets', [])) or 'none'}.", f"- Partial-period eligible markets: {', '.join(elig.get('partial_period_eligible_markets', [])) or 'none'}.", f"- Eligible date ranges: {json.dumps(elig.get('eligible_date_ranges', []), default=_json_default)}.", f"- Blocked intervals: {elig.get('blocked_intervals', 0)}.", "- Exact blockers:"]
    lines.extend([f"  - {index_id}: {reason}" for index_id, reason in sorted((elig.get("blockers", {}) or {}).items())] or ["  - none"]); lines.extend(["", "## DATA OUTPUT", "", f"- PIT snapshot rows: {output.get('pit_snapshot_rows', 0)}.", f"- PIT raw-OHLC rows: {output.get('pit_raw_ohlc_rows', 0)}.", f"- Matched current-universe rows: {output.get('matched_current_universe_rows', 0)}.", f"- Descriptive comparison rows: {output.get('descriptive_comparison_rows', 0)}.", "", "## TESTS", "", f"- Passed: {tests.get('passed')}; failed: {tests.get('failed')}; status: {tests.get('status')}.", "", "No forecasting model was run. Intervals that fail any locked gate remain blocked."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8"); artifact_manifest(root, md_path, "PIT v3 completion status report", None); artifact_manifest(root, json_path, "PIT v3 completion status JSON", None); return md_path, json_path


def build_v3(repo_root: Path, root: Path) -> dict[str, Any]:
    ensure_dirs(root); name_only = build_name_only_resolution(root); name_path = root / "reconciliation" / "name_only_resolution_v3.parquet"; write_parquet(name_path, name_only); artifact_manifest(root, name_path, "Deterministic identity-resolution decisions for name-only records", len(name_only)); _copy_identity_v3(root)
    memberships, tickers, securities, _ = build_membership_v3(root); membership_path = root / "processed" / "index_membership_v3.parquet"; write_parquet(membership_path, memberships, INTERVAL_COLUMNS); artifact_manifest(root, membership_path, "Evidence-backed v3 membership intervals; v2 snapshot-only evidence is excluded", len(memberships)); unqueueable = build_unqueueable_report(root); raw = _raw_by_security_date(root)
    eligibility, eligible_ids, stats_by_id = build_eligibility(root, memberships, raw); eligibility_path = root / "qa" / "pit_interval_eligibility_v3.parquet"; write_parquet(eligibility_path, eligibility); artifact_manifest(root, eligibility_path, "Interval-specific fail-closed PIT eligibility assessment", len(eligibility)); coverage = build_coverage_report(root, stats_by_id, raw); build_set50_audit(root, memberships, stats_by_id); snapshots = build_snapshots(root, memberships, eligibility, raw); pit, current_panel = build_price_panels(root, snapshots, raw, repo_root); overlap = build_comparison(root, snapshots, pit, current_panel, repo_root); readiness = build_forecasting_readiness(root, eligibility, pit); klci = build_klci_unresolved(root)
    log_path, unresolved_path = root / "reports" / "raw_ohlc_acquisition_log_v3.csv", root / "reports" / "raw_ohlc_unresolved_v3.csv"; acquisition_log = pd.read_csv(log_path, dtype=str).fillna("") if log_path.is_file() else pd.DataFrame(); acquisition = {"queued_securities": int(acquisition_log["security_id"].nunique()) if not acquisition_log.empty else 0, "successful_provider_attempts": int(acquisition_log[(acquisition_log["provider"] == "yahoo") & acquisition_log["retrieval_status"].isin(["success", "partial"])]["security_id"].nunique()) if not acquisition_log.empty else 0, "failed_downloads": int(pd.read_csv(unresolved_path).shape[0]) if unresolved_path.is_file() else 0}
    failures = _membership_validation(memberships); validation_path = root / "qa" / "v3_membership_validation.json"; write_json(validation_path, {"passed": not failures, "failures": failures, "checked_at": now_utc()}); artifact_manifest(root, validation_path, "v3 interval/ticker validation", None); payload = _status_payload(root, memberships, name_only, eligibility, raw, pit, current_panel, overlap, readiness, acquisition); write_status_report(root, payload); write_json(root / "manifests" / "point_in_time_v3_run.manifest.json", {"pipeline": "point_in_time_v3", "pipeline_version": PIPELINE_VERSION, "target_period": {"start": TARGET_START, "end": TARGET_END}, "eligible_interval_ids": sorted(eligible_ids), "artifacts": sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file() and ("v3" in path.name or path in {membership_path, eligibility_path}))}); return {"status": payload["status"], "membership_intervals": len(memberships), "resolved_identities": len(securities), "unresolved_identities": len(name_only), "raw_price_rows": len(raw), "eligible_intervals": int(eligibility["eligible"].astype(bool).sum()) if not eligibility.empty else 0, "pit_snapshot_rows": len(snapshots), "pit_raw_ohlc_rows": len(pit), "matched_current_universe_rows": len(current_panel), "overlap_date_rows": len(overlap), "status_report": str(root.parent.parent / "reports" / "data_execution" / "PIT_COMPLETION_STATUS_V3.md"), "unqueueable": len(unqueueable), "klci_unresolved": len(klci)}


def finalize_tests(root: Path, passed: int, failed: int, command: str) -> dict[str, Any]:
    json_path = root.parent.parent / "reports" / "data_execution" / "PIT_COMPLETION_STATUS_V3.json"; payload = json.loads(json_path.read_text(encoding="utf-8")); payload["tests"] = {"passed": int(passed), "failed": int(failed), "status": "PASS" if failed == 0 else "FAIL", "command": command, "recorded_at": now_utc()}; write_status_report(root, payload); return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("acquire-prices", "build"):
        command = sub.add_parser(name); command.add_argument("--repo-root"); command.add_argument("--output-root")
    acquire = sub.choices["acquire-prices"]; acquire.add_argument("--sleep-seconds", type=float, default=1.0); acquire.add_argument("--timeout-seconds", type=int, default=30); acquire.add_argument("--retries", type=int, default=2)
    final = sub.add_parser("finalize-tests"); final.add_argument("--repo-root"); final.add_argument("--output-root"); final.add_argument("--passed", required=True, type=int); final.add_argument("--failed", required=True, type=int); final.add_argument("--command", required=True)
    args = parser.parse_args(argv); repo_root = Path(args.repo_root).resolve() if args.repo_root else REPO_ROOT; root = Path(args.output_root).resolve() if args.output_root else repo_root / "research_data" / "point_in_time"
    if args.command == "acquire-prices": result = acquire_raw_prices(repo_root, root, sleep_seconds=args.sleep_seconds, timeout_seconds=args.timeout_seconds, retries=args.retries)
    elif args.command == "build": result = build_v3(repo_root, root)
    else: result = finalize_tests(root, args.passed, args.failed, args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
