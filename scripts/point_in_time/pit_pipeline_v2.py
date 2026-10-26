#!/usr/bin/env python3
"""Evidence-backed, additive v2 PIT constituent reconstruction.

This module deliberately leaves the v1 offline baseline untouched.  It adds a
separate, source-led flow that can archive originals, record failed searches,
parse only reviewed official constituent lists, and emit point-in-time outputs
only where the evidence chain is explicit.  It never derives a historical
universe from the June-2026 comparator.

The first supported acquisition adapter is the public SET50 archive.  Other
markets are accepted through checksummed source-inventory CSV files, manually
reviewed event drafts, and (only when the exact list and boundaries are
explicit) direct-snapshot anchor drafts.  This retains raw originals and
review findings without changing the baseline pipeline or inventing historical
members from an unreviewed source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - explicit runtime diagnostic
    PdfReader = None  # type: ignore[assignment,misc]


PIPELINE_VERSION = "2.0.0"
PARSER_VERSION = "pit-set50-pdf-v2.0.0"
SCHEMA_VERSION = "2.0.0"
TARGET_START = date(2016, 1, 1)
TARGET_END = date(2026, 6, 1)
TARGET_END_EXCLUSIVE = TARGET_END + timedelta(days=1)
PIT_SEMANTICS = "POINT_IN_TIME"
CURRENT_SEMANTICS = "CURRENT_2026_APPLIED_BACKWARD"
SNAPSHOT_VERSION = "pit-universe-v2"
UTC = timezone.utc

SET_ARCHIVE_URL = (
    "https://www.set.or.th/en/market/information/securities-list/"
    "constituents-list-set50-set100"
)
SET_METHODOLOGY_URL = "https://media.set.or.th/set/Documents/2025/Feb/Index_Ground_Rule_EN.pdf"


INDEX_CONFIG: dict[str, dict[str, Any]] = {
    "VN30": {
        "market": "Vietnam",
        "exchange": "HOSE",
        "index_name": "VN30",
        "constituent_count": 30,
        "review_frequency": "semiannual",
        "review_months": (1, 7),
        "announcement_window": "December/June before the January/July implementation",
        "effective_window": "January/July implementation; exact date must come from a dated notice",
        "extraordinary_rules": "Corporate actions, delistings, suspension and replacement require a dated official notice.",
        "official_search_url": "https://www.hsx.vn/Modules/News/NewsList.aspx",
        "price_directory": "vietnam",
        "exchange_suffix": ".VN",
        "preferred_provider": "EODHD",
        "fallback_provider": "Yahoo Finance",
    },
    "SET50": {
        "market": "Thailand",
        "exchange": "SET",
        "index_name": "SET50",
        "constituent_count": 50,
        "review_frequency": "semiannual",
        "review_months": (1, 7),
        "announcement_window": "December before January; June before July",
        "effective_window": "January 1/first trading day and July 1/first trading day, as stated in each official list or notice",
        "extraordinary_rules": "New listings, delistings, suspension, merger and restructuring can change constituents between periodic reviews.",
        "official_search_url": SET_ARCHIVE_URL,
        "methodology_url": SET_METHODOLOGY_URL,
        "price_directory": "thailand",
        "exchange_suffix": ".BK",
        "preferred_provider": "EODHD",
        "fallback_provider": "Yahoo Finance",
    },
    "IDX30": {
        "market": "Indonesia",
        "exchange": "IDX",
        "index_name": "IDX30",
        "constituent_count": 30,
        "review_frequency": "semiannual through 2018; quarterly scheduled evaluation from 2019",
        "review_months": (1, 4, 7, 10),
        "announcement_window": "January/July through 2018; January/April/July/October from 2019, as stated by official IDX notices",
        "effective_window": "Following-month implementation (February/August through 2018; February/May/August/November from 2019); exact date must come from a dated notice",
        "extraordinary_rules": "Corporate-action, delisting, suspension, fast-entry, free-float and methodology adjustments require a dated official IDX notice.",
        "official_search_url": "https://www.idx.co.id/en-us/market-data/indices/",
        "price_directory": "indonesia",
        "exchange_suffix": ".JK",
        "preferred_provider": "EODHD",
        "fallback_provider": "Yahoo Finance",
    },
    "FBM_KLCI": {
        "market": "Malaysia",
        "exchange": "BURSA",
        "index_name": "FBM KLCI",
        "constituent_count": 30,
        "review_frequency": "semiannual",
        "review_months": (6, 12),
        "announcement_window": "June/December FTSE Russell review notice",
        "effective_window": "After the third Friday review close / following Monday, as stated in the notice",
        "extraordinary_rules": "Fast entry, deletion, merger, delisting and corporate-action changes require a dated FTSE Russell/Bursa notice.",
        "official_search_url": "https://research.ftserussell.com/products/index-notices/home/getnotice/",
        "price_directory": "malaysia",
        "exchange_suffix": ".KL",
        "preferred_provider": "EODHD",
        "fallback_provider": "Yahoo Finance",
    },
    "PSEI": {
        "market": "Philippines",
        "exchange": "PSE",
        "index_name": "PSEi",
        "constituent_count": 30,
        "review_frequency": "semiannual",
        "review_months": (2, 8),
        "announcement_window": "January/July before the February/August implementation",
        "effective_window": "February/August implementation; exact date must come from a dated PSE notice",
        "extraordinary_rules": "Delistings, mergers, suspension and replacement changes require a dated PSE notice.",
        "official_search_url": "https://edge.pse.com.ph/",
        "price_directory": "philippines",
        "exchange_suffix": ".PS",
        "preferred_provider": "EODHD",
        "fallback_provider": "Yahoo Finance",
    },
}


def _schema(fields: Sequence[tuple[str, pa.DataType]]) -> pa.Schema:
    return pa.schema([pa.field(name, dtype, nullable=True) for name, dtype in fields])


SCHEMAS: dict[str, pa.Schema] = {
    "source_evidence_v2": _schema(
        [
            ("source_id", pa.string()),
            ("index_id", pa.string()),
            ("market", pa.string()),
            ("review_period", pa.string()),
            ("publisher", pa.string()),
            ("document_title", pa.string()),
            ("original_url", pa.string()),
            ("raw_path", pa.string()),
            ("content_type", pa.string()),
            ("source_type", pa.string()),
            ("source_quality", pa.string()),
            ("publication_date", pa.date32()),
            ("announcement_date", pa.date32()),
            ("stated_effective_date", pa.date32()),
            ("retrieval_timestamp", pa.timestamp("us", tz="UTC")),
            ("file_checksum", pa.string()),
            ("parser_version", pa.string()),
            ("evidence_scope", pa.string()),
            ("source_status", pa.string()),
            ("notes", pa.string()),
        ]
    ),
    "membership_events_extracted_v2": _schema(
        [
            ("index_id", pa.string()),
            ("market", pa.string()),
            ("security_name_as_published", pa.string()),
            ("ticker_as_published", pa.string()),
            ("event_type", pa.string()),
            ("announcement_date", pa.date32()),
            ("effective_date", pa.date32()),
            ("review_period", pa.string()),
            ("source_id", pa.string()),
            ("page_or_section", pa.string()),
            ("extraction_method", pa.string()),
            ("confidence", pa.float64()),
            ("manual_review_required", pa.bool_()),
            ("source_record", pa.string()),
            ("notes", pa.string()),
        ]
    ),
    "security_master_v2": _schema(
        [
            ("security_id", pa.string()),
            ("market", pa.string()),
            ("exchange", pa.string()),
            ("official_exchange_code", pa.string()),
            ("isin", pa.string()),
            ("legal_name", pa.string()),
            ("listing_date", pa.date32()),
            ("delisting_date", pa.date32()),
            ("source_count", pa.int64()),
            ("identity_match_rule", pa.string()),
            ("resolution_status", pa.string()),
            ("manual_review_required", pa.bool_()),
            ("notes", pa.string()),
        ]
    ),
    "ticker_history_v2": _schema(
        [
            ("security_id", pa.string()),
            ("ticker", pa.string()),
            ("exchange_suffix", pa.string()),
            ("valid_from", pa.date32()),
            ("valid_to", pa.date32()),
            ("source_id", pa.string()),
            ("match_rule", pa.string()),
            ("confidence", pa.float64()),
            ("resolution_status", pa.string()),
        ]
    ),
    "entity_resolution_candidates_v2": _schema(
        [
            ("candidate_id", pa.string()),
            ("source_record", pa.string()),
            ("market", pa.string()),
            ("index_id", pa.string()),
            ("ticker_as_published", pa.string()),
            ("security_name_as_published", pa.string()),
            ("candidate_security_id", pa.string()),
            ("match_rule", pa.string()),
            ("similarity_score", pa.float64()),
            ("date_consistency", pa.bool_()),
            ("exchange_consistency", pa.bool_()),
            ("decision", pa.string()),
            ("manual_review_required", pa.bool_()),
            ("review_notes", pa.string()),
        ]
    ),
    "reconciliation_log_v2": _schema(
        [
            ("log_id", pa.string()),
            ("issue_type", pa.string()),
            ("severity", pa.string()),
            ("index_id", pa.string()),
            ("market", pa.string()),
            ("security_id", pa.string()),
            ("source_id", pa.string()),
            ("event_date", pa.date32()),
            ("status", pa.string()),
            ("details", pa.string()),
        ]
    ),
    "index_membership_v2": _schema(
        [
            ("index_id", pa.string()),
            ("market", pa.string()),
            ("security_id", pa.string()),
            ("effective_from", pa.date32()),
            ("effective_to", pa.date32()),
            ("event_type", pa.string()),
            ("announcement_date", pa.date32()),
            ("source_id", pa.string()),
            ("reconstruction_method", pa.string()),
            ("anchor_source_id", pa.string()),
            ("event_chain_complete", pa.bool_()),
            ("evidence_count", pa.int64()),
            ("confidence", pa.float64()),
            ("resolution_status", pa.string()),
            ("notes", pa.string()),
        ]
    ),
    "pit_universe_daily_v2": _schema(
        [
            ("date", pa.date32()),
            ("market", pa.string()),
            ("index_id", pa.string()),
            ("security_id", pa.string()),
            ("ticker_on_date", pa.string()),
            ("membership_effective_from", pa.date32()),
            ("membership_effective_to", pa.date32()),
            ("source_chain", pa.string()),
            ("confidence", pa.float64()),
            ("eligibility_interval", pa.string()),
            ("snapshot_version", pa.string()),
            ("universe_semantics", pa.string()),
            ("is_evidence_backed", pa.bool_()),
        ]
    ),
    "current_vs_pit_universe_descriptive_v2": _schema(
        [
            ("date", pa.date32()),
            ("market", pa.string()),
            ("index_id", pa.string()),
            ("current_universe_size", pa.int64()),
            ("pit_universe_size", pa.int64()),
            ("intersection", pa.int64()),
            ("union", pa.int64()),
            ("jaccard_overlap", pa.float64()),
            ("current_members_not_historically_active", pa.int64()),
            ("historical_members_absent_from_june_2026", pa.int64()),
            ("constituent_turnover", pa.int64()),
            ("market_date_coverage", pa.string()),
            ("current_universe_semantics", pa.string()),
            ("pit_universe_semantics", pa.string()),
        ]
    ),
}


TABLE_PATHS = {
    "source_evidence_v2": "processed/source_evidence_v2.parquet",
    "membership_events_extracted_v2": "interim/membership_events_extracted_v2.parquet",
    "security_master_v2": "processed/security_master_v2.parquet",
    "ticker_history_v2": "processed/ticker_history_v2.parquet",
    "entity_resolution_candidates_v2": "reconciliation/entity_resolution_candidates_v2.parquet",
    "reconciliation_log_v2": "reconciliation/reconciliation_log_v2.parquet",
    "index_membership_v2": "processed/index_membership_v2.parquet",
    "pit_universe_daily_v2": "snapshots/pit_universe_daily_v2.parquet",
    "current_vs_pit_universe_descriptive_v2": "reports/current_vs_pit_universe_descriptive_v2.parquet",
}

PRIMARY_KEYS = {
    "source_evidence_v2": ["source_id"],
    "membership_events_extracted_v2": ["source_record"],
    "security_master_v2": ["security_id"],
    "ticker_history_v2": ["security_id", "ticker", "valid_from", "valid_to", "source_id"],
    "entity_resolution_candidates_v2": ["candidate_id"],
    "reconciliation_log_v2": ["log_id"],
    "index_membership_v2": ["index_id", "security_id", "effective_from", "effective_to", "source_id"],
    "pit_universe_daily_v2": ["date", "index_id", "security_id"],
    "current_vs_pit_universe_descriptive_v2": ["date", "index_id"],
}


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    text = "|".join("" if item is None else str(item) for item in parts)
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:length]}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_date(value: Any) -> date | None:
    if value is None or value == "" or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %B %Y",
    ):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def date_to_iso(value: date | None) -> str | None:
    return value.isoformat() if value else None


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def ensure_output_layout(root: Path) -> None:
    for name in (
        "schemas",
        "processed",
        "interim",
        "reconciliation",
        "snapshots",
        "reports",
        "qa",
        "manifests",
        "raw_sources",
        "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)


def rows_to_table(name: str, rows: Iterable[Mapping[str, Any]]) -> pa.Table:
    schema = SCHEMAS[name]
    records = list(rows)
    arrays = [pa.array([row.get(field.name) for row in records], type=field.type, from_pandas=True) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


def empty_table(name: str) -> pa.Table:
    return rows_to_table(name, [])


def write_table(root: Path, name: str, table: pa.Table) -> Path:
    expected = SCHEMAS[name]
    if not table.schema.equals(expected, check_metadata=False):
        table = table.cast(expected, safe=False)
    path = root / TABLE_PATHS[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=False, version="2.6")
    os.replace(temporary, path)
    return path


def read_table(root: Path, name: str) -> pa.Table:
    path = root / TABLE_PATHS[name]
    return pq.read_table(path) if path.exists() else empty_table(name)


def duplicate_count(table: pa.Table, columns: Sequence[str]) -> int:
    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    for row in table.select(list(columns)).to_pylist():
        key = tuple(row.get(column) for column in columns)
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def schema_payload(name: str) -> dict[str, Any]:
    return {
        "dataset": name,
        "schema_version": SCHEMA_VERSION,
        "primary_key": PRIMARY_KEYS[name],
        "fields": [
            {"name": field.name, "arrow_type": str(field.type), "nullable": field.nullable}
            for field in SCHEMAS[name]
        ],
    }


def write_schema_files(root: Path) -> None:
    for name in SCHEMAS:
        atomic_write_json(root / "schemas" / f"{name}.schema.json", schema_payload(name))


def write_table_manifest(root: Path, name: str, table: pa.Table) -> Path:
    path = root / TABLE_PATHS[name]
    payload = {
        "dataset": name,
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "path": TABLE_PATHS[name],
        "format": "parquet",
        "sha256": sha256_file(path),
        "row_count": int(table.num_rows),
        "primary_key": PRIMARY_KEYS[name],
        "duplicate_rows_by_primary_key": duplicate_count(table, PRIMARY_KEYS[name]),
        "schema_path": f"schemas/{name}.schema.json",
        "schema_sha256": sha256_file(root / "schemas" / f"{name}.schema.json"),
    }
    manifest = root / "manifests" / f"{name}.manifest.json"
    atomic_write_json(manifest, payload)
    return manifest


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(columns))
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def content_type_for_path(path_or_url: str) -> str:
    suffix = Path(urlparse(path_or_url).path).suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".csv": "text/csv",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
        ".xls": "application/vnd.ms-excel",
        ".txt": "text/plain",
    }.get(suffix, "application/octet-stream")


def clean_filename(url: str) -> str:
    name = Path(unquote(urlparse(url).path)).name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "source.bin")
    return name[:180]


def download_original(
    *,
    url: str,
    market: str,
    storage_year: int,
    root: Path,
    timeout_seconds: int = 45,
    retries: int = 3,
) -> tuple[Path | None, str | None, str | None, datetime, list[dict[str, Any]]]:
    """Acquire original bytes unchanged; failed attempts are returned for the search log."""

    attempts: list[dict[str, Any]] = []
    retrieved_at = utc_now()
    payload: bytes | None = None
    error: str | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "VSEF-PIT-Research/2.0 (+archival-source-acquisition)"})
            with urlopen(request, timeout=timeout_seconds) as response:  # nosec B310 - explicit public source URLs
                payload = response.read()
            if not payload:
                raise RuntimeError("empty response")
            attempts.append({"attempt": attempt, "result": "SUCCESS"})
            break
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({"attempt": attempt, "result": "FAILED", "error": error})
            if attempt < retries:
                time.sleep(float(attempt))
    if payload is None:
        return None, None, error, retrieved_at, attempts
    checksum = sha256_bytes(payload)
    destination = root / "raw_sources" / market / str(storage_year) / f"{checksum[:16]}_{clean_filename(url)}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != checksum:
            raise RuntimeError(f"Refusing to overwrite a checksum-mismatched original: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(payload)
        if sha256_file(temporary) != checksum:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch while archiving {url}")
        os.replace(temporary, destination)
    return destination, checksum, None, retrieved_at, attempts


def review_period_for(index_id: str, effective_start: date) -> str:
    """Return a stable period label whose year is the calendar effective year."""

    month = effective_start.month
    if index_id in {"VN30", "SET50"}:
        return f"{effective_start.year}-H1" if month <= 1 else f"{effective_start.year}-H2"
    if index_id == "IDX30":
        announcement_month = effective_start.month - 1
        if announcement_month <= 0:
            return f"{effective_start.year - 1}-12"
        return f"{effective_start.year}-{announcement_month:02d}"
    if index_id == "PSEI":
        return f"{effective_start.year}-H1" if month <= 2 else f"{effective_start.year}-H2"
    return f"{effective_start.year}-06" if month <= 6 else f"{effective_start.year}-12"


def expected_review_periods(index_id: str) -> list[dict[str, Any]]:
    """Generate the declared review calendar without treating it as membership evidence."""

    config = INDEX_CONFIG[index_id]
    rows: list[dict[str, Any]] = []
    if index_id == "IDX30":
        schedule = [
            (year, month, f"{year}-{month:02d}")
            for year in range(TARGET_START.year, TARGET_END.year + 1)
            for month in ((1, 7) if year <= 2018 else (1, 4, 7, 10))
            if date(year, month + 1, 1) <= TARGET_END
        ]
    elif index_id == "FBM_KLCI":
        # The June 2026 semi-annual review was announced after the 2026-06-01
        # PIT cutoff, so only periods whose scheduled implementation can occur
        # by the cutoff are expected here.  The out-of-scope search is still
        # retained in the source-search log when it was performed.
        schedule = [
            (year, month, f"{year}-{month:02d}")
            for year in range(TARGET_START.year, TARGET_END.year + 1)
            for month in (6, 12)
            if date(year, month, 1) < TARGET_END
        ]
    else:
        schedule = [
            (year, month, review_period_for(index_id, date(year, month, 1)))
            for year in range(TARGET_START.year, TARGET_END.year + 1)
            for month in config["review_months"]
            if date(year, month, 1) <= TARGET_END
        ]
    for year, month, period in schedule:
        rows.append(
            {
                "index_id": index_id,
                "market": config["market"],
                "review_period": period,
                "expected_announcement_window": config["announcement_window"],
                "expected_effective_window": config["effective_window"],
                "review_type": "SCHEDULED_PERIODIC_REVIEW",
                "source": config["official_search_url"],
                "source_quality": "OFFICIAL_METHODOLOGY_OR_ARCHIVE_PENDING_PERIOD_EVIDENCE",
                "coverage_status": "SEARCH_REQUIRED",
                "notes": config["extraordinary_rules"],
            }
        )
    return sorted(rows, key=lambda item: (item["index_id"], item["review_period"]))


def parse_set_archive_entries(raw_html: Path) -> list[dict[str, Any]]:
    """Read the publicly rendered SET archive and retain its own period labels.

    A PDF URL is accepted only when the archive rendered an explicit period label
    in the 2016-01-01 to 2026-06-01 study scope.  That prevents guessed filename
    paths from being mistaken for source evidence.
    """

    text = raw_html.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(
        r'<div data-element="element_file_download".*?<div class="inner-text[^>]*">\s*(.*?)\s*</div>.*?'
        r'href="/services/download\?url=([^&\"]+)',
        re.S,
    )
    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for match in pattern.finditer(text):
        label = re.sub(r"<[^>]+>", "", html.unescape(match.group(1))).strip()
        label = re.sub(r"\s+", " ", label)
        url = html.unescape(match.group(2))
        if "SET50" not in url.upper():
            continue
        interval = parse_set_interval_label(label)
        if interval is None:
            continue
        start, end_inclusive = interval
        if end_inclusive < TARGET_START or start > TARGET_END:
            continue
        key = (label, url)
        if key in seen:
            continue
        seen.add(key)
        entries.append(
            {
                "label": label,
                "url": url,
                "effective_start": start,
                "effective_end_inclusive": end_inclusive,
                "review_period": review_period_for("SET50", start),
                "is_revision": "between the periodic review update" in label.lower(),
            }
        )
    if not entries:
        raise ValueError("SET archive parser did not find any target-period SET50 source URLs")
    return sorted(entries, key=lambda item: (item["effective_start"], item["is_revision"], item["url"]))


def parse_set_interval_label(label: str) -> tuple[date, date] | None:
    clean = label.replace("–", "-")
    match = re.search(
        r"([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})\s+to\s+([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
        clean,
    )
    if not match:
        return None
    try:
        start = datetime.strptime(" ".join(match.group(1, 2, 3)), "%B %d %Y").date()
        end = datetime.strptime(" ".join(match.group(4, 5, 6)), "%B %d %Y").date()
    except ValueError:
        return None
    return start, end


def archive_set50_sources(
    *, root: Path, timeout_seconds: int = 45, retries: int = 3
) -> dict[str, Any]:
    """Archive SET's official listing page and every source it exposes in scope."""

    ensure_output_layout(root)
    archive_path, checksum, error, retrieved_at, attempts = download_original(
        url=SET_ARCHIVE_URL,
        market="Thailand",
        storage_year=utc_now().year,
        root=root,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    acquisition_rows: list[dict[str, Any]] = [
        {
            "market": "Thailand",
            "index_id": "SET50",
            "review_period": "ARCHIVE",
            "query": "Official SET50/SET100 constituent archive page",
            "source_checked": SET_ARCHIVE_URL,
            "result": "ACCEPTED_OFFICIAL_ARCHIVE" if archive_path else "FAILED",
            "accepted_source_id": None,
            "rejection_reason": error,
            "checked_at": retrieved_at.isoformat(),
        }
    ]
    if archive_path is None or checksum is None:
        log_path = root / "logs" / "set50_acquisition_v2.json"
        atomic_write_json(log_path, {"status": "BLOCKED", "archive_attempts": attempts, "error": error})
        return {"status": "BLOCKED", "catalog_path": None, "log_path": str(log_path), "source_count": 0}

    archive_source_id = stable_id("SRC", "SET50", SET_ARCHIVE_URL, checksum)
    entries = parse_set_archive_entries(archive_path)
    records: list[dict[str, Any]] = [
        {
            "source_id": archive_source_id,
            "index_id": "SET50",
            "market": "Thailand",
            "review_period": "ARCHIVE",
            "publisher": "The Stock Exchange of Thailand",
            "document_title": "SET50 and SET100 constituents archive page",
            "original_url": SET_ARCHIVE_URL,
            "raw_path": relative_path(archive_path, root),
            "content_type": "text/html",
            "source_type": "OFFICIAL_ARCHIVE_PAGE",
            "source_quality": "OFFICIAL_EXCHANGE_ARCHIVE",
            "publication_date": None,
            "announcement_date": None,
            "stated_effective_date": None,
            "retrieval_timestamp": retrieved_at,
            "file_checksum": checksum,
            "parser_version": PARSER_VERSION,
            "evidence_scope": "OFFICIAL_ARCHIVE_DISCOVERY",
            "source_status": "ACCEPTED",
            "notes": "Rendered official archive entry; its labelled links are separately retained as immutable originals.",
        }
    ]
    acquisition_rows[0]["accepted_source_id"] = archive_source_id

    methodology_path, methodology_checksum, methodology_error, methodology_retrieved_at, _methodology_attempts = download_original(
        url=SET_METHODOLOGY_URL,
        market="Thailand",
        storage_year=2025,
        root=root,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    methodology_source_id = (
        stable_id("SRC", "SET50", SET_METHODOLOGY_URL, methodology_checksum)
        if methodology_path is not None and methodology_checksum is not None
        else None
    )
    acquisition_rows.append(
        {
            "market": "Thailand",
            "index_id": "SET50",
            "review_period": "METHODOLOGY",
            "query": "Official SET index ground rules: review frequency and extraordinary constituent changes",
            "source_checked": SET_METHODOLOGY_URL,
            "result": "ACCEPTED_OFFICIAL_METHODOLOGY" if methodology_source_id else "DOWNLOAD_FAILED",
            "accepted_source_id": methodology_source_id,
            "rejection_reason": methodology_error,
            "checked_at": methodology_retrieved_at.isoformat(),
        }
    )
    if methodology_source_id and methodology_path and methodology_checksum:
        records.append(
            {
                "source_id": methodology_source_id,
                "index_id": "SET50",
                "market": "Thailand",
                "review_period": "METHODOLOGY",
                "publisher": "The Stock Exchange of Thailand",
                "document_title": "Index Ground Rules",
                "original_url": SET_METHODOLOGY_URL,
                "raw_path": relative_path(methodology_path, root),
                "content_type": "application/pdf",
                "source_type": "OFFICIAL_INDEX_METHODOLOGY",
                "source_quality": "OFFICIAL_EXCHANGE_OR_INDEX_PROVIDER_METHODOLOGY",
                "publication_date": None,
                "announcement_date": None,
                "stated_effective_date": None,
                "retrieval_timestamp": methodology_retrieved_at,
                "file_checksum": methodology_checksum,
                "parser_version": PARSER_VERSION,
                "evidence_scope": "REVIEW_FREQUENCY_CONSTITUENT_COUNT_AND_EXTRAORDINARY_RULES",
                "source_status": "ACCEPTED",
                "notes": "Official ground rules are retained to support the SET50 twice-yearly review convention and treatment of corporate-action changes; the methodology is not a constituent snapshot.",
            }
        )

    for entry in entries:
        raw_path, file_checksum, fetch_error, source_retrieved_at, source_attempts = download_original(
            url=entry["url"],
            market="Thailand",
            storage_year=entry["effective_start"].year,
            root=root,
            timeout_seconds=timeout_seconds,
            retries=retries,
        )
        source_id = (
            stable_id("SRC", "SET50", entry["url"], file_checksum)
            if raw_path is not None and file_checksum is not None
            else None
        )
        acquisition_rows.append(
            {
                "market": "Thailand",
                "index_id": "SET50",
                "review_period": entry["review_period"],
                "query": f"Official SET archive entry: {entry['label']}",
                "source_checked": entry["url"],
                "result": "ACCEPTED_OFFICIAL_SOURCE" if source_id else "DOWNLOAD_FAILED",
                "accepted_source_id": source_id,
                "rejection_reason": fetch_error,
                "checked_at": source_retrieved_at.isoformat(),
            }
        )
        if source_id is None:
            continue
        records.append(
            {
                "source_id": source_id,
                "index_id": "SET50",
                "market": "Thailand",
                "review_period": entry["review_period"],
                "publisher": "The Stock Exchange of Thailand",
                "document_title": Path(urlparse(entry["url"]).path).name,
                "original_url": entry["url"],
                "raw_path": relative_path(raw_path, root),
                "content_type": content_type_for_path(entry["url"]),
                "source_type": "OFFICIAL_REVISED_CONSTITUENT_LIST" if entry["is_revision"] else "OFFICIAL_CONSTITUENT_LIST",
                "source_quality": "OFFICIAL_EXCHANGE_OR_INDEX_PROVIDER",
                "publication_date": None,
                "announcement_date": None,
                "stated_effective_date": entry["effective_start"],
                "retrieval_timestamp": source_retrieved_at,
                "file_checksum": file_checksum,
                "parser_version": PARSER_VERSION,
                "evidence_scope": "DIRECT_CONSTITUENT_SNAPSHOT" if not entry["is_revision"] else "REVIEWED_REVISION_REQUIRES_EFFECTIVE_DATE",
                "source_status": "ACCEPTED",
                "notes": json.dumps(
                    {
                        "archive_source_id": archive_source_id,
                        "archive_period_label": entry["label"],
                        "period_start": date_to_iso(entry["effective_start"]),
                        "period_end_inclusive": date_to_iso(entry["effective_end_inclusive"]),
                        "is_revision": bool(entry["is_revision"]),
                    },
                    sort_keys=True,
                ),
            }
        )

    catalog = {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": utc_now().isoformat(),
        "source_records": [{key: (date_to_iso(value) if isinstance(value, date) else value) for key, value in row.items()} for row in records],
        "acquisition_attempts": acquisition_rows,
    }
    catalog_path = root / "interim" / "set50_official_source_catalog_v2.json"
    atomic_write_json(catalog_path, catalog)
    log_path = root / "logs" / "set50_acquisition_v2.json"
    atomic_write_json(log_path, {"status": "COMPLETE", "archive_attempts": attempts, "source_count": len(records), "period_source_count": len(entries)})
    return {
        "status": "COMPLETE",
        "catalog_path": str(catalog_path),
        "log_path": str(log_path),
        "source_count": len(records),
        "period_source_count": len(entries),
    }


def _parse_iso_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    parsed = parse_date(value)
    return parsed if parsed is not None else value


def load_set_catalog(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = root / "interim" / "set50_official_source_catalog_v2.json"
    if not path.exists():
        return [], []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for raw in payload.get("source_records", []):
        row = dict(raw)
        for field in ("publication_date", "announcement_date", "stated_effective_date"):
            row[field] = parse_date(row.get(field))
        retrieved = row.get("retrieval_timestamp")
        row["retrieval_timestamp"] = (
            datetime.fromisoformat(retrieved.replace("Z", "+00:00")) if isinstance(retrieved, str) else retrieved
        )
        records.append(row)
    return records, [dict(row) for row in payload.get("acquisition_attempts", [])]


def parse_pdf_publication_date(text: str) -> date | None:
    patterns = (
        r"(?:Published|Update(?:d)?)\s+on\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})",
        r"(?:Published|Update(?:d)?)\s*:\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            parsed = parse_date(match.group(1))
            if parsed:
                return parsed
    return None


def parse_set50_constituent_pdf(path: Path) -> dict[str, Any]:
    """Parse official SET50 PDF text while treating non-50 output as a hard failure.

    The SET PDFs are laid out in two columns.  `pypdf` emits the columns in a
    sequence where the reserve list may be interleaved with ranks 21--30.  The
    parser consequently selects the first valid official ticker for each rank
    1--50 rather than relying on line order.  It never fills a missing rank.
    """

    if PdfReader is None:
        return {"status": "PARSER_UNAVAILABLE", "error": "pypdf is not installed"}
    try:
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:  # pragma: no cover - source-specific diagnostics
        return {"status": "PARSE_FAILED", "error": f"{type(exc).__name__}: {exc}"}
    if not text.strip():
        return {"status": "PARSE_FAILED", "error": "No text extracted from PDF"}

    start = text.find("SET50 INDEX CONSTITUENTS")
    end = text.find("SET100 INDEX CONSTITUENTS", start + 1) if start >= 0 else -1
    if start < 0 or end < 0 or end <= start:
        # The newer official PDFs use a separate titled page where the first
        # table is SET50 and the following titled table is SET100.  Both table
        # labels occur in extracted text, even though the older section title
        # does not.
        start = text.find("No Symbol")
        end = text.find("No Symbol SET100", start + 1) if start >= 0 else -1
        if end < 0 and start >= 0:
            end = text.find("SET100", start + 100)
    if start < 0 or end < 0 or end <= start:
        return {"status": "PARSE_FAILED", "error": "SET50/SET100 section boundary not found", "text": text}
    section = text[start:end]
    candidate_pattern = re.compile(
        r"(?<!\d)(?P<rank>50|[1-4]?\d)\s*\*?\s*(?P<ticker>[A-Z][A-Z0-9.\-]{0,18})\s+"
    )
    invalid_tokens = {
        "INDEX",
        "COMPANY",
        "PUBLIC",
        "RESERVE",
        "LIST",
        "SET50",
        "INCLUSION",
        "EXCLUSION",
    }
    tickers_by_rank: dict[int, str] = {}
    for match in candidate_pattern.finditer(section):
        rank = int(match.group("rank"))
        ticker = match.group("ticker")
        # pypdf joins this one official 2022 rendering as
        # "INTUCHINTOUCH"; the exact published security code is the leading
        # INTUCH token and the remainder is the display-name start.  This is a
        # source-format exception, not a fuzzy match.
        if ticker == "INTUCHINTOUCH":
            ticker = "INTUCH"
        if not 1 <= rank <= 50 or ticker in invalid_tokens:
            continue
        tickers_by_rank.setdefault(rank, ticker)
    missing = [rank for rank in range(1, 51) if rank not in tickers_by_rank]
    tickers = [tickers_by_rank[rank] for rank in range(1, 51) if rank in tickers_by_rank]
    if missing or len(set(tickers)) != 50:
        return {
            "status": "PARSE_FAILED",
            "error": f"Expected exactly 50 distinct SET50 ranks; missing={missing}, distinct={len(set(tickers))}",
            "text": text,
            "tickers": tickers,
        }

    inclusion = extract_set50_event_tickers(section, "Inclusion")
    exclusion = extract_set50_event_tickers(section, "Exclusion")
    return {
        "status": "PARSED",
        "text": text,
        "tickers": tickers,
        "publication_date": parse_pdf_publication_date(text),
        "inclusion": inclusion,
        "exclusion": exclusion,
        "page_count": len(reader.pages),
    }


def extract_set50_event_tickers(section: str, label: str) -> list[str]:
    match = re.search(
        rf"SET50\s+{label}\s*:\s*(.*?)(?=SET50\s+(?:Inclusion|Exclusion)\s*:|Disclaimer:|For calculating|$)",
        section,
        re.I | re.S,
    )
    if not match:
        return []
    candidate_text = match.group(1)
    tickers = re.findall(r"\b[A-Z][A-Z0-9.\-]{0,9}\b", candidate_text)
    ignored = {"SET", "INDEX", "PUBLIC", "COMPANY", "LIMITED", "INCLUSION", "EXCLUSION"}
    return [ticker for ticker in tickers if ticker not in ignored]


def parse_source_notes(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def process_set50_sources(
    *, root: Path, source_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return source updates, extracted records, interval anchors and QA conflicts."""

    updated_sources: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    anchors: list[dict[str, Any]] = []
    reconciliation: list[dict[str, Any]] = []
    constituent_source_types = {"OFFICIAL_CONSTITUENT_LIST", "OFFICIAL_REVISED_CONSTITUENT_LIST"}
    source_by_period: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for source in source_rows:
        if source.get("index_id") == "SET50" and source.get("source_type") in constituent_source_types:
            source_by_period[str(source.get("review_period"))].append(source)

    for source in source_rows:
        row = dict(source)
        if row.get("index_id") != "SET50" or row.get("source_type") not in constituent_source_types:
            updated_sources.append(row)
            continue
        raw_path = root / str(row.get("raw_path"))
        parsed = parse_set50_constituent_pdf(raw_path) if raw_path.exists() else {"status": "RAW_FILE_MISSING"}
        if parsed.get("status") != "PARSED":
            row["source_status"] = "ACCEPTED_UNPARSED_REVIEW_REQUIRED"
            row["notes"] = str(row.get("notes") or "") + f" | parser failure: {parsed.get('error', parsed['status'])}"
            updated_sources.append(row)
            reconciliation.append(
                {
                    "log_id": stable_id("RCL", row.get("source_id"), "parse-failure"),
                    "issue_type": "OFFICIAL_SOURCE_PARSE_FAILURE",
                    "severity": "HIGH",
                    "index_id": "SET50",
                    "market": "Thailand",
                    "security_id": None,
                    "source_id": row.get("source_id"),
                    "event_date": None,
                    "status": "OPEN",
                    "details": str(parsed.get("error", parsed.get("status"))),
                }
            )
            continue
        publication_date = parsed.get("publication_date")
        row["publication_date"] = publication_date
        row["parser_version"] = PARSER_VERSION
        row["source_status"] = "PARSED"
        updated_sources.append(row)
        notes = parse_source_notes(row.get("notes"))
        is_revision = bool(notes.get("is_revision")) or row.get("source_type") == "OFFICIAL_REVISED_CONSTITUENT_LIST"
        period_start = parse_date(notes.get("period_start")) or row.get("stated_effective_date")
        period_end_inclusive = parse_date(notes.get("period_end_inclusive"))
        source_id = str(row["source_id"])
        review_period = str(row.get("review_period"))
        for sequence, ticker in enumerate(parsed["tickers"], start=1):
            events.append(
                {
                    "index_id": "SET50",
                    "market": "Thailand",
                    "security_name_as_published": None,
                    "ticker_as_published": ticker,
                    "event_type": "CONSTITUENT_SNAPSHOT_REVISION" if is_revision else "CONSTITUENT_SNAPSHOT",
                    "announcement_date": publication_date,
                    "effective_date": None if is_revision else period_start,
                    "review_period": review_period,
                    "source_id": source_id,
                    "page_or_section": "SET50 INDEX CONSTITUENTS",
                    "extraction_method": "pypdf_ranked_two_column_set50_parser",
                    "confidence": 0.98,
                    "manual_review_required": bool(is_revision),
                    "source_record": stable_id("EVT", source_id, "snapshot", sequence, ticker),
                    "notes": "Revision list does not itself state the implementation date." if is_revision else "Official full constituent list with stated calculation period.",
                }
            )
        for event_type, tickers in (("ADDITION", parsed["inclusion"]), ("REMOVAL", parsed["exclusion"])):
            for sequence, ticker in enumerate(tickers, start=1):
                events.append(
                    {
                        "index_id": "SET50",
                        "market": "Thailand",
                        "security_name_as_published": None,
                        "ticker_as_published": ticker,
                        "event_type": event_type,
                        "announcement_date": publication_date,
                        "effective_date": None if is_revision else period_start,
                        "review_period": review_period,
                        "source_id": source_id,
                        "page_or_section": f"SET50 {event_type.title()}",
                        "extraction_method": "pypdf_explicit_set50_inclusion_exclusion_parser",
                        "confidence": 0.96,
                        "manual_review_required": bool(is_revision),
                        "source_record": stable_id("EVT", source_id, event_type, sequence, ticker),
                        "notes": "Explicit source text; effective date is withheld because revision list does not state it." if is_revision else "Explicit source text and listed period start.",
                    }
                )
        if is_revision:
            reconciliation.append(
                {
                    "log_id": stable_id("RCL", source_id, "revision-effective-date"),
                    "issue_type": "REVISION_WITHOUT_STATED_EFFECTIVE_DATE",
                    "severity": "HIGH",
                    "index_id": "SET50",
                    "market": "Thailand",
                    "security_id": None,
                    "source_id": source_id,
                    "event_date": publication_date,
                    "status": "OPEN",
                    "details": "The official revised constituent list gives the calculation period and publication date but not an explicit implementation date. Publication and effective dates are kept separate; no interval is inferred.",
                }
            )
            continue
        if not isinstance(period_start, date) or not isinstance(period_end_inclusive, date):
            reconciliation.append(
                {
                    "log_id": stable_id("RCL", source_id, "no-period-boundary"),
                    "issue_type": "OFFICIAL_SNAPSHOT_MISSING_STATED_PERIOD_BOUNDARY",
                    "severity": "HIGH",
                    "index_id": "SET50",
                    "market": "Thailand",
                    "security_id": None,
                    "source_id": source_id,
                    "event_date": publication_date,
                    "status": "OPEN",
                    "details": "A full list was parsed but the archive did not provide both date boundaries.",
                }
            )
            continue
        revisions = [
            candidate
            for candidate in source_by_period.get(review_period, [])
            if candidate.get("source_type") == "OFFICIAL_REVISED_CONSTITUENT_LIST"
        ]
        if revisions:
            # A periodic list provides an anchor at its stated start, but a later
            # revision without an effective date makes the full interval unsafe.
            reconciliation.append(
                {
                    "log_id": stable_id("RCL", source_id, "period-superseded-by-undated-revision"),
                    "issue_type": "INCOMPLETE_EVENT_CHAIN_FOR_PERIOD",
                    "severity": "HIGH",
                    "index_id": "SET50",
                    "market": "Thailand",
                    "security_id": None,
                    "source_id": source_id,
                    "event_date": period_start,
                    "status": "OPEN",
                    "details": "A full anchor list exists, but an official between-review revision is retained without a stated effective date. No daily interval is reconstructed for this period.",
                }
            )
            continue
        anchors.append(
            {
                "index_id": "SET50",
                "market": "Thailand",
                "source_id": source_id,
                "announcement_date": publication_date,
                "effective_from": period_start,
                "effective_to": period_end_inclusive + timedelta(days=1),
                "tickers": list(parsed["tickers"]),
                "review_period": review_period,
            }
        )
    return updated_sources, events, anchors, reconciliation


EXTERNAL_INVENTORY_COLUMNS = {
    "source_id": ("source_id", "source_id_candidate"),
    "index_id": ("index_id",),
    "market": ("market",),
    "review_period": ("review_period",),
    "publisher": ("publisher",),
    "document_title": ("document_title", "title"),
    "original_url": ("original_url", "url", "source_url"),
    "raw_path": ("raw_path", "cached_path"),
    "content_type": ("content_type",),
    "source_type": ("source_type",),
    "source_quality": ("source_quality", "quality_rank"),
    "publication_date": ("publication_date",),
    "announcement_date": ("announcement_date",),
    "stated_effective_date": ("stated_effective_date", "effective_date"),
    "retrieval_timestamp": ("retrieval_timestamp", "retrieved_at", "checked_at"),
    "file_checksum": ("file_checksum", "sha256", "checksum"),
    "parser_version": ("parser_version",),
    "evidence_scope": ("evidence_scope", "scope"),
    "source_status": ("source_status", "status"),
    "notes": ("notes",),
}


def _first_present(row: Mapping[str, Any], choices: Sequence[str]) -> Any:
    for name in choices:
        value = row.get(name)
        if value is not None and str(value).strip() not in {"", "nan", "NaN"}:
            return value
    return None


def load_external_inventories(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load new v2 research inventory files without making them historical facts.

    An external source is accepted only when it has an immutable raw path and a
    matching SHA-256.  Otherwise it remains a documented search result in the
    search log/reconciliation output rather than entering source evidence.
    """

    accepted: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    interim = root / "interim"
    patterns = ("*_source_inventory_v2*.csv", "*_source_inventory_v2*.json")
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(sorted(interim.glob(pattern)))
    for file_path in sorted(set(candidates)):
        if file_path.name == "set50_official_source_catalog_v2.json":
            continue
        if file_path.suffix.lower() == ".csv":
            raw_rows: list[dict[str, Any]] = pd.read_csv(file_path, dtype=str, keep_default_na=False).to_dict("records")
        else:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
            raw_rows = payload.get("source_records", payload) if isinstance(payload, dict) else payload
            raw_rows = [dict(row) for row in raw_rows] if isinstance(raw_rows, list) else []
        for raw in raw_rows:
            row = {canonical: _first_present(raw, aliases) for canonical, aliases in EXTERNAL_INVENTORY_COLUMNS.items()}
            index_id = str(row.get("index_id") or "")
            if index_id not in INDEX_CONFIG:
                search_rows.append(
                    {
                        "market": row.get("market") or "UNKNOWN",
                        "index_id": index_id or "UNKNOWN",
                        "review_period": row.get("review_period") or "UNKNOWN",
                        "query": f"External inventory validation: {file_path.name}",
                        "source_checked": row.get("original_url") or str(file_path),
                        "result": "REJECTED_INVENTORY_ROW",
                        "accepted_source_id": None,
                        "rejection_reason": "Unknown or missing index_id",
                        "checked_at": utc_now().isoformat(),
                    }
                )
                continue
            row["market"] = row.get("market") or INDEX_CONFIG[index_id]["market"]
            row["publisher"] = row.get("publisher") or "Unspecified publisher"
            row["source_quality"] = str(row.get("source_quality") or "UNASSESSED").strip().upper()
            row["source_type"] = str(row.get("source_type") or "EXTERNAL_RESEARCH_SOURCE").strip().upper()
            supplied_status = str(row.get("source_status") or "ACCEPTED").strip().upper()
            if supplied_status.startswith("ACCEPTED"):
                row["source_status"] = "ACCEPTED"
            elif supplied_status.startswith(("CANDIDATE", "REJECTED", "UNAVAILABLE", "FAILED")):
                row["source_status"] = "REJECTED"
            else:
                row["source_status"] = supplied_status
            row["parser_version"] = row.get("parser_version") or "external-inventory-v2"
            row["evidence_scope"] = row.get("evidence_scope") or "UNPARSED_REVIEW_REQUIRED"
            for field in ("publication_date", "announcement_date", "stated_effective_date"):
                row[field] = parse_date(row.get(field))
            stamp = row.get("retrieval_timestamp")
            if isinstance(stamp, str) and stamp:
                try:
                    row["retrieval_timestamp"] = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                except ValueError:
                    row["retrieval_timestamp"] = utc_now()
            elif not isinstance(stamp, datetime):
                row["retrieval_timestamp"] = utc_now()
            raw_path_value = row.get("raw_path")
            supplied_raw_path = Path(str(raw_path_value)) if raw_path_value else None
            if supplied_raw_path and supplied_raw_path.is_absolute():
                try:
                    raw_path = supplied_raw_path.resolve().relative_to(root.resolve())
                    raw_path = root / raw_path
                except ValueError:
                    raw_path = None
            else:
                candidates = [root / supplied_raw_path] if supplied_raw_path else []
                # Intake work may record a repository-relative path such as
                # research_data/point_in_time/raw_sources/... rather than a
                # point-in-time-root-relative one.  Normalize it only when it
                # resolves under this same output root.
                if supplied_raw_path:
                    candidates.append(root.parent.parent / supplied_raw_path)
                raw_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0] if candidates else None)
            expected_checksum = str(row.get("file_checksum") or "")
            if raw_path is None or not raw_path.is_file() or not expected_checksum:
                reason = "Missing immutable raw_path or SHA-256"
            elif sha256_file(raw_path) != expected_checksum:
                reason = "Raw file checksum does not match inventory SHA-256"
            elif row["source_status"] == "REJECTED":
                reason = f"Inventory status does not admit this source as evidence: {supplied_status}"
            else:
                reason = None
            if reason:
                search_rows.append(
                    {
                        "market": row["market"],
                        "index_id": index_id,
                        "review_period": row.get("review_period") or "UNKNOWN",
                        "query": f"External inventory validation: {file_path.name}",
                        "source_checked": row.get("original_url") or str(raw_path or file_path),
                        "result": "REJECTED_INVENTORY_ROW",
                        "accepted_source_id": None,
                        "rejection_reason": reason,
                        "checked_at": utc_now().isoformat(),
                    }
                )
                continue
            row["raw_path"] = relative_path(raw_path, root)
            row["source_id"] = row.get("source_id") or stable_id("SRC", index_id, row.get("original_url"), expected_checksum)
            row["content_type"] = row.get("content_type") or content_type_for_path(str(raw_path))
            accepted.append(row)
            search_rows.append(
                {
                    "market": row["market"],
                    "index_id": index_id,
                    "review_period": row.get("review_period") or "UNKNOWN",
                    "query": f"External inventory validation: {file_path.name}",
                    "source_checked": row.get("original_url") or str(raw_path),
                    "result": "ACCEPTED_CHECKSUMMED_SOURCE",
                    "accepted_source_id": row["source_id"],
                    "rejection_reason": None,
                    "checked_at": utc_now().isoformat(),
                }
            )
    return accepted, search_rows


def load_external_search_logs(root: Path) -> list[dict[str, Any]]:
    """Load researchers' period-specific search results without changing evidence.

    A search log is evidence of the search itself, not proof that an unaccepted
    result contains no historical membership information.  Consequently the
    loader preserves both accepted and rejected checks exactly as logged and
    only fills an absent timestamp with the intake time.
    """

    rows: list[dict[str, Any]] = []
    interim = root / "interim"
    candidates: list[Path] = []
    for pattern in ("*_source_search_log_v2*.csv", "*_source_search_log_v2*.json"):
        candidates.extend(sorted(interim.glob(pattern)))
    for path in sorted(set(candidates)):
        if path.suffix.lower() == ".csv":
            raw_rows: list[dict[str, Any]] = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
            raw_rows = [dict(item) for item in raw_rows] if isinstance(raw_rows, list) else []
        for raw in raw_rows:
            record = normalise_search_log_row(raw)
            if not record["market"] and record["index_id"] in INDEX_CONFIG:
                record["market"] = INDEX_CONFIG[str(record["index_id"])]["market"]
            record["checked_at"] = record["checked_at"] or utc_now().isoformat()
            rows.append(record)
    return rows


def deduplicate_source_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate only exact source content and report conflicts instead of choosing one."""

    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    seen_ids: dict[str, Mapping[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        key = (str(row.get("original_url")), str(row.get("file_checksum")))
        source_id = str(row.get("source_id"))
        if source_id in seen_ids and key != (str(seen_ids[source_id].get("original_url")), str(seen_ids[source_id].get("file_checksum"))):
            conflicts.append(
                {
                    "log_id": stable_id("RCL", source_id, "duplicate-source-id"),
                    "issue_type": "DUPLICATE_SOURCE_ID_WITH_DIFFERENT_CONTENT",
                    "severity": "HIGH",
                    "index_id": row.get("index_id"),
                    "market": row.get("market"),
                    "security_id": None,
                    "source_id": source_id,
                    "event_date": None,
                    "status": "OPEN",
                    "details": "Source ID maps to more than one URL/checksum pair; neither record is silently preferred.",
                }
            )
        seen_ids[source_id] = row
        if key not in deduped:
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: (str(item.get("index_id")), str(item.get("source_id")))), conflicts


def create_security_and_intervals(
    anchors: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve official exchange-code anchors without fuzzy matching.

    Each code is an official security code published in a constituent list.
    The identity rule is intentionally narrow: exact market + exchange +
    official code.  A future raw source that shows a ticker reuse can surface a
    conflict; this function does not merge by display name.
    """

    security_sources: defaultdict[str, set[str]] = defaultdict(set)
    security_metadata: dict[str, dict[str, Any]] = {}
    memberships: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    seen_membership: set[tuple[str, str, date, date, str]] = set()
    for anchor in sorted(anchors, key=lambda item: (item["effective_from"], item["source_id"])):
        index_id = str(anchor["index_id"])
        config = INDEX_CONFIG[index_id]
        tickers = list(anchor["tickers"])
        if len(tickers) != config["constituent_count"] or len(set(tickers)) != len(tickers):
            conflicts.append(
                {
                    "log_id": stable_id("RCL", anchor["source_id"], "anchor-count"),
                    "issue_type": "ANCHOR_CONSTITUENT_COUNT_INVALID",
                    "severity": "HIGH",
                    "index_id": index_id,
                    "market": config["market"],
                    "security_id": None,
                    "source_id": anchor["source_id"],
                    "event_date": anchor["effective_from"],
                    "status": "OPEN",
                    "details": f"Expected {config['constituent_count']} distinct members, parsed {len(tickers)} / {len(set(tickers))}.",
                }
            )
            continue
        for ordinal, ticker in enumerate(tickers, start=1):
            security_id = stable_id("SEC", config["market"], config["exchange"], ticker)
            security_sources[security_id].add(str(anchor["source_id"]))
            security_metadata.setdefault(
                security_id,
                {
                    "market": config["market"],
                    "exchange": config["exchange"],
                    "official_exchange_code": ticker,
                    "legal_name": dict(anchor.get("security_names") or {}).get(ticker),
                },
            )
            membership_key = (
                index_id,
                security_id,
                anchor["effective_from"],
                anchor["effective_to"],
                str(anchor["source_id"]),
            )
            if membership_key in seen_membership:
                continue
            seen_membership.add(membership_key)
            memberships.append(
                {
                    "index_id": index_id,
                    "market": config["market"],
                    "security_id": security_id,
                    "effective_from": anchor["effective_from"],
                    "effective_to": anchor["effective_to"],
                    "event_type": "DIRECT_CONSTITUENT_SNAPSHOT",
                    "announcement_date": anchor.get("announcement_date"),
                    "source_id": anchor["source_id"],
                    "reconstruction_method": anchor.get("reconstruction_method") or "direct_snapshot",
                    "anchor_source_id": anchor["source_id"],
                    "event_chain_complete": True,
                    "evidence_count": 1,
                    "confidence": float(anchor.get("confidence") or 0.98),
                    "resolution_status": "RESOLVED_OFFICIAL_EXCHANGE_CODE",
                    "notes": anchor.get("notes")
                    or f"Official {index_id} constituent list states the calculation period; effective_to is exclusive.",
                }
            )
            ticker_rows.append(
                {
                    "security_id": security_id,
                    "ticker": ticker,
                    "exchange_suffix": config["exchange_suffix"],
                    "valid_from": anchor["effective_from"],
                    "valid_to": anchor["effective_to"],
                    "source_id": anchor["source_id"],
                    "match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE",
                    "confidence": float(anchor.get("confidence") or 0.98),
                    "resolution_status": "RESOLVED",
                }
            )
            candidates.append(
                {
                    "candidate_id": stable_id("ERC", anchor["source_id"], ordinal, ticker),
                    "source_record": f"{anchor['source_id']}:{index_id}_INDEX_CONSTITUENTS:rank={ordinal}",
                    "market": config["market"],
                    "index_id": index_id,
                    "ticker_as_published": ticker,
                    "security_name_as_published": None,
                    "candidate_security_id": security_id,
                    "match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE",
                    "similarity_score": 1.0,
                    "date_consistency": True,
                    "exchange_consistency": True,
                    "decision": "RESOLVED",
                    "manual_review_required": False,
                    "review_notes": "No fuzzy matching used; official exchange code is the match key.",
                }
            )
    securities: list[dict[str, Any]] = []
    for security_id, source_ids in sorted(security_sources.items()):
        matching = next((row for row in ticker_rows if row["security_id"] == security_id), None)
        if matching is None:
            continue
        metadata = security_metadata[security_id]
        securities.append(
            {
                "security_id": security_id,
                "market": metadata["market"],
                "exchange": metadata["exchange"],
                "official_exchange_code": metadata["official_exchange_code"],
                "isin": None,
                "legal_name": metadata["legal_name"],
                "listing_date": None,
                "delisting_date": None,
                "source_count": len(source_ids),
                "identity_match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE",
                "resolution_status": "RESOLVED_OFFICIAL_EXCHANGE_CODE",
                "manual_review_required": False,
                "notes": "Identity is based on exact official exchange security code; legal-name and ISIN enrichment remains separate.",
            }
        )
    return securities, ticker_rows, memberships, candidates, conflicts


def enrich_identity_from_observed_events(
    *,
    securities: Sequence[Mapping[str, Any]],
    ticker_history: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve identity-only observations without promoting them to membership.

    An exact official exchange code on a dated constituent snapshot establishes
    a security identity even when interval boundaries are too ambiguous for a
    daily PIT state.  Such observations get a single-day ticker observation and
    an explicit `RESOLVED_IDENTITY_ONLY` decision; they are never added to
    ``index_membership_v2`` or used to construct price requirements.
    """

    sources = {str(row.get("source_id")): row for row in source_rows}
    security_by_id = {str(row.get("security_id")): dict(row) for row in securities}
    ticker_rows = [dict(row) for row in ticker_history]
    candidate_by_id = {str(row.get("candidate_id")): dict(row) for row in candidates}
    source_ids_by_security: defaultdict[str, set[str]] = defaultdict(set)
    for security in security_by_id.values():
        # Existing source_count is retained as a lower bound; source-specific
        # increments from interval anchors are already represented there.
        source_ids_by_security[str(security["security_id"])].update(
            {f"existing-count-{position}" for position in range(int(security.get("source_count") or 0))}
        )
    observed_ticker_keys = {
        (str(row.get("security_id")), str(row.get("ticker")), row.get("valid_from"), row.get("valid_to"), str(row.get("source_id")))
        for row in ticker_rows
    }
    for event in events:
        source_id = str(event.get("source_id") or "")
        source = sources.get(source_id)
        ticker = str(event.get("ticker_as_published") or "").strip()
        index_id = str(event.get("index_id") or "")
        if (
            index_id not in INDEX_CONFIG
            or source is None
            or not str(source.get("source_quality", "")).startswith("OFFICIAL")
            or (_parse_confidence(event.get("confidence")) or 0.0) < 0.95
        ):
            continue
        config = INDEX_CONFIG[index_id]
        if not ticker:
            # A published name without an exchange code is deliberately kept
            # in the unresolved queue.  No fuzzy comparison is attempted or
            # used as a final decision.
            if event.get("event_type") in {"ADDITION", "REMOVAL", "NAME_CHANGE", "TICKER_CHANGE", "MERGER", "DELETION", "CONSTITUENT_SNAPSHOT"} and event.get("security_name_as_published"):
                candidate_id = stable_id("ERC", event.get("source_record"), "name-only-unresolved")
                candidate_by_id.setdefault(
                    candidate_id,
                    {
                        "candidate_id": candidate_id,
                        "source_record": event.get("source_record"),
                        "market": config["market"],
                        "index_id": index_id,
                        "ticker_as_published": None,
                        "security_name_as_published": event.get("security_name_as_published"),
                        "candidate_security_id": None,
                        "match_rule": "NO_FINAL_FUZZY_MATCH",
                        "similarity_score": None,
                        "date_consistency": isinstance(event.get("effective_date"), date),
                        "exchange_consistency": True,
                        "decision": "UNRESOLVED_NAME_ONLY_OFFICIAL_EVENT",
                        "manual_review_required": True,
                        "review_notes": "Published source contains no exchange code/ticker; retained for manual identity resolution without fuzzy auto-match.",
                    },
                )
            continue
        security_id = stable_id("SEC", config["market"], config["exchange"], ticker)
        source_ids_by_security[security_id].add(source_id)
        if security_id not in security_by_id:
            security_by_id[security_id] = {
                "security_id": security_id,
                "market": config["market"],
                "exchange": config["exchange"],
                "official_exchange_code": ticker,
                "isin": None,
                "legal_name": event.get("security_name_as_published"),
                "listing_date": None,
                "delisting_date": None,
                "source_count": 0,
                "identity_match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE",
                "resolution_status": "RESOLVED_IDENTITY_ONLY",
                "manual_review_required": False,
                "notes": "Exact official code observed in retained source evidence; no membership interval was inferred from this identity-only observation.",
            }
        event_date = event.get("effective_date")
        if isinstance(event_date, date):
            ticker_key = (security_id, ticker, event_date, event_date + timedelta(days=1), source_id)
            if ticker_key not in observed_ticker_keys:
                observed_ticker_keys.add(ticker_key)
                ticker_rows.append(
                    {
                        "security_id": security_id,
                        "ticker": ticker,
                        "exchange_suffix": config["exchange_suffix"],
                        "valid_from": event_date,
                        "valid_to": event_date + timedelta(days=1),
                        "source_id": source_id,
                        "match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE_DATED_OBSERVATION",
                        "confidence": float(event.get("confidence") or 0.98),
                        "resolution_status": "RESOLVED_IDENTITY_ONLY",
                    }
                )
        candidate_id = stable_id("ERC", event.get("source_record"), "identity-only", security_id)
        candidate_by_id.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "source_record": event.get("source_record"),
                "market": config["market"],
                "index_id": index_id,
                "ticker_as_published": ticker,
                "security_name_as_published": event.get("security_name_as_published"),
                "candidate_security_id": security_id,
                "match_rule": "EXACT_OFFICIAL_EXCHANGE_SECURITY_CODE",
                "similarity_score": 1.0,
                "date_consistency": isinstance(event_date, date),
                "exchange_consistency": True,
                "decision": "RESOLVED_IDENTITY_ONLY",
                "manual_review_required": False,
                "review_notes": "Exact official exchange code only; membership-state reconstruction remains independently gated.",
            },
        )
    for security_id, security in security_by_id.items():
        observed_count = len(source_ids_by_security.get(security_id, set()))
        security["source_count"] = max(int(security.get("source_count") or 0), observed_count)
    return (
        sorted(security_by_id.values(), key=lambda item: str(item["security_id"])),
        sorted(ticker_rows, key=lambda item: (str(item["security_id"]), item.get("valid_from") or date.min, str(item.get("source_id")))),
        sorted(candidate_by_id.values(), key=lambda item: str(item["candidate_id"])),
    )


def validate_source_evidence(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    by_id: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_checksum: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_id[str(row.get("source_id"))].append(row)
        checksum = row.get("file_checksum")
        if checksum:
            by_checksum[str(checksum)].append(row)
        raw = row.get("raw_path")
        if raw and not str(raw).startswith("raw_sources/"):
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "RAW_SOURCE_OUTSIDE_ARCHIVE",
                    "source_id": row.get("source_id"),
                    "details": "Accepted source raw_path must be under raw_sources/.",
                }
            )
    for source_id, values in by_id.items():
        if len(values) > 1:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "DUPLICATE_SOURCE_EVIDENCE_ID",
                    "source_id": source_id,
                    "details": f"{len(values)} records share source_id.",
                }
            )
    for checksum, values in by_checksum.items():
        urls = {str(value.get("original_url")) for value in values}
        if len(urls) > 1:
            findings.append(
                {
                    "severity": "MEDIUM",
                    "rule": "SAME_CHECKSUM_MULTIPLE_URLS",
                    "source_id": None,
                    "details": f"Checksum {checksum} appears under {len(urls)} original URLs.",
                }
            )
    return findings


def validate_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in events:
        source_record = str(row.get("source_record"))
        if source_record in seen:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "DUPLICATE_EVENT_SOURCE_RECORD",
                    "source_id": row.get("source_id"),
                    "details": f"Duplicate source_record {source_record}",
                }
            )
        seen.add(source_record)
        announced, effective = row.get("announcement_date"), row.get("effective_date")
        if announced is not None and effective is not None and announced > effective:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "ANNOUNCEMENT_AFTER_EFFECTIVE_DATE",
                    "source_id": row.get("source_id"),
                    "details": "Announcement_date is after effective_date; fields may have been conflated or source needs review.",
                }
            )
        if row.get("event_type") in {"ADDITION", "REMOVAL", "REPLACEMENT"} and effective is None and not row.get("manual_review_required"):
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "UNQUALIFIED_EVENT_MISSING_EFFECTIVE_DATE",
                    "source_id": row.get("source_id"),
                    "details": "An explicit membership event lacks effective_date without manual-review flag.",
                }
            )
    return findings


def validate_intervals(
    memberships: Sequence[Mapping[str, Any]], ticker_history: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    by_pair: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    ticker_by_security: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for ticker in ticker_history:
        ticker_by_security[str(ticker.get("security_id"))].append(ticker)
    for row in memberships:
        index_id, security_id = str(row.get("index_id")), str(row.get("security_id"))
        start, end = row.get("effective_from"), row.get("effective_to")
        if not isinstance(start, date) or not isinstance(end, date) or end <= start:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "NON_POSITIVE_OR_MISSING_INTERVAL",
                    "source_id": row.get("source_id"),
                    "details": f"Invalid interval {start!r} -> {end!r}",
                }
            )
            continue
        if not row.get("event_chain_complete"):
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "ASSERTED_INTERVAL_WITH_INCOMPLETE_EVENT_CHAIN",
                    "source_id": row.get("source_id"),
                    "details": "Intervals are only permitted when event_chain_complete is true.",
                }
            )
        candidates = ticker_by_security.get(security_id, [])
        covers = any(
            isinstance(candidate.get("valid_from"), date)
            and isinstance(candidate.get("valid_to"), date)
            and candidate["valid_from"] <= start
            and candidate["valid_to"] >= end
            for candidate in candidates
        )
        if not covers:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "TICKER_VALIDITY_DOES_NOT_COVER_INTERVAL",
                    "source_id": row.get("source_id"),
                    "details": f"No dated ticker history covers {security_id} {start} -> {end}",
                }
            )
        by_pair[(index_id, security_id)].append(row)
    for key, rows in by_pair.items():
        ordered = sorted(rows, key=lambda item: (item["effective_from"], item["effective_to"]))
        previous_end: date | None = None
        for row in ordered:
            if previous_end and row["effective_from"] < previous_end:
                findings.append(
                    {
                        "severity": "HIGH",
                        "rule": "OVERLAPPING_MEMBERSHIP_INTERVAL",
                        "source_id": row.get("source_id"),
                        "details": f"Overlapping interval for {key[0]} / {key[1]}",
                    }
                )
            previous_end = max(previous_end, row["effective_to"]) if previous_end else row["effective_to"]
    return findings


def validate_constituent_counts(memberships: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    by_interval: defaultdict[tuple[str, date, date, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in memberships:
        by_interval[(str(row["index_id"]), row["effective_from"], row["effective_to"], str(row["source_id"]))].append(row)
    for (index_id, start, end, source_id), rows in by_interval.items():
        expected = INDEX_CONFIG[index_id]["constituent_count"]
        if len(rows) != expected:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "INDEX_CONSTITUENT_COUNT_MISMATCH",
                    "source_id": source_id,
                    "details": f"{index_id} interval {start} -> {end} has {len(rows)}, expected {expected}.",
                }
            )
    return findings


def validate_daily_snapshot_constituent_counts(snapshot_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Verify each emitted eligible PIT date has the declared index size."""

    findings: list[dict[str, Any]] = []
    groups: defaultdict[tuple[date, str], set[str]] = defaultdict(set)
    for row in snapshot_rows:
        on_date = row.get("date")
        index_id = str(row.get("index_id") or "")
        security_id = row.get("security_id")
        if isinstance(on_date, date) and index_id in INDEX_CONFIG and security_id:
            groups[(on_date, index_id)].add(str(security_id))
    for (on_date, index_id), security_ids in sorted(groups.items()):
        expected = INDEX_CONFIG[index_id]["constituent_count"]
        if len(security_ids) != expected:
            findings.append(
                {
                    "severity": "HIGH",
                    "rule": "DAILY_SNAPSHOT_CONSTITUENT_COUNT_MISMATCH",
                    "source_id": None,
                    "details": f"{index_id} {on_date.isoformat()} has {len(security_ids)} distinct snapshot members, expected {expected}.",
                }
            )
    return findings


def write_reconciliation_from_findings(
    existing: Sequence[Mapping[str, Any]], findings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in existing]
    for position, finding in enumerate(findings, start=1):
        rows.append(
            {
                "log_id": stable_id("RCL", finding.get("rule"), finding.get("source_id"), finding.get("details"), position),
                "issue_type": finding.get("rule"),
                "severity": finding.get("severity", "HIGH"),
                "index_id": finding.get("index_id"),
                "market": finding.get("market"),
                "security_id": finding.get("security_id"),
                "source_id": finding.get("source_id"),
                "event_date": finding.get("event_date"),
                "status": "OPEN",
                "details": finding.get("details"),
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped[str(row["log_id"])] = row
    return sorted(deduped.values(), key=lambda item: (str(item.get("index_id")), str(item.get("log_id"))))


def ticker_by_date(
    security_id: str, on_date: date, ticker_history: Sequence[Mapping[str, Any]]
) -> str | None:
    values = [
        row
        for row in ticker_history
        if row.get("security_id") == security_id
        and row.get("valid_from") <= on_date < row.get("valid_to")
        and row.get("resolution_status") == "RESOLVED"
    ]
    if len(values) != 1:
        return None
    return str(values[0].get("ticker"))


def build_snapshots(
    memberships: Sequence[Mapping[str, Any]],
    ticker_history: Sequence[Mapping[str, Any]],
    eligible_intervals: set[tuple[str, date, date, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for membership in sorted(memberships, key=lambda item: (item["effective_from"], item["security_id"])):
        key = (
            str(membership["index_id"]),
            membership["effective_from"],
            membership["effective_to"],
            str(membership["source_id"]),
        )
        if key not in eligible_intervals:
            continue
        start = max(membership["effective_from"], TARGET_START)
        end = min(membership["effective_to"], TARGET_END_EXCLUSIVE)
        if end <= start:
            continue
        for stamp in pd.date_range(start, end - timedelta(days=1), freq="D"):
            on_date = stamp.date()
            ticker = ticker_by_date(str(membership["security_id"]), on_date, ticker_history)
            if ticker is None:
                continue
            rows.append(
                {
                    "date": on_date,
                    "market": membership["market"],
                    "index_id": membership["index_id"],
                    "security_id": membership["security_id"],
                    "ticker_on_date": ticker,
                    "membership_effective_from": membership["effective_from"],
                    "membership_effective_to": membership["effective_to"],
                    "source_chain": f"{membership['reconstruction_method']}:{membership['anchor_source_id']}",
                    "confidence": membership["confidence"],
                    "eligibility_interval": f"{membership['effective_from'].isoformat()}/{membership['effective_to'].isoformat()}",
                    "snapshot_version": SNAPSHOT_VERSION,
                    "universe_semantics": PIT_SEMANTICS,
                    "is_evidence_backed": True,
                }
            )
    return sorted(rows, key=lambda item: (item["date"], item["index_id"], item["security_id"]))


RAW_OHLC_COLUMNS = {"open", "high", "low", "close"}


def _price_file_for_ticker(repo_root: Path, index_id: str, ticker: str) -> Path | None:
    config = INDEX_CONFIG[index_id]
    directory = repo_root / "data" / "cache" / "yahoo_daily_2016" / config["price_directory"]
    preferred = directory / f"{ticker}__{ticker}.csv"
    if preferred.is_file():
        return preferred
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"{ticker}__*.csv"))
    return matches[0] if matches else None


def _read_raw_ohlc_dates(path: Path) -> set[date]:
    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return set()
    lower = {str(column).strip().lower(): column for column in frame.columns}
    if not RAW_OHLC_COLUMNS.issubset(lower):
        return set()
    date_column = next((lower[name] for name in ("date", "datetime", "timestamp") if name in lower), None)
    if date_column is None:
        return set()
    values = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    raw_mask = pd.Series(True, index=frame.index)
    for field in RAW_OHLC_COLUMNS:
        raw_mask &= pd.to_numeric(frame[lower[field]], errors="coerce").notna()
    return {value for value in values[raw_mask].tolist() if isinstance(value, date)}


def local_price_coverage(
    *, repo_root: Path, memberships: Sequence[Mapping[str, Any]], ticker_history: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, date, date, str], float]]:
    """Assess raw OHLC only; adjusted-close data is never substituted.

    The expected days are local observed market dates from the union of raw-OHLC
    files, not calendar days.  That avoids falsely penalising weekends and
    exchange holidays while avoiding any forward fill after delisting.
    """

    all_dates_by_index: dict[str, set[date]] = defaultdict(set)
    ticker_dates: dict[tuple[str, str], set[date]] = {}
    checked: set[tuple[str, str]] = set()
    for membership in memberships:
        index_id = str(membership["index_id"])
        security_id = str(membership["security_id"])
        ticker = next(
            (
                str(row["ticker"])
                for row in ticker_history
                if row.get("security_id") == security_id
                and row.get("valid_from") <= membership["effective_from"]
                and row.get("valid_to") >= membership["effective_to"]
            ),
            None,
        )
        if not ticker or (index_id, ticker) in checked:
            continue
        checked.add((index_id, ticker))
        price_path = _price_file_for_ticker(repo_root, index_id, ticker)
        dates = _read_raw_ohlc_dates(price_path) if price_path else set()
        ticker_dates[(index_id, ticker)] = dates
        all_dates_by_index[index_id].update(dates)

    queue_rows: list[dict[str, Any]] = []
    interval_coverage: dict[tuple[str, date, date, str], float] = {}
    grouped_memberships: defaultdict[tuple[str, date, date, str], list[Mapping[str, Any]]] = defaultdict(list)
    for membership in memberships:
        grouped_memberships[
            (
                str(membership["index_id"]),
                membership["effective_from"],
                membership["effective_to"],
                str(membership["source_id"]),
            )
        ].append(membership)
        security_id = str(membership["security_id"])
        ticker = ticker_by_date(security_id, membership["effective_from"], ticker_history)
        if ticker is None:
            continue
        price_path = _price_file_for_ticker(repo_root, str(membership["index_id"]), ticker)
        dates = ticker_dates.get((str(membership["index_id"]), ticker), set())
        expected_dates = {
            value
            for value in all_dates_by_index[str(membership["index_id"])]
            if membership["effective_from"] <= value < membership["effective_to"]
        }
        covered_dates = dates.intersection(expected_dates)
        queue_rows.append(
            {
                "market": membership["market"],
                "security_id": security_id,
                "historical_ticker": ticker,
                "exchange_suffix": INDEX_CONFIG[str(membership["index_id"])]["exchange_suffix"],
                "membership_start": membership["effective_from"].isoformat(),
                "membership_end": (membership["effective_to"] - timedelta(days=1)).isoformat(),
                "required_price_start": membership["effective_from"].isoformat(),
                "required_price_end": (membership["effective_to"] - timedelta(days=1)).isoformat(),
                "preferred_provider": INDEX_CONFIG[str(membership["index_id"])]["preferred_provider"],
                "fallback_provider": INDEX_CONFIG[str(membership["index_id"])]["fallback_provider"],
                "local_data_exists": "YES_RAW_OHLC" if price_path and dates else "NO_RAW_OHLC",
                "download_status": "LOCAL_RAW_OHLC_COVERED" if expected_dates and len(covered_dates) == len(expected_dates) else "QUEUED",
                "failure_reason": (
                    None
                    if expected_dates and len(covered_dates) == len(expected_dates)
                    else "No raw OHLC file" if not price_path else "Raw OHLC file does not cover every observed market date; adjusted data was not substituted."
                ),
            }
        )
    for key, rows in grouped_memberships.items():
        index_id, start, end, _source_id = key
        expected_dates = {value for value in all_dates_by_index[index_id] if start <= value < end}
        expected_member_days = len(expected_dates) * len(rows)
        covered_member_days = 0
        for row in rows:
            ticker = ticker_by_date(str(row["security_id"]), start, ticker_history)
            if ticker:
                covered_member_days += len(ticker_dates.get((index_id, ticker), set()).intersection(expected_dates))
        interval_coverage[key] = covered_member_days / expected_member_days if expected_member_days else 0.0
    return queue_rows, interval_coverage


def append_identity_observation_price_queue(
    *,
    repo_root: Path,
    queue_rows: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    securities: Sequence[Mapping[str, Any]],
    ticker_history: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Queue raw OHLC for exact, dated snapshot observations without intervals.

    This preserves the useful distinction between knowing a member on one
    source-stated date and knowing a continuous membership interval.  A one-day
    price requirement is queued for the former; it is not used in interval
    price coverage or daily PIT snapshots.
    """

    rows = [dict(row) for row in queue_rows]
    interval_security_ids = {str(row.get("security_id")) for row in memberships}
    security_market = {str(row.get("security_id")): str(row.get("market")) for row in securities}
    index_by_market = {str(config["market"]): index_id for index_id, config in INDEX_CONFIG.items()}
    seen = {
        (str(row.get("security_id")), str(row.get("historical_ticker")), str(row.get("required_price_start")))
        for row in rows
    }
    for observation in ticker_history:
        if observation.get("resolution_status") != "RESOLVED_IDENTITY_ONLY":
            continue
        security_id = str(observation.get("security_id"))
        if security_id in interval_security_ids:
            continue
        observed_date = observation.get("valid_from")
        ticker = str(observation.get("ticker") or "")
        market = security_market.get(security_id)
        index_id = index_by_market.get(str(market))
        if not isinstance(observed_date, date) or not ticker or not index_id:
            continue
        key = (security_id, ticker, observed_date.isoformat())
        if key in seen:
            continue
        seen.add(key)
        price_path = _price_file_for_ticker(repo_root, index_id, ticker)
        raw_dates = _read_raw_ohlc_dates(price_path) if price_path else set()
        locally_covered = observed_date in raw_dates
        rows.append(
            {
                "market": market,
                "security_id": security_id,
                "historical_ticker": ticker,
                "exchange_suffix": INDEX_CONFIG[index_id]["exchange_suffix"],
                "membership_start": observed_date.isoformat(),
                "membership_end": observed_date.isoformat(),
                "required_price_start": observed_date.isoformat(),
                "required_price_end": observed_date.isoformat(),
                "preferred_provider": INDEX_CONFIG[index_id]["preferred_provider"],
                "fallback_provider": INDEX_CONFIG[index_id]["fallback_provider"],
                "local_data_exists": "YES_RAW_OHLC" if locally_covered else "NO_RAW_OHLC",
                "download_status": "LOCAL_RAW_OHLC_COVERED" if locally_covered else "QUEUED",
                "failure_reason": (
                    "Identity-only exact source observation; no continuous membership interval was inferred."
                    if locally_covered
                    else "No raw OHLC for this exact source-stated observation date; no adjusted-data substitution or forward-fill was used."
                ),
            }
        )
    return sorted(rows, key=lambda row: (str(row.get("market")), str(row.get("security_id")), str(row.get("required_price_start"))))


def build_eligibility_rows(
    *,
    calendar_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    securities: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    reconciliation: Sequence[Mapping[str, Any]],
    interval_coverage: Mapping[tuple[str, date, date, str], float],
) -> tuple[list[dict[str, Any]], set[tuple[str, date, date, str]]]:
    accepted_by_period: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    source_period_by_id: dict[str, str] = {}
    for source in source_rows:
        if (
            source.get("source_status") in {"ACCEPTED", "PARSED"}
            and str(source.get("source_quality", "")).startswith("OFFICIAL")
            and source.get("review_period") not in {None, "ARCHIVE"}
        ):
            accepted_by_period[(str(source.get("index_id")), str(source.get("review_period")))].append(source)
            source_period_by_id[str(source.get("source_id"))] = str(source.get("review_period"))
    completed_intervals = {
        (str(row["index_id"]), row["effective_from"], row["effective_to"], str(row["source_id"]))
        for row in memberships
        if row.get("event_chain_complete") and row.get("resolution_status") == "RESOLVED_OFFICIAL_EXCHANGE_CODE"
    }
    conflict_sources = {
        str(row.get("source_id"))
        for row in reconciliation
        if row.get("severity") == "HIGH" and row.get("status") == "OPEN" and row.get("source_id")
    }
    eligible_intervals = {
        key
        for key in completed_intervals
        if key[3] not in conflict_sources and interval_coverage.get(key, 0.0) >= 0.95
    }
    rows: list[dict[str, Any]] = []
    for index_id, config in INDEX_CONFIG.items():
        expected = [row for row in calendar_rows if row["index_id"] == index_id]
        covered = [row for row in expected if accepted_by_period.get((index_id, row["review_period"]))]
        member_rows = [row for row in memberships if row["index_id"] == index_id]
        resolved_security_ids = {
            row["security_id"]
            for row in securities
            if row.get("market") == config["market"] and row.get("resolution_status", "").startswith("RESOLVED")
        }
        member_security_ids = {row["security_id"] for row in member_rows}
        market_candidates = [row for row in candidates if row.get("index_id") == index_id]
        candidate_population = {
            str(row.get("candidate_security_id") or row.get("source_record"))
            for row in market_candidates
            if row.get("candidate_security_id") or row.get("source_record")
        }
        resolved_candidate_population = {
            str(row.get("candidate_security_id"))
            for row in market_candidates
            if row.get("candidate_security_id") and str(row.get("decision") or "").startswith("RESOLVED")
        }
        identity_population = member_security_ids.union(candidate_population)
        resolved_population = member_security_ids.intersection(resolved_security_ids).union(resolved_candidate_population)
        entity_rate = len(resolved_population.intersection(identity_population)) / len(identity_population) if identity_population else 0.0
        market_conflicts = [row for row in reconciliation if row.get("index_id") == index_id and row.get("severity") == "HIGH" and row.get("status") == "OPEN"]
        interval_keys = [key for key in completed_intervals if key[0] == index_id]
        complete_periods = {
            source_period_by_id.get(key[3])
            for key in interval_keys
            if source_period_by_id.get(key[3]) is not None
        }
        weighted_coverage = (
            sum(interval_coverage.get(key, 0.0) for key in interval_keys) / len(interval_keys) if interval_keys else 0.0
        )
        market_eligible = (
            (len(covered) / len(expected) if expected else 0.0) >= 0.90
            and len(complete_periods) == len(expected)
            and not market_conflicts
            and entity_rate >= 0.95
            and weighted_coverage >= 0.95
        )
        ranges = sorted(
            f"{key[1].isoformat()}..{(key[2] - timedelta(days=1)).isoformat()}"
            for key in eligible_intervals
            if key[0] == index_id
        )
        blockers: list[str] = []
        if not expected:
            blockers.append("No declared review calendar.")
        if len(covered) < len(expected):
            blockers.append(f"Accepted review evidence {len(covered)}/{len(expected)}.")
        if len(complete_periods) < len(expected):
            blockers.append(f"Event-chain-complete review coverage {len(complete_periods)}/{len(expected)}.")
        if market_conflicts:
            blockers.append(f"{len(market_conflicts)} unresolved high-severity source/event-chain conflict(s).")
        if entity_rate < 0.95:
            blockers.append(f"Entity resolution rate {entity_rate:.3f} is below 0.95.")
        if weighted_coverage < 0.95:
            blockers.append(f"Raw OHLC active-member-day coverage {weighted_coverage:.3f} is below 0.95.")
        if not ranges:
            blockers.append("No market-date interval passes source, identity and raw-OHLC gates.")
        rows.append(
            {
                "market": config["market"],
                "index_id": index_id,
                "expected_review_periods": len(expected),
                "review_periods_with_accepted_evidence": len(covered),
                "event_chain_complete_intervals": len(interval_keys),
                "event_chain_complete_review_periods": len(complete_periods),
                "entity_resolution_rate": round(entity_rate, 6),
                "unresolved_high_severity_conflicts": len(market_conflicts),
                "historical_active_member_day_price_coverage": round(weighted_coverage, 6),
                "eligible": market_eligible,
                "eligible_date_ranges": " | ".join(ranges) if ranges else None,
                "blocker_list": " | ".join(blockers) if blockers else None,
            }
        )
    return rows, eligible_intervals


def load_current_comparator_tickers(repo_root: Path) -> dict[str, set[str]]:
    paths = {
        "VN30": repo_root / "config" / "vn30_constituents_2026_06.csv",
        "SET50": repo_root / "config" / "benchmark_universe" / "thailand_set50_2026_06.csv",
        "IDX30": repo_root / "config" / "benchmark_universe" / "indonesia_idx30_2026_06.csv",
        "FBM_KLCI": repo_root / "config" / "benchmark_universe" / "malaysia_klci_2026_06.csv",
        "PSEI": repo_root / "config" / "benchmark_universe" / "philippines_psei_2026_06.csv",
    }
    values: dict[str, set[str]] = {}
    for index_id, path in paths.items():
        if not path.is_file():
            values[index_id] = set()
            continue
        frame = pd.read_csv(path, dtype=str)
        column = next((name for name in ("symbol", "local_symbol", "ticker") if name in frame.columns), None)
        values[index_id] = set(frame[column].dropna().astype(str).str.strip()) if column else set()
    return values


def build_current_vs_pit_rows(
    snapshot_rows: Sequence[Mapping[str, Any]], repo_root: Path
) -> list[dict[str, Any]]:
    current = load_current_comparator_tickers(repo_root)
    groups: defaultdict[tuple[date, str], set[str]] = defaultdict(set)
    for row in snapshot_rows:
        groups[(row["date"], str(row["index_id"]))].add(str(row["ticker_on_date"]))
    previous: dict[str, set[str]] = {}
    result: list[dict[str, Any]] = []
    for (on_date, index_id), pit_tickers in sorted(groups.items()):
        now_current = current.get(index_id, set())
        intersect = pit_tickers.intersection(now_current)
        union = pit_tickers.union(now_current)
        turnover = len(pit_tickers.symmetric_difference(previous[index_id])) if index_id in previous else 0
        previous[index_id] = pit_tickers
        result.append(
            {
                "date": on_date,
                "market": INDEX_CONFIG[index_id]["market"],
                "index_id": index_id,
                "current_universe_size": len(now_current),
                "pit_universe_size": len(pit_tickers),
                "intersection": len(intersect),
                "union": len(union),
                "jaccard_overlap": len(intersect) / len(union) if union else None,
                "current_members_not_historically_active": len(now_current.difference(pit_tickers)),
                "historical_members_absent_from_june_2026": len(pit_tickers.difference(now_current)),
                "constituent_turnover": turnover,
                "market_date_coverage": "PIT_SOURCE_ENTITY_AND_RAW_OHLC_ELIGIBLE",
                "current_universe_semantics": CURRENT_SEMANTICS,
                "pit_universe_semantics": PIT_SEMANTICS,
            }
        )
    return result


SEARCH_LOG_COLUMNS = [
    "market",
    "index_id",
    "review_period",
    "query",
    "source_checked",
    "result",
    "accepted_source_id",
    "rejection_reason",
    "checked_at",
]


def normalise_search_log_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {column: row.get(column) for column in SEARCH_LOG_COLUMNS}


def build_calendar_and_search_log(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    acquisition_rows: Sequence[Mapping[str, Any]],
    external_search_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    calendar = [row for index_id in INDEX_CONFIG for row in expected_review_periods(index_id)]
    accepted_by_key: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for source in source_rows:
        if source.get("source_status") in {"ACCEPTED", "PARSED"} and source.get("review_period") not in {None, "ARCHIVE"}:
            accepted_by_key[(str(source.get("index_id")), str(source.get("review_period")))].append(source)
    logged_by_key: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in [*acquisition_rows, *external_search_rows]:
        if row.get("index_id") and row.get("review_period"):
            logged_by_key[(str(row["index_id"]), str(row["review_period"]))].append(row)
    for row in calendar:
        key = (str(row["index_id"]), str(row["review_period"]))
        sources = accepted_by_key.get(key, [])
        official_sources = [source for source in sources if str(source.get("source_quality", "")).startswith("OFFICIAL")]
        if official_sources:
            row["coverage_status"] = "ACCEPTED_OFFICIAL_OR_CHECKSUMMED_SOURCE"
            row["source"] = " | ".join(sorted({str(source.get("original_url")) for source in official_sources}))
            row["source_quality"] = " | ".join(sorted({str(source.get("source_quality")) for source in official_sources}))
        elif sources:
            row["coverage_status"] = "SECONDARY_CORROBORATION_ONLY"
            row["source"] = " | ".join(sorted({str(source.get("original_url")) for source in sources}))
            row["source_quality"] = " | ".join(sorted({str(source.get("source_quality")) for source in sources}))
            row["notes"] = f"{row['notes']} Secondary source(s) retained as corroboration only; no official membership evidence was accepted."
        elif logged_by_key.get(key):
            row["coverage_status"] = "SEARCHED_NO_ACCEPTED_SOURCE"
        else:
            row["coverage_status"] = "SEARCH_LOG_NOT_YET_SUPPLIED"
    search_rows: list[dict[str, Any]] = [normalise_search_log_row(row) for row in [*acquisition_rows, *external_search_rows]]
    for row in calendar:
        key = (str(row["index_id"]), str(row["review_period"]))
        accepted = accepted_by_key.get(key, [])
        existing = logged_by_key.get(key, [])
        if accepted:
            for source in accepted:
                search_rows.append(
                    {
                        "market": row["market"],
                        "index_id": row["index_id"],
                        "review_period": row["review_period"],
                        "query": f"Review-period evidence reconciliation for {row['review_period']}",
                        "source_checked": source.get("original_url"),
                        "result": "ACCEPTED_SOURCE_REGISTERED",
                        "accepted_source_id": source.get("source_id"),
                        "rejection_reason": None,
                        "checked_at": utc_now().isoformat(),
                    }
                )
        elif not existing:
            # This is intentionally not a claim that a source is unavailable.
            # It makes an unperformed per-period query visible and blocks source
            # acquisition completion until a researcher supplies its result.
            search_rows.append(
                {
                    "market": row["market"],
                    "index_id": row["index_id"],
                    "review_period": row["review_period"],
                    "query": f"Required official review search for {row['review_period']}",
                    "source_checked": INDEX_CONFIG[row["index_id"]]["official_search_url"],
                    "result": "SEARCH_RESULT_NOT_YET_DOCUMENTED",
                    "accepted_source_id": None,
                    "rejection_reason": "No period-specific source-search result was supplied; this is not a statement that no source exists.",
                    "checked_at": utc_now().isoformat(),
                }
            )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in search_rows:
        row = normalise_search_log_row(row)
        key = tuple(row.get(column) for column in ("index_id", "review_period", "source_checked", "result", "accepted_source_id", "rejection_reason"))
        unique[key] = row
    return calendar, sorted(unique.values(), key=lambda item: (str(item["index_id"]), str(item["review_period"]), str(item["source_checked"])))


def source_search_reconciliation(
    *, calendar_rows: Sequence[Mapping[str, Any]], search_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Turn documented source gaps/corrections into explicit reconciliation items."""

    rows: list[dict[str, Any]] = []
    for calendar in calendar_rows:
        status = str(calendar.get("coverage_status") or "")
        index_id = str(calendar.get("index_id"))
        if status not in {"SECONDARY_CORROBORATION_ONLY", "SEARCHED_NO_ACCEPTED_SOURCE"}:
            continue
        issue_type = (
            "OFFICIAL_SOURCE_GAP_SECONDARY_CORROBORATION_ONLY"
            if status == "SECONDARY_CORROBORATION_ONLY"
            else "SCHEDULED_REVIEW_WITHOUT_ACCEPTED_OFFICIAL_SOURCE"
        )
        rows.append(
            {
                "log_id": stable_id("RCL", issue_type, index_id, calendar.get("review_period")),
                "issue_type": issue_type,
                "severity": "HIGH",
                "index_id": index_id,
                "market": calendar.get("market"),
                "security_id": None,
                "source_id": None,
                "event_date": None,
                "status": "OPEN",
                "details": (
                    f"{calendar.get('review_period')}: {status}. "
                    f"Sources/search location: {calendar.get('source')}. {calendar.get('notes') or ''}"
                ),
            }
        )
    for search in search_rows:
        result = str(search.get("result") or "").lower()
        if "correction_unarchived" not in result and "correction" not in result:
            continue
        index_id = str(search.get("index_id") or "")
        if index_id not in INDEX_CONFIG:
            continue
        rows.append(
            {
                "log_id": stable_id("RCL", "unarchived-correction", index_id, search.get("review_period"), search.get("source_checked")),
                "issue_type": "LOCATED_OFFICIAL_CORRECTION_NOT_ARCHIVED",
                "severity": "HIGH",
                "index_id": index_id,
                "market": search.get("market") or INDEX_CONFIG[index_id]["market"],
                "security_id": None,
                "source_id": None,
                "event_date": None,
                "status": "OPEN",
                "details": (
                    f"{search.get('review_period')}: {search.get('source_checked')}. "
                    f"{search.get('rejection_reason') or 'Official correction was located but is not retained as raw evidence.'}"
                ),
            }
        )
    return rows


def write_qa_reports(
    *,
    root: Path,
    source_rows: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    ticker_history: Sequence[Mapping[str, Any]],
    snapshot_rows: Sequence[Mapping[str, Any]],
    eligibility_rows: Sequence[Mapping[str, Any]],
    reconciliation: Sequence[Mapping[str, Any]],
) -> dict[str, Path]:
    source_findings = validate_source_evidence(source_rows)
    event_findings = validate_events(events)
    interval_findings = validate_intervals(memberships, ticker_history)
    count_findings = validate_constituent_counts(memberships)
    snapshot_count_findings = validate_daily_snapshot_constituent_counts(snapshot_rows)
    source_path = root / "qa" / "source_evidence_v2_validation.json"
    events_path = root / "qa" / "membership_events_v2_validation.json"
    interval_path = root / "qa" / "interval_validation_report_v2.csv"
    counts_path = root / "qa" / "index_constituent_count_validation_v2.csv"
    snapshot_counts_path = root / "qa" / "daily_snapshot_constituent_count_validation_v2.csv"
    eligibility_path = root / "qa" / "pit_market_eligibility_v2.csv"
    atomic_write_json(source_path, {"findings": source_findings, "passed": not any(item["severity"] == "HIGH" for item in source_findings)})
    atomic_write_json(events_path, {"findings": event_findings, "passed": not any(item["severity"] == "HIGH" for item in event_findings)})
    write_csv(interval_path, interval_findings, ["severity", "rule", "source_id", "details"])
    write_csv(counts_path, count_findings, ["severity", "rule", "source_id", "details"])
    write_csv(snapshot_counts_path, snapshot_count_findings, ["severity", "rule", "source_id", "details"])
    write_csv(
        eligibility_path,
        eligibility_rows,
        [
            "market",
            "index_id",
            "expected_review_periods",
            "review_periods_with_accepted_evidence",
            "event_chain_complete_intervals",
            "event_chain_complete_review_periods",
            "entity_resolution_rate",
            "unresolved_high_severity_conflicts",
            "historical_active_member_day_price_coverage",
            "eligible",
            "eligible_date_ranges",
            "blocker_list",
        ],
    )
    semantic_path = root / "qa" / "current_pit_semantic_separation_v2.json"
    atomic_write_json(
        semantic_path,
        {
            "pit_snapshot_semantics": PIT_SEMANTICS,
            "current_comparator_semantics": CURRENT_SEMANTICS,
            "valid": PIT_SEMANTICS != CURRENT_SEMANTICS,
            "details": "The v2 descriptive comparison reports only compare labelled current and PIT universes; neither is pooled into the other.",
        },
    )
    return {
        "source": source_path,
        "events": events_path,
        "intervals": interval_path,
        "counts": counts_path,
        "snapshot_counts": snapshot_counts_path,
        "eligibility": eligibility_path,
        "semantics": semantic_path,
    }


def write_file_manifest(root: Path, path: Path, purpose: str) -> Path:
    relative = relative_path(path, root)
    manifest = {
        "path": relative,
        "format": path.suffix.lstrip("."),
        "sha256": sha256_file(path),
        "byte_count": path.stat().st_size,
        "purpose": purpose,
        "pipeline_version": PIPELINE_VERSION,
    }
    target = root / "manifests" / f"{relative.replace('/', '__').replace('.', '_')}.manifest.json"
    atomic_write_json(target, manifest)
    return target


def write_raw_source_manifest(root: Path, source_rows: Sequence[Mapping[str, Any]]) -> Path:
    records: list[dict[str, Any]] = []
    for source in sorted(source_rows, key=lambda item: str(item.get("source_id"))):
        raw_path_value = source.get("raw_path")
        raw_path = root / str(raw_path_value) if raw_path_value else None
        checksum = source.get("file_checksum")
        records.append(
            {
                "source_id": source.get("source_id"),
                "index_id": source.get("index_id"),
                "market": source.get("market"),
                "review_period": source.get("review_period"),
                "original_url": source.get("original_url"),
                "raw_path": raw_path_value,
                "expected_sha256": checksum,
                "observed_sha256": sha256_file(raw_path) if raw_path and raw_path.is_file() else None,
                "checksum_valid": bool(raw_path and raw_path.is_file() and checksum and sha256_file(raw_path) == checksum),
                "retrieval_timestamp": source.get("retrieval_timestamp"),
                "publication_date": source.get("publication_date"),
                "announcement_date": source.get("announcement_date"),
                "stated_effective_date": source.get("stated_effective_date"),
                "parser_version": source.get("parser_version"),
                "source_quality": source.get("source_quality"),
            }
        )
    path = root / "manifests" / "raw_source_evidence_v2.manifest.json"
    atomic_write_json(
        path,
        {
            "pipeline_version": PIPELINE_VERSION,
            "record_count": len(records),
            "records": records,
        },
    )
    return path


def summary_payload(
    *,
    calendar: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    memberships: Sequence[Mapping[str, Any]],
    securities: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    price_queue: Sequence[Mapping[str, Any]],
    snapshots: Sequence[Mapping[str, Any]],
    search_rows: Sequence[Mapping[str, Any]],
    tests: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    coverage: dict[str, dict[str, Any]] = {}
    for index_id, config in INDEX_CONFIG.items():
        expected = [row for row in calendar if row["index_id"] == index_id]
        covered = [row for row in expected if row["coverage_status"] == "ACCEPTED_OFFICIAL_OR_CHECKSUMMED_SOURCE"]
        unresolved = [row["review_period"] for row in expected if row not in covered]
        accepted = [
            row
            for row in sources
            if row.get("index_id") == index_id and row.get("source_status") in {"ACCEPTED", "PARSED"}
        ]
        searched_periods = {
            str(row.get("review_period"))
            for row in search_rows
            if row.get("index_id") == index_id
            and str(row.get("review_period")) in {str(item["review_period"]) for item in expected}
            and row.get("result") != "SEARCH_RESULT_NOT_YET_DOCUMENTED"
        }
        coverage[index_id] = {
            "market": config["market"],
            "expected_review_periods": len(expected),
            "searched_review_periods": len(searched_periods),
            "covered_review_periods": len(covered),
            "accepted_official_sources": sum(str(item.get("source_quality", "")).startswith("OFFICIAL") for item in accepted),
            "accepted_secondary_sources": sum(not str(item.get("source_quality", "")).startswith("OFFICIAL") for item in accepted),
            "unresolved_periods": unresolved,
            "source_acquisition_complete": len(searched_periods) == len(expected) and bool(expected),
        }
    market_rows = {str(row["index_id"]): row for row in eligibility}
    historical_members = len({row.get("security_id") for row in securities})
    statuses_by_security: defaultdict[str, set[str]] = defaultdict(set)
    for row in price_queue:
        statuses_by_security[str(row.get("security_id"))].add(str(row.get("download_status")))
    locally_covered = sum(statuses == {"LOCAL_RAW_OHLC_COVERED"} for statuses in statuses_by_security.values())
    queued = sum("QUEUED" in statuses for statuses in statuses_by_security.values())
    not_price_queueable = historical_members - len(statuses_by_security)
    has_evidence = bool(sources) and any(row.get("index_id") in INDEX_CONFIG and row.get("review_period") != "ARCHIVE" for row in sources)
    five_market_complete = all(bool(row.get("eligible")) for row in eligibility) and len(eligibility) == len(INDEX_CONFIG)
    return {
        "status": "COMPLETE" if five_market_complete else "PARTIAL" if has_evidence else "BLOCKED",
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": utc_now().isoformat(),
        "target_period": {"start": TARGET_START.isoformat(), "end": TARGET_END.isoformat()},
        "source_coverage": coverage,
        "membership": {
            "extracted_events": len(events),
            "validated_intervals": len(memberships),
            "resolved_securities": sum(row.get("resolution_status", "").startswith("RESOLVED") for row in securities),
            "unresolved_entities": sum(bool(row.get("manual_review_required")) for row in candidates),
            "daily_snapshot_rows": len(snapshots),
        },
        "eligibility": {
            "eligible_markets": [index_id for index_id, row in market_rows.items() if row.get("eligible")],
            "eligible_date_ranges": {index_id: row.get("eligible_date_ranges") for index_id, row in market_rows.items() if row.get("eligible_date_ranges")},
            "ineligible_markets": {index_id: row.get("blocker_list") for index_id, row in market_rows.items() if not row.get("eligible")},
        },
        "price_coverage": {
            "historical_members_identified": historical_members,
            "locally_covered": locally_covered,
            "queued_for_download": queued,
            "unavailable": 0,
            "unavailable_note": "No external provider download attempt was made; queued does not mean unavailable.",
            "not_price_queueable_due_to_missing_verified_date": not_price_queueable,
        },
        "tests": dict(tests or {"passed": None, "failed": None, "status": "NOT_RUN_BY_PIPELINE"}),
        "completion_note": "A five-market PIT dataset is not declared complete unless every required market satisfies its fail-closed gates. Source-acquisition completeness is reported separately from PIT reconstruction eligibility.",
    }


def write_status_report(root: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    report_dir = root.parent.parent / "reports" / "data_execution"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "PIT_SOURCE_ACQUISITION_STATUS.json"
    md_path = report_dir / "PIT_SOURCE_ACQUISITION_STATUS.md"
    atomic_write_json(json_path, dict(payload))
    lines = [
        "# PIT Source Acquisition Status",
        "",
        f"**STATUS: {payload['status']}**",
        "",
        f"Target period: {payload['target_period']['start']} through {payload['target_period']['end']}",
        "",
        "## Source coverage",
        "",
        "| Index | Expected | Searched | Covered | Official sources | Secondary sources | Source acquisition | Unresolved periods |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for index_id, item in payload["source_coverage"].items():
        unresolved = ", ".join(item["unresolved_periods"]) or "None"
        lines.append(
            f"| {index_id} | {item['expected_review_periods']} | {item['searched_review_periods']} | {item['covered_review_periods']} | "
            f"{item['accepted_official_sources']} | {item['accepted_secondary_sources']} | "
            f"{'COMPLETE' if item['source_acquisition_complete'] else 'INCOMPLETE'} | {unresolved} |"
        )
    lines.extend(
        [
            "",
            "`Source acquisition COMPLETE` means every scheduled period has a documented search result and retained accepted raw evidence where obtained. It does not mean every period has accepted official membership evidence or that the market is PIT-eligible.",
        ]
    )
    membership = payload["membership"]
    lines.extend(
        [
            "",
            "## Membership",
            "",
            f"- Extracted events: {membership['extracted_events']}",
            f"- Validated intervals: {membership['validated_intervals']}",
            f"- Resolved securities: {membership['resolved_securities']}",
            f"- Unresolved entities: {membership['unresolved_entities']}",
            f"- Eligible daily PIT snapshot rows: {membership['daily_snapshot_rows']}",
            "",
            "## Eligibility",
            "",
            f"- Eligible markets: {', '.join(payload['eligibility']['eligible_markets']) or 'None'}",
        ]
    )
    for index_id, range_text in payload["eligibility"]["eligible_date_ranges"].items():
        lines.append(f"- {index_id} source/identity/raw-OHLC-qualified dates: {range_text}")
    for index_id, reason in payload["eligibility"]["ineligible_markets"].items():
        lines.append(f"- {index_id} blockers: {reason}")
    prices = payload["price_coverage"]
    tests = payload["tests"]
    lines.extend(
        [
            "",
            "## Price coverage",
            "",
            f"- Historical members identified: {prices['historical_members_identified']}",
            f"- Locally covered: {prices['locally_covered']}",
            f"- Queued for download: {prices['queued_for_download']}",
            f"- Unavailable: {prices['unavailable']}",
            f"- Note: {prices.get('unavailable_note', '')}",
            f"- Not price-queueable due to no verified observation date: {prices.get('not_price_queueable_due_to_missing_verified_date', 0)}",
            "",
            "## Tests",
            "",
            f"- Passed: {tests.get('passed')}",
            f"- Failed: {tests.get('failed')}",
            f"- Status: {tests.get('status')}",
            "",
            payload["completion_note"],
            "",
        ]
    )
    atomic_write_text(md_path, "\n".join(lines))
    return md_path, json_path


EVENT_INVENTORY_COLUMNS = [
    "index_id",
    "market",
    "security_name_as_published",
    "ticker_as_published",
    "event_type",
    "announcement_date",
    "effective_date",
    "review_period",
    "source_id",
    "page_or_section",
    "extraction_method",
    "confidence",
    "manual_review_required",
    "source_record",
    "notes",
]


ANCHOR_INVENTORY_COLUMNS = [
    "index_id",
    "market",
    "ticker_as_published",
    "security_name_as_published",
    "snapshot_date",
    "effective_from",
    "effective_to",
    "effective_to_boundary",
    "announcement_date",
    "review_period",
    "source_id",
    "page_or_section",
    "extraction_method",
    "confidence",
    "manual_review_required",
    "source_record",
    "reconstruction_method",
    "event_chain_complete",
    "notes",
]


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_confidence(value: Any) -> float | None:
    """Normalize transparent draft labels without treating them as evidence."""

    if value is None or str(value).strip() == "":
        return None
    labels = {"high": 0.98, "medium": 0.75, "low": 0.50}
    label = str(value).strip().lower()
    if label in labels:
        return labels[label]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_external_event_drafts(root: Path, source_ids: set[str]) -> list[dict[str, Any]]:
    """Import manually reviewed extraction drafts only when they cite accepted sources.

    This lets a reviewer retain source-specific additions/removals from markets
    whose layouts need a bespoke parser.  The importer does not turn a draft
    into an interval; it only records the evidence-backed event separately.
    """

    rows: list[dict[str, Any]] = []
    replacement_groups: defaultdict[tuple[str, str, str, date | None, str], list[dict[str, Any]]] = defaultdict(list)
    for path in sorted((root / "interim").glob("*_membership_events_v2_draft.csv")):
        for raw in pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"):
            index_id = raw.get("index_id")
            source_id = raw.get("source_id")
            if index_id not in INDEX_CONFIG or source_id not in source_ids:
                continue
            record = {column: raw.get(column) for column in EVENT_INVENTORY_COLUMNS}
            record["market"] = record.get("market") or INDEX_CONFIG[index_id]["market"]
            record["announcement_date"] = parse_date(record.get("announcement_date"))
            record["effective_date"] = parse_date(record.get("effective_date"))
            record["confidence"] = _parse_confidence(record.get("confidence"))
            record["manual_review_required"] = _parse_bool(record.get("manual_review_required"))
            record["event_type"] = str(record.get("event_type") or "").strip().upper() or None
            record["ticker_as_published"] = str(record.get("ticker_as_published") or "").strip() or None
            record["source_record"] = record.get("source_record") or stable_id(
                "EVT", source_id, record.get("event_type"), record.get("ticker_as_published"), record.get("effective_date")
            )
            record["extraction_method"] = record.get("extraction_method") or "manual_reviewed_external_event_draft"
            replacement_group_id = str(raw.get("replacement_group_id") or "").strip()
            if replacement_group_id:
                record["notes"] = " | ".join(filter(None, [str(record.get("notes") or ""), f"replacement_group_id={replacement_group_id}"]))
                replacement_groups[
                    (index_id, source_id, replacement_group_id, record.get("effective_date"), str(record.get("review_period") or ""))
                ].append(record)
            rows.append(record)
    for (index_id, source_id, group_id, effective_date, review_period), members in sorted(replacement_groups.items()):
        additions = sorted(str(row.get("ticker_as_published")) for row in members if row.get("event_type") == "ADDITION")
        removals = sorted(str(row.get("ticker_as_published")) for row in members if row.get("event_type") == "REMOVAL")
        if not additions or not removals:
            continue
        reference = members[0]
        rows.append(
            {
                "index_id": index_id,
                "market": reference.get("market"),
                "security_name_as_published": f"IN: {', '.join(additions)}; OUT: {', '.join(removals)}",
                "ticker_as_published": None,
                "event_type": "REPLACEMENT",
                "announcement_date": reference.get("announcement_date"),
                "effective_date": effective_date,
                "review_period": review_period,
                "source_id": source_id,
                "page_or_section": reference.get("page_or_section"),
                "extraction_method": "replacement_grouped_from_manual_reviewed_explicit_addition_removal_records",
                "confidence": min(float(row.get("confidence") or 0.0) for row in members),
                "manual_review_required": any(bool(row.get("manual_review_required")) for row in members),
                "source_record": stable_id("EVT", source_id, "replacement", group_id, effective_date),
                "notes": f"Explicit replacement relation retained from replacement_group_id={group_id}; constituent addition/removal records remain separately extracted.",
            }
        )
    return rows


def load_external_anchor_drafts(
    root: Path, source_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load explicit direct anchors and retain boundary-ambiguous snapshots.

    A direct snapshot can prove that a security was listed at its explicit
    stated date, while still failing to prove a daily SCD interval.  Rather
    than discarding that primary evidence or silently converting an endpoint,
    this loader emits a manual-review snapshot event and a single reconciliation
    item per affected source.  Only an explicitly documented exclusive (or
    explicitly inclusive and mechanically converted) endpoint plus a complete
    event chain can create a membership interval.
    """

    sources = {str(row.get("source_id")): row for row in source_rows}
    grouped: defaultdict[tuple[str, str, date, date, str], list[dict[str, Any]]] = defaultdict(list)
    snapshot_events: list[dict[str, Any]] = []
    unresolved_snapshot_groups: defaultdict[tuple[str, str, date, str, str], list[dict[str, Any]]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []
    for path in sorted((root / "interim").glob("*_membership_anchors_v2_draft.csv")):
        raw_rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
        for ordinal, raw in enumerate(raw_rows, start=1):
            record = {column: raw.get(column) for column in ANCHOR_INVENTORY_COLUMNS}
            index_id = str(record.get("index_id") or "")
            source_id = str(record.get("source_id") or raw.get("anchor_source_id") or "")
            source = sources.get(source_id)
            start = parse_date(record.get("effective_from")) or parse_date(record.get("snapshot_date"))
            end = parse_date(record.get("effective_to"))
            confidence = _parse_confidence(record.get("confidence"))
            manual = _parse_bool(record.get("manual_review_required"))
            chain_complete = _parse_bool(record.get("event_chain_complete"))
            boundary = str(
                record.get("effective_to_boundary")
                or raw.get("effective_to_convention")
                or raw.get("interval_end_convention")
                or ""
            ).strip().upper()
            record["source_id"] = source_id
            record["page_or_section"] = record.get("page_or_section") or raw.get("source_page_or_section") or raw.get("source_page")
            record["reconstruction_method"] = (
                record.get("reconstruction_method")
                or raw.get("reconstruction_direction")
                or raw.get("anchor_type")
                or "direct_snapshot"
            )
            reason: str | None = None
            if index_id not in INDEX_CONFIG:
                reason = "Unknown or missing index_id"
            elif source is None:
                reason = "Anchor cites no accepted checksummed source_id"
            elif not str(source.get("source_quality", "")).startswith("OFFICIAL"):
                reason = "A secondary source may corroborate but cannot supply a direct PIT anchor"
            elif not str(record.get("ticker_as_published") or "").strip():
                reason = "Exact ticker_as_published is required"
            elif start is None:
                reason = "An explicit snapshot/effective date is required"
            elif confidence is None or confidence < 0.95:
                reason = "Anchor confidence must be at least 0.95"
            if reason:
                conflicts.append(
                    {
                        "log_id": stable_id("RCL", "external-anchor", path.name, ordinal, reason),
                        "issue_type": "EXTERNAL_ANCHOR_DRAFT_REJECTED",
                        "severity": "HIGH",
                        "index_id": index_id or None,
                        "market": record.get("market") or (INDEX_CONFIG[index_id]["market"] if index_id in INDEX_CONFIG else None),
                        "security_id": None,
                        "source_id": source_id or None,
                        "event_date": start,
                        "status": "OPEN",
                        "details": f"{path.name} row {ordinal}: {reason}.",
                    }
                )
                continue
            record["market"] = record.get("market") or INDEX_CONFIG[index_id]["market"]
            record["effective_from"] = start
            record["announcement_date"] = parse_date(record.get("announcement_date"))
            record["confidence"] = confidence
            record["manual_review_required"] = manual
            record["event_chain_complete"] = chain_complete
            record["source_record"] = record.get("source_record") or stable_id(
                "ANCHOR", source_id, record.get("ticker_as_published"), start, end
            )
            record["extraction_method"] = record.get("extraction_method") or "manual_reviewed_official_full_list"
            review_period = str(record.get("review_period") or source.get("review_period") or review_period_for(index_id, start))

            # The draft can use a provider's explicit inclusive end date, but
            # it may only be converted to the pipeline's exclusive endpoint
            # when the source/reviewer explicitly labels that convention.
            interval_end: date | None = None
            interval_notes: str | None = None
            if boundary == "EXCLUSIVE" and end is not None and end > start:
                interval_end = end
            elif boundary == "INCLUSIVE" and end is not None and end >= start:
                interval_end = end + timedelta(days=1)
                interval_notes = "Explicit source-inclusive endpoint was converted mechanically to the v2 exclusive SCD boundary."

            if manual is False and chain_complete is True and interval_end is not None:
                record["effective_to"] = interval_end
                if interval_notes:
                    record["notes"] = " | ".join(filter(None, [str(record.get("notes") or ""), interval_notes]))
                grouped[(index_id, source_id, start, interval_end, review_period)].append(record)
                continue

            if chain_complete is not True:
                boundary_reason = "Draft explicitly marks event_chain_complete=false"
            elif manual is not False:
                boundary_reason = "Draft requires manual review"
            elif end is None:
                boundary_reason = "No stated interval end was supplied"
            elif boundary not in {"EXCLUSIVE", "INCLUSIVE"}:
                boundary_reason = "Stated interval endpoint lacks an explicit inclusive/exclusive convention"
            else:
                boundary_reason = "No valid positive interval boundary was supplied"
            ticker = str(record["ticker_as_published"]).strip()
            snapshot_events.append(
                {
                    "index_id": index_id,
                    "market": record["market"],
                    "security_name_as_published": record.get("security_name_as_published"),
                    "ticker_as_published": ticker,
                    "event_type": "CONSTITUENT_SNAPSHOT",
                    "announcement_date": record.get("announcement_date") or source.get("announcement_date"),
                    "effective_date": start,
                    "review_period": review_period,
                    "source_id": source_id,
                    "page_or_section": record.get("page_or_section"),
                    "extraction_method": record["extraction_method"],
                    "confidence": confidence,
                    "manual_review_required": True,
                    "source_record": stable_id("EVT", source_id, "snapshot-source-evidence", ticker, start),
                    "notes": " | ".join(
                        filter(
                            None,
                            [
                                str(record.get("notes") or ""),
                                f"DIRECT_SNAPSHOT_SOURCE_EVIDENCE_ONLY: {boundary_reason}. No SCD interval or daily PIT snapshot was generated.",
                            ],
                        )
                    ),
                }
            )
            unresolved_snapshot_groups[(index_id, source_id, start, review_period, boundary_reason)].append(record)

    anchors: list[dict[str, Any]] = []
    for (index_id, source_id, start, end, review_period), rows in sorted(grouped.items()):
        source = sources[source_id]
        tickers = [str(row["ticker_as_published"]).strip() for row in rows]
        names = {str(row["ticker_as_published"]).strip(): row.get("security_name_as_published") for row in rows}
        confidences = [float(row["confidence"]) for row in rows]
        anchors.append(
            {
                "index_id": index_id,
                "market": INDEX_CONFIG[index_id]["market"],
                "source_id": source_id,
                "announcement_date": rows[0].get("announcement_date") or source.get("announcement_date"),
                "effective_from": start,
                "effective_to": end,
                "tickers": tickers,
                "security_names": names,
                "review_period": review_period,
                "reconstruction_method": rows[0]["reconstruction_method"],
                "confidence": min(confidences),
                "notes": " | ".join(sorted({str(row.get("notes")) for row in rows if row.get("notes")})) or None,
            }
        )
    for (index_id, source_id, start, review_period, reason), rows in sorted(unresolved_snapshot_groups.items()):
        source = sources[source_id]
        conflicts.append(
            {
                "log_id": stable_id("RCL", source_id, "snapshot-boundary", start, reason),
                "issue_type": "UNRESOLVED_INTERVAL_BOUNDARY_SEMANTICS",
                "severity": "HIGH",
                "index_id": index_id,
                "market": INDEX_CONFIG[index_id]["market"],
                "security_id": None,
                "source_id": source_id,
                "event_date": start,
                "status": "OPEN",
                "details": (
                    f"{len(rows)} exact published constituent record(s) were retained as direct snapshot evidence for {review_period}, "
                    f"but no daily [effective_from, effective_to) interval was emitted: {reason}. "
                    f"Source: {source.get('original_url')}"
                ),
            }
        )
    return anchors, snapshot_events, conflicts


def load_external_reconciliation_drafts(root: Path) -> list[dict[str, Any]]:
    """Retain reviewer-documented source conflicts in the v2 reconciliation log."""

    rows: list[dict[str, Any]] = []
    reconciliation_root = root / "reconciliation"
    for path in sorted(reconciliation_root.glob("*_v2_draft.csv")):
        for ordinal, raw in enumerate(pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records"), start=1):
            index_id = str(raw.get("index_id") or "")
            if index_id not in INDEX_CONFIG:
                continue
            source_id = raw.get("source_id") or raw.get("anchor_source_id") or raw.get("accepted_source_id")
            if raw.get("issue_type"):
                issue_type = str(raw.get("issue_type"))
                details = str(raw.get("details") or raw.get("notes") or "")
                severity = str(raw.get("severity") or "HIGH").upper()
                event_date = parse_date(raw.get("event_date"))
            elif raw.get("conflict_id"):
                issue_type = "DOCUMENTED_SOURCE_CONFLICT"
                event_date = parse_date(raw.get("accepted_source_value")) if str(raw.get("field_name")) in {"effective_date", "announcement_date"} else None
                severity = "HIGH"
                details = " | ".join(
                    filter(
                        None,
                        [
                            f"Conflict ID: {raw.get('conflict_id')}",
                            f"Review period: {raw.get('review_period')}" if raw.get("review_period") else None,
                            f"Field: {raw.get('field_name')}" if raw.get("field_name") else None,
                            f"Accepted source/value: {raw.get('accepted_source_id')} / {raw.get('accepted_source_value')}" if raw.get("accepted_source_id") else None,
                            f"Corrective source/value: {raw.get('corrective_source_reference')} / {raw.get('corrective_source_value')}" if raw.get("corrective_source_reference") else None,
                            f"Corrective raw retained: {raw.get('corrective_raw_retained')}" if raw.get("corrective_raw_retained") else None,
                            f"Reconciliation status: {raw.get('reconciliation_status')}" if raw.get("reconciliation_status") else None,
                            raw.get("resolution"),
                            raw.get("impact"),
                            raw.get("notes"),
                        ],
                    )
                )
            else:
                issue_type = "DOCUMENTED_SOURCE_BOUNDARY_CONFLICT"
                stated_range = str(raw.get("anchor_source_stated_range") or "")
                event_date = parse_date(re.search(r"\d{4}-\d{2}-\d{2}", stated_range).group(0)) if re.search(r"\d{4}-\d{2}-\d{2}", stated_range) else None
                severity = "HIGH"
                details = " | ".join(
                    filter(
                        None,
                        [
                            f"Review period: {raw.get('review_period')}" if raw.get("review_period") else None,
                            f"Anchor stated range: {stated_range}" if stated_range else None,
                            f"Next source: {raw.get('next_source_id')} {raw.get('next_source_stated_effective_or_range')}" if raw.get("next_source_id") else None,
                            raw.get("boundary_assessment"),
                            raw.get("notes"),
                            f"Draft evidence paths: {raw.get('evidence_paths')}" if raw.get("evidence_paths") else None,
                        ],
                    )
                )
            rows.append(
                {
                    "log_id": str(raw.get("log_id") or stable_id("RCL", path.name, raw.get("conflict_id") or ordinal, issue_type, source_id, details)),
                    "issue_type": issue_type,
                    "severity": severity,
                    "index_id": index_id,
                    "market": raw.get("market") or INDEX_CONFIG[index_id]["market"],
                    "security_id": raw.get("security_id") or None,
                    "source_id": source_id or None,
                    "event_date": event_date,
                    "status": str(raw.get("status") or ("OPEN" if str(raw.get("reconciliation_status") or "").lower().startswith("unresolved") else "OPEN")).upper(),
                    "details": details,
                }
            )
    return rows


def run_pipeline_v2(
    *,
    repo_root: Path | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Build v2 outputs from retained evidence only; no network access occurs here."""

    repo_root = (repo_root or repo_root_from_script()).resolve()
    root = (output_root or repo_root / "research_data" / "point_in_time").resolve()
    ensure_output_layout(root)
    write_schema_files(root)

    set_sources, acquisition_rows = load_set_catalog(root)
    external_sources, external_search_rows = load_external_inventories(root)
    external_search_rows.extend(load_external_search_logs(root))
    source_rows, source_conflicts = deduplicate_source_rows([*set_sources, *external_sources])
    processed_sources, set_events, anchors, process_conflicts = process_set50_sources(root=root, source_rows=source_rows)
    source_rows, post_process_conflicts = deduplicate_source_rows(processed_sources)
    external_events = load_external_event_drafts(root, {str(row["source_id"]) for row in source_rows})
    external_anchors, snapshot_anchor_events, anchor_conflicts = load_external_anchor_drafts(root, source_rows)
    external_reconciliation = load_external_reconciliation_drafts(root)
    anchors = [*anchors, *external_anchors]
    events_by_record: dict[str, dict[str, Any]] = {}
    for event in [*set_events, *external_events, *snapshot_anchor_events]:
        events_by_record[str(event["source_record"])] = dict(event)
    events = sorted(events_by_record.values(), key=lambda item: str(item["source_record"]))

    securities, ticker_history, memberships, candidates, identity_conflicts = create_security_and_intervals(anchors)
    securities, ticker_history, candidates = enrich_identity_from_observed_events(
        securities=securities,
        ticker_history=ticker_history,
        candidates=candidates,
        events=events,
        source_rows=source_rows,
    )
    calendar_rows, search_rows = build_calendar_and_search_log(
        source_rows=source_rows,
        acquisition_rows=acquisition_rows,
        external_search_rows=external_search_rows,
    )
    search_conflicts = source_search_reconciliation(calendar_rows=calendar_rows, search_rows=search_rows)
    validation_findings = [
        *validate_source_evidence(source_rows),
        *validate_events(events),
        *validate_intervals(memberships, ticker_history),
        *validate_constituent_counts(memberships),
    ]
    reconciliation = write_reconciliation_from_findings(
        [
            *source_conflicts,
            *post_process_conflicts,
            *process_conflicts,
            *anchor_conflicts,
            *external_reconciliation,
            *search_conflicts,
            *identity_conflicts,
        ],
        validation_findings,
    )
    price_queue, interval_coverage = local_price_coverage(
        repo_root=repo_root, memberships=memberships, ticker_history=ticker_history
    )
    price_queue = append_identity_observation_price_queue(
        repo_root=repo_root,
        queue_rows=price_queue,
        memberships=memberships,
        securities=securities,
        ticker_history=ticker_history,
    )
    eligibility_rows, eligible_intervals = build_eligibility_rows(
        calendar_rows=calendar_rows,
        source_rows=source_rows,
        memberships=memberships,
        securities=securities,
        candidates=candidates,
        reconciliation=reconciliation,
        interval_coverage=interval_coverage,
    )
    snapshot_rows = build_snapshots(memberships, ticker_history, eligible_intervals)
    comparison_rows = build_current_vs_pit_rows(snapshot_rows, repo_root)

    tables = {
        "source_evidence_v2": rows_to_table("source_evidence_v2", source_rows),
        "membership_events_extracted_v2": rows_to_table("membership_events_extracted_v2", events),
        "security_master_v2": rows_to_table("security_master_v2", securities),
        "ticker_history_v2": rows_to_table("ticker_history_v2", ticker_history),
        "entity_resolution_candidates_v2": rows_to_table("entity_resolution_candidates_v2", candidates),
        "reconciliation_log_v2": rows_to_table("reconciliation_log_v2", reconciliation),
        "index_membership_v2": rows_to_table("index_membership_v2", memberships),
        "pit_universe_daily_v2": rows_to_table("pit_universe_daily_v2", snapshot_rows),
        "current_vs_pit_universe_descriptive_v2": rows_to_table("current_vs_pit_universe_descriptive_v2", comparison_rows),
    }
    for name, table in tables.items():
        write_table(root, name, table)
        write_table_manifest(root, name, table)

    calendar_path = root / "reports" / "index_review_calendar.csv"
    write_csv(
        calendar_path,
        calendar_rows,
        [
            "index_id",
            "market",
            "review_period",
            "expected_announcement_window",
            "expected_effective_window",
            "review_type",
            "source",
            "source_quality",
            "coverage_status",
            "notes",
        ],
    )
    search_path = root / "reports" / "source_search_log.csv"
    write_csv(search_path, search_rows, SEARCH_LOG_COLUMNS)
    price_path = root / "reports" / "price_download_queue_by_provider.csv"
    write_csv(
        price_path,
        price_queue,
        [
            "market",
            "security_id",
            "historical_ticker",
            "exchange_suffix",
            "membership_start",
            "membership_end",
            "required_price_start",
            "required_price_end",
            "preferred_provider",
            "fallback_provider",
            "local_data_exists",
            "download_status",
            "failure_reason",
        ],
    )
    overlap_path = root / "reports" / "current_vs_pit_overlap_by_date_v2.csv"
    write_csv(
        overlap_path,
        comparison_rows,
        [
            "date",
            "market",
            "index_id",
            "current_universe_size",
            "pit_universe_size",
            "intersection",
            "union",
            "jaccard_overlap",
            "current_members_not_historically_active",
            "historical_members_absent_from_june_2026",
            "constituent_turnover",
            "market_date_coverage",
            "current_universe_semantics",
            "pit_universe_semantics",
        ],
    )
    qa_paths = write_qa_reports(
        root=root,
        source_rows=source_rows,
        events=events,
        memberships=memberships,
        ticker_history=ticker_history,
        snapshot_rows=snapshot_rows,
        eligibility_rows=eligibility_rows,
        reconciliation=reconciliation,
    )
    raw_manifest = write_raw_source_manifest(root, source_rows)
    for file_path, purpose in (
        (calendar_path, "Expected PIT review calendar and coverage status"),
        (search_path, "Historical source search log"),
        (price_path, "Provider-ready historical raw-OHLC acquisition queue"),
        (overlap_path, "Current versus PIT descriptive overlap by date"),
        (raw_manifest, "Raw source evidence checksum manifest"),
        *[(path, f"v2 QA: {name}") for name, path in qa_paths.items()],
    ):
        write_file_manifest(root, file_path, purpose)

    run_manifest = root / "manifests" / "point_in_time_v2_run.manifest.json"
    atomic_write_json(
        run_manifest,
        {
            "pipeline": "point_in_time_v2",
            "pipeline_version": PIPELINE_VERSION,
            "target_period": {"start": TARGET_START.isoformat(), "end": TARGET_END.isoformat()},
            "network_behavior": "Build reads retained raw evidence only; acquisition is a separate explicit command.",
            "tables": [TABLE_PATHS[name] for name in tables],
            "reports": [relative_path(path, root) for path in (calendar_path, search_path, price_path, overlap_path)],
            "raw_source_manifest": relative_path(raw_manifest, root),
            "semantic_separation": "PIT and current-2026 comparison universes remain distinct labelled tables.",
        },
    )
    write_file_manifest(root, run_manifest, "v2 point-in-time run manifest")

    payload = summary_payload(
        calendar=calendar_rows,
        sources=source_rows,
        events=events,
        memberships=memberships,
        securities=securities,
        candidates=candidates,
        eligibility=eligibility_rows,
        price_queue=price_queue,
        snapshots=snapshot_rows,
        search_rows=search_rows,
    )
    md_path, json_path = write_status_report(root, payload)
    return {
        "status": payload["status"],
        "output_root": str(root),
        "source_evidence_rows": tables["source_evidence_v2"].num_rows,
        "membership_events": tables["membership_events_extracted_v2"].num_rows,
        "membership_intervals": tables["index_membership_v2"].num_rows,
        "snapshot_rows": tables["pit_universe_daily_v2"].num_rows,
        "eligible_markets": payload["eligibility"]["eligible_markets"],
        "status_report": str(md_path),
        "status_json": str(json_path),
    }


def finalise_test_status(
    *, root: Path, passed: int, failed: int, command: str
) -> dict[str, Any]:
    """Refresh the existing v2 status report after the requested test run."""

    json_path = root.parent.parent / "reports" / "data_execution" / "PIT_SOURCE_ACQUISITION_STATUS.json"
    if not json_path.exists():
        raise FileNotFoundError("Build v2 outputs before finalising test status")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["tests"] = {
        "passed": passed,
        "failed": failed,
        "status": "PASS" if failed == 0 else "FAIL",
        "command": command,
        "recorded_at": utc_now().isoformat(),
    }
    write_status_report(root, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire-set50", help="Archive official SET50 period lists from the official archive page.")
    acquire.add_argument("--repo-root")
    acquire.add_argument("--output-root")
    acquire.add_argument("--timeout-seconds", type=int, default=45)
    acquire.add_argument("--retries", type=int, default=3)
    build = subparsers.add_parser("build", help="Build v2 outputs from retained evidence only.")
    build.add_argument("--repo-root")
    build.add_argument("--output-root")
    finalize = subparsers.add_parser("finalize-tests", help="Record full-suite test totals in the v2 status report.")
    finalize.add_argument("--repo-root")
    finalize.add_argument("--output-root")
    finalize.add_argument("--passed", required=True, type=int)
    finalize.add_argument("--failed", required=True, type=int)
    finalize.add_argument("--command", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    root = Path(args.output_root).resolve() if args.output_root else None
    if args.command == "acquire-set50":
        output = (root or (repo_root or repo_root_from_script()) / "research_data" / "point_in_time").resolve()
        result = archive_set50_sources(root=output, timeout_seconds=args.timeout_seconds, retries=args.retries)
    elif args.command == "build":
        result = run_pipeline_v2(repo_root=repo_root, output_root=root)
    else:
        output = (root or (repo_root or repo_root_from_script()) / "research_data" / "point_in_time").resolve()
        result = finalise_test_status(root=output, passed=args.passed, failed=args.failed, command=args.command)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
