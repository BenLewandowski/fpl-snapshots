#!/usr/bin/env python3
"""
fpl_snapshot.py — Task C4: prospective data acquisition for the FPL goal model.

Snapshots, with UTC timestamps and an append-only manifest:
  1. football-data.co.uk fixtures.csv (upcoming fixtures + odds; the future
     closing-line / odds-movement source)
  2. football-data.co.uk E0 season files (results + full odds history)
  3. FPL bootstrap-static, compacted to the per-player fields that matter
     (selected_by_percent, transfer flows, chance_of_playing, news/status)
  4. FPL fixtures JSON (kickoff times; small, hash-deduped)

Design notes:
  - Idempotent: content is sha256-hashed; unchanged content is NOT re-saved
    (manifest still records the fetch attempt).
  - Manifest is the integration contract. The main model reads manifest.csv
    later to reconstruct odds movement and transfer velocity timelines.
  - No third-party deps beyond requests. CSV output only (no pyarrow needed).

Usage:
  python fpl_snapshot.py                          # snapshot everything
  python fpl_snapshot.py --base-dir /path/to/dir  # override output root
  python fpl_snapshot.py --test                   # offline unit tests
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FD_BASE = "https://www.football-data.co.uk"
FPL_BASE = "https://fantasy.premierleague.com/api"

USER_AGENT = (
    "Mozilla/5.0 (compatible; fpl-snapshot/1.0; "
    "+research; contact: local-user)"
)

# Compact per-player columns extracted from bootstrap-static elements.
COMPACT_PLAYER_COLS = [
    "id", "code", "web_name", "first_name", "second_name", "team", "team_code",
    "element_type", "now_cost", "selected_by_percent",
    "transfers_in_event", "transfers_out_event",
    "chance_of_playing_this_round", "chance_of_playing_next_round",
    "status", "news", "news_added",
    "total_points", "points_per_game", "form",
    "minutes", "starts", "goals_scored", "assists",
    "expected_goals", "expected_assists", "expected_goal_involvements",
]

MANIFEST_COLS = [
    "source", "fetched_at_utc", "url", "http_status", "bytes", "sha256",
    "rows", "changed_vs_prev", "saved_path", "notes",
]


@dataclass
class SnapshotConfig:
    base_dir: Path = Path("/content/drive/MyDrive/fpl_cache/snapshots")
    seasons: list = field(default_factory=lambda: ["2026-27"])
    include_raw_bootstrap: bool = False   # raw JSON is ~10-20 MB; off by default
    max_keep_per_source: int = 500        # retention: snapshots per source
    request_timeout_s: int = 60
    retry_attempts: int = 3
    retry_backoff_s: float = 5.0


# ---------------------------------------------------------------------------
# Small pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def season_to_fd_code(season: str) -> str:
    """'2026-27' -> '2627' (football-data directory code)."""
    start = int(season.split("-")[0])
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def csv_row_count(content: bytes) -> int:
    """Data-row count for a CSV blob (header excluded). -1 on parse failure."""
    try:
        text = content.decode("utf-8-sig", errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        return max(len(rows) - 1, 0)
    except Exception:
        return -1


def extract_compact_bootstrap(payload: dict) -> tuple[str, dict]:
    """
    Reduce bootstrap-static JSON to a compact CSV string + event summary.

    Returns (csv_text, summary) where summary carries GW context used for
    manifest notes and raw-snapshot gating.
    """
    elements = payload.get("elements", [])
    events = payload.get("events", [])

    current_event, next_event = None, None
    for ev in events:
        if ev.get("is_current"):
            current_event = ev
        if ev.get("is_next"):
            next_event = ev
    if current_event is None and next_event is not None:
        current_event = next_event  # pre-season / between seasons

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COMPACT_PLAYER_COLS, extrasaction="ignore")
    writer.writeheader()
    for el in elements:
        row = {c: el.get(c) for c in COMPACT_PLAYER_COLS}
        writer.writerow(row)

    summary = {
        "n_players": len(elements),
        "current_event": (current_event or {}).get("id"),
        "current_event_finished": (current_event or {}).get("finished"),
        "next_deadline": (next_event or {}).get("deadline_time"),
    }
    return buf.getvalue(), summary


def prune_files(source_dir: Path, prefix: str, max_keep: int) -> int:
    """Keep the newest max_keep files matching prefix; delete older. Returns n deleted."""
    files = sorted(source_dir.glob(f"{prefix}_*"), key=lambda p: p.name)
    to_delete = files[:-max_keep] if len(files) > max_keep else []
    for f in to_delete:
        try:
            f.unlink()
        except OSError:
            pass
    return len(to_delete)


# ---------------------------------------------------------------------------
# Fetch + persistence
# ---------------------------------------------------------------------------

def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def fetch_bytes(session: requests.Session, url: str, cfg: SnapshotConfig):
    """Return (status_code, content_bytes, error_note)."""
    last_err = ""
    for attempt in range(1, cfg.retry_attempts + 1):
        try:
            r = session.get(url, timeout=cfg.request_timeout_s)
            if r.status_code == 200 and r.content:
                return r.status_code, r.content, ""
            last_err = f"status={r.status_code}"
        except requests.RequestException as e:
            last_err = f"{type(e).__name__}: {e}"
        if attempt < cfg.retry_attempts:
            time.sleep(cfg.retry_backoff_s * attempt)
    return 0, b"", last_err


def manifest_path(cfg: SnapshotConfig) -> Path:
    return cfg.base_dir / "manifest.csv"


def load_manifest_hashes(cfg: SnapshotConfig) -> dict:
    """Latest saved-content sha256 per source (for change detection)."""
    path = manifest_path(cfg)
    latest = {}
    if not path.exists():
        return latest
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("saved_path"):
                latest[row["source"]] = row["sha256"]
    return latest


def append_manifest(cfg: SnapshotConfig, rows: list) -> None:
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in MANIFEST_COLS})


def save_snapshot(cfg: SnapshotConfig, source: str, content: bytes, ext: str) -> Path:
    d = cfg.base_dir / source
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{source}_{utc_stamp()}.{ext}"
    path.write_bytes(content)
    return path


def record(cfg: SnapshotConfig, manifest_rows: list, source: str, url: str,
           status: int, content: bytes, n_rows: int, notes: str,
           prev_hashes: dict, ext: str = "csv") -> bool:
    """Hash-dedupe + save + manifest-append for one fetched blob. Returns ok."""
    ok = status == 200 and len(content) > 0
    if not ok:
        manifest_rows.append({
            "source": source, "fetched_at_utc": utc_now_iso(), "url": url,
            "http_status": status, "bytes": 0, "sha256": "", "rows": "",
            "changed_vs_prev": "", "saved_path": "", "notes": f"FETCH_FAIL {notes}",
        })
        return False

    digest = sha256_bytes(content)
    changed = prev_hashes.get(source) != digest
    saved = ""
    if changed:
        saved_path = save_snapshot(cfg, source, content, ext)
        saved = str(saved_path)
        prev_hashes[source] = digest
        prune_files(cfg.base_dir / source, source, cfg.max_keep_per_source)

    manifest_rows.append({
        "source": source, "fetched_at_utc": utc_now_iso(), "url": url,
        "http_status": status, "bytes": len(content), "sha256": digest,
        "rows": n_rows, "changed_vs_prev": int(changed),
        "saved_path": saved, "notes": notes,
    })
    return True


# ---------------------------------------------------------------------------
# Snapshot jobs
# ---------------------------------------------------------------------------

def snapshot_football_data(cfg: SnapshotConfig, session: requests.Session,
                           manifest_rows: list, prev_hashes: dict) -> bool:
    ok_all = True
    urls = [f"{FD_BASE}/fixtures.csv"]
    for season in cfg.seasons:
        urls.append(f"{FD_BASE}/mmz4281/{season_to_fd_code(season)}/E0.csv")

    for url in urls:
        source = ("fd_fixtures" if url.endswith("/fixtures.csv")
                  else f"fd_e0_{url.split('/')[-2]}")
        status, content, err = fetch_bytes(session, url, cfg)
        n = csv_row_count(content) if status == 200 else ""
        ok = record(cfg, manifest_rows, source, url, status, content,
                    n, err, prev_hashes, ext="csv")
        ok_all = ok_all and ok
        print(f"  [{source}] status={status} rows={n} "
              f"{'saved' if ok and prev_hashes.get(source) else ''}".rstrip())
    return ok_all


def snapshot_fpl(cfg: SnapshotConfig, session: requests.Session,
                 manifest_rows: list, prev_hashes: dict) -> bool:
    ok_all = True

    # --- bootstrap-static (compact) ---
    url = f"{FPL_BASE}/bootstrap-static/"
    status, content, err = fetch_bytes(session, url, cfg)
    if status == 200:
        try:
            payload = json.loads(content.decode("utf-8"))
            csv_text, summary = extract_compact_bootstrap(payload)
            compact_bytes = csv_text.encode("utf-8")
            notes = (f"gw={summary['current_event']} "
                     f"finished={summary['current_event_finished']} "
                     f"next_deadline={summary['next_deadline']}")
            ok = record(cfg, manifest_rows, "fpl_bootstrap_compact", url,
                        status, compact_bytes, summary["n_players"],
                        notes, prev_hashes, ext="csv")
            ok_all = ok_all and ok
            print(f"  [fpl_bootstrap_compact] gw={summary['current_event']} "
                  f"players={summary['n_players']} next_deadline={summary['next_deadline']}")

            # Optional raw dump: only when the current GW rolls over, to keep
            # storage bounded (one raw file per GW is plenty for backfills).
            if cfg.include_raw_bootstrap:
                raw_source = "fpl_bootstrap_raw"
                raw_key = f"{raw_source}_gw{summary['current_event']}"
                raw_dir = cfg.base_dir / raw_source
                raw_dir.mkdir(parents=True, exist_ok=True)
                if not any(raw_dir.glob(f"{raw_key}_*")):
                    raw_path = raw_dir / f"{raw_key}_{utc_stamp()}.json"
                    raw_path.write_bytes(content)
                    manifest_rows.append({
                        "source": raw_source, "fetched_at_utc": utc_now_iso(),
                        "url": url, "http_status": status, "bytes": len(content),
                        "sha256": sha256_bytes(content), "rows": summary["n_players"],
                        "changed_vs_prev": 1, "saved_path": str(raw_path),
                        "notes": f"per-GW raw dump gw={summary['current_event']}",
                    })
        except (json.JSONDecodeError, KeyError) as e:
            manifest_rows.append({
                "source": "fpl_bootstrap_compact", "fetched_at_utc": utc_now_iso(),
                "url": url, "http_status": status, "bytes": len(content),
                "sha256": "", "rows": "", "changed_vs_prev": "", "saved_path": "",
                "notes": f"PARSE_FAIL {type(e).__name__}: {e}",
            })
            ok_all = False
    else:
        record(cfg, manifest_rows, "fpl_bootstrap_compact", url, status,
               b"", "", err, prev_hashes)
        ok_all = False

    # --- fixtures JSON (kickoff times; hash-deduped) ---
    url = f"{FPL_BASE}/fixtures/"
    status, content, err = fetch_bytes(session, url, cfg)
    n = ""
    if status == 200:
        try:
            n = len(json.loads(content.decode("utf-8")))
        except json.JSONDecodeError:
            n = ""
    ok = record(cfg, manifest_rows, "fpl_fixtures", url, status, content,
                n, err, prev_hashes, ext="json")
    ok_all = ok_all and ok
    print(f"  [fpl_fixtures] status={status} fixtures={n}")
    return ok_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: SnapshotConfig) -> int:
    print("=" * 60)
    print(f"FPL SNAPSHOT — {utc_now_iso()}")
    print(f"  base_dir: {cfg.base_dir}")
    print("=" * 60)
    cfg.base_dir.mkdir(parents=True, exist_ok=True)

    session = get_session()
    prev_hashes = load_manifest_hashes(cfg)
    manifest_rows: list = []

    print("Snapshotting football-data...")
    ok1 = snapshot_football_data(cfg, session, manifest_rows, prev_hashes)
    print("Snapshotting FPL API...")
    ok2 = snapshot_fpl(cfg, session, manifest_rows, prev_hashes)

    append_manifest(cfg, manifest_rows)
    n_saved = sum(1 for r in manifest_rows if r.get("saved_path"))
    print(f"Done. {len(manifest_rows)} sources fetched, {n_saved} new snapshots saved "
          f"(hash-deduped the rest). Manifest: {manifest_path(cfg)}")
    # Exit code 2 = partial failure (alerting hook for cron/CI).
    return 0 if (ok1 and ok2) else 2


# ---------------------------------------------------------------------------
# Offline unit tests
# ---------------------------------------------------------------------------

class SnapshotTests(unittest.TestCase):
    def test_season_code(self):
        self.assertEqual(season_to_fd_code("2026-27"), "2627")
        self.assertEqual(season_to_fd_code("2019-20"), "1920")

    def test_hash_stable_and_sensitive(self):
        a = sha256_bytes(b"Div,Date\nE0,01/08/2026\n")
        self.assertEqual(a, sha256_bytes(b"Div,Date\nE0,01/08/2026\n"))
        self.assertNotEqual(a, sha256_bytes(b"Div,Date\nE0,02/08/2026\n"))

    def test_csv_row_count(self):
        self.assertEqual(csv_row_count(b"a,b,c\n1,2,3\n4,5,6\n"), 2)
        self.assertEqual(csv_row_count(b"a,b\n"), 0)
        self.assertEqual(csv_row_count(b""), 0)

    def test_extract_compact_bootstrap(self):
        payload = {
            "events": [
                {"id": 5, "is_current": True, "finished": False,
                 "deadline_time": "2026-10-03T17:30:00Z"},
                {"id": 6, "is_next": True,
                 "deadline_time": "2026-10-10T17:30:00Z"},
            ],
            "elements": [
                {"id": 1, "code": 101, "web_name": "Haaland", "team": 13,
                 "element_type": 4, "now_cost": 150, "selected_by_percent": "55.1",
                 "transfers_in_event": 12345, "transfers_out_event": 678,
                 "status": "a", "news": "", "total_points": 40,
                 "irrelevant_field": "x"},
                {"id": 2, "code": 102, "web_name": "Saka", "team": 1,
                 "element_type": 3, "now_cost": 105, "selected_by_percent": "30.2",
                 "transfers_in_event": 5000, "transfers_out_event": 900,
                 "status": "d", "news": "Knock - 50% chance of playing",
                 "total_points": 30},
            ],
        }
        csv_text, summary = extract_compact_bootstrap(payload)
        rows = list(csv.DictReader(io.StringIO(csv_text)))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["web_name"], "Haaland")
        self.assertEqual(rows[0]["selected_by_percent"], "55.1")
        self.assertEqual(rows[1]["news"], "Knock - 50% chance of playing")
        self.assertNotIn("irrelevant_field", rows[0])  # compaction works
        self.assertEqual(summary["current_event"], 5)
        self.assertEqual(summary["next_deadline"], "2026-10-10T17:30:00Z")
        self.assertEqual(summary["n_players"], 2)

    def test_prune_keeps_newest(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(5):
                (d / f"src_2026100{i}T000000Z.csv").write_text(f"v{i}")
            deleted = prune_files(d, "src", max_keep=3)
            self.assertEqual(deleted, 2)
            remaining = sorted(p.name for p in d.glob("src_*"))
            self.assertEqual(len(remaining), 3)
            self.assertIn("src_20261004T000000Z.csv", remaining)  # newest kept

    def test_manifest_roundtrip_and_hash_tracking(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SnapshotConfig(base_dir=Path(td))
            rows = [{
                "source": "fd_fixtures", "fetched_at_utc": utc_now_iso(),
                "url": "u", "http_status": 200, "bytes": 10,
                "sha256": "abc123", "rows": 5, "changed_vs_prev": 1,
                "saved_path": "/x/y.csv", "notes": "",
            }]
            append_manifest(cfg, rows)
            hashes = load_manifest_hashes(cfg)
            self.assertEqual(hashes["fd_fixtures"], "abc123")

    def test_record_skips_unchanged_content(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SnapshotConfig(base_dir=Path(td))
            prev = {}
            mrows: list = []
            content = b"Div,Date\nE0,01/08/2026\n"
            record(cfg, mrows, "fd_fixtures", "u", 200, content, 1, "", prev)
            record(cfg, mrows, "fd_fixtures", "u", 200, content, 1, "", prev)
            self.assertEqual(mrows[0]["changed_vs_prev"], 1)   # first: saved
            self.assertEqual(mrows[1]["changed_vs_prev"], 0)   # second: deduped
            self.assertEqual(mrows[1]["saved_path"], "")
            self.assertTrue(mrows[0]["saved_path"])


def run_tests() -> int:
    suite = unittest.TestLoader().loadTestsFromTestCase(SnapshotTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main() -> int:
    p = argparse.ArgumentParser(description="FPL/football-data snapshotter (Task C4)")
    p.add_argument("--base-dir", default=str(SnapshotConfig.base_dir),
                   help="Snapshot root (default: Colab Drive cache path)")
    p.add_argument("--seasons", default="2026-27",
                   help="Comma-separated seasons for E0 files, e.g. '2025-26,2026-27'")
    p.add_argument("--include-raw", action="store_true",
                   help="Also dump raw bootstrap-static JSON once per GW")
    p.add_argument("--max-keep", type=int, default=500,
                   help="Snapshots retained per source (default 500)")
    p.add_argument("--test", action="store_true", help="Run offline unit tests")
    args = p.parse_args()

    if args.test:
        return run_tests()

    cfg = SnapshotConfig(
        base_dir=Path(args.base_dir),
        seasons=[s.strip() for s in args.seasons.split(",") if s.strip()],
        include_raw_bootstrap=args.include_raw,
        max_keep_per_source=args.max_keep,
    )
    return run(cfg)


if __name__ == "__main__":
    sys.exit(main())
