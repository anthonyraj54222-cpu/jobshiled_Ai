"""
database.py
===========
SQLite Persistent User History & Audit Database for JobShield AI.

Manages `jobshield_history.db` with:
  - Full scan history logging across all input modalities (Text, PDF, Image OCR,
    Word .docx, .txt, Email .eml, Spreadsheets, HTML, etc.)
  - Filtering, search, bookmarking, user notes, record deletion, and CSV/JSON export
  - Real-time analytics aggregation (fraud rate, risk score distribution, modality breakdown)
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobshield_history.db")


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes the SQLite database schema and indexes."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_profile TEXT DEFAULT 'Analyst',
                source_type TEXT NOT NULL,
                file_name TEXT DEFAULT 'Direct Text Input',
                file_size_kb REAL DEFAULT 0.0,
                extraction_method TEXT DEFAULT 'Direct Text',
                job_title TEXT,
                company_name TEXT,
                extracted_text TEXT,
                word_count INTEGER DEFAULT 0,
                prediction INTEGER NOT NULL,
                verdict_label TEXT NOT NULL,
                risk_tier TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                confidence_pct REAL NOT NULL,
                risk_score REAL NOT NULL,
                urgency_score INTEGER DEFAULT 0,
                scam_score INTEGER DEFAULT 0,
                legitimacy_score INTEGER DEFAULT 0,
                vagueness_score INTEGER DEFAULT 0,
                hype_score INTEGER DEFAULT 0,
                branch_scores_json TEXT,
                reasons_json TEXT,
                positive_notes_json TEXT,
                tokens_json TEXT,
                similar_cases_json TEXT,
                user_notes TEXT DEFAULT '',
                is_bookmarked INTEGER DEFAULT 0
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_ts ON scan_history(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_pred ON scan_history(prediction)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_history_source ON scan_history(source_type)")
    conn.close()


def log_scan(
    analysis_result: dict,
    extracted_text: str,
    source_type: str = "Direct Text",
    file_name: str = "Pasted Job Text",
    file_size_kb: float = 0.0,
    extraction_method: str = "Direct Text Input",
    job_title: str = "",
    company_name: str = "",
    user_profile: str = "Default User",
    timestamp: str = None,
    db_path: str = DEFAULT_DB_PATH,
) -> int:
    """Saves an analyzed job posting and its full explainability report into SQLite."""
    init_db(db_path)
    if not timestamp:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    signals = analysis_result.get("signals", {})
    word_count = int(signals.get("text_length", len((extracted_text or "").split())))

    if not job_title:
        lines = [ln.strip() for ln in (extracted_text or "").splitlines() if ln.strip()]
        job_title = lines[0][:80] if lines else "Job Posting Scan"
    if not company_name:
        company_name = "Not Specified"

    tokens_payload = {
        "suspicious": analysis_result.get("suspicious_tokens", []),
        "legitimate": analysis_result.get("legitimate_tokens", []),
    }

    conn = get_connection(db_path)
    with conn:
        cur = conn.execute(
            """
            INSERT INTO scan_history (
                timestamp, user_profile, source_type, file_name, file_size_kb,
                extraction_method, job_title, company_name, extracted_text,
                word_count, prediction, verdict_label, risk_tier,
                fraud_probability, confidence_pct, risk_score,
                urgency_score, scam_score, legitimacy_score, vagueness_score, hype_score,
                branch_scores_json, reasons_json, positive_notes_json,
                tokens_json, similar_cases_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                user_profile,
                source_type,
                file_name,
                float(file_size_kb),
                extraction_method,
                job_title[:90],
                company_name[:90],
                extracted_text,
                word_count,
                int(analysis_result["prediction"]),
                analysis_result["verdict_label"],
                analysis_result["risk_tier"],
                float(analysis_result["fraud_probability"]),
                float(analysis_result["confidence_pct"]),
                float(analysis_result["risk_score"]),
                int(signals.get("urgency_score", 0)),
                int(signals.get("scam_composite_score", 0)),
                int(signals.get("legitimacy_score", 0)),
                int(signals.get("vagueness_score", 0)),
                int(signals.get("hype_score", 0)),
                json.dumps(analysis_result.get("branch_scores", {})),
                json.dumps(analysis_result.get("reasons", [])),
                json.dumps(analysis_result.get("positive_notes", [])),
                json.dumps(tokens_payload),
                json.dumps(analysis_result.get("similar_cases", [])),
            ),
        )
        scan_id = cur.lastrowid
    conn.close()
    return int(scan_id)


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k, target in [
        ("branch_scores_json", "branch_scores"),
        ("reasons_json", "reasons"),
        ("positive_notes_json", "positive_notes"),
        ("tokens_json", "tokens"),
        ("similar_cases_json", "similar_cases"),
    ]:
        try:
            d[target] = json.loads(d.get(k) or "{}")
        except Exception:
            d[target] = [] if "reasons" in k or "notes" in k or "cases" in k else {}
    return d


def get_history(
    limit: int = 200,
    search_query: str = "",
    verdict_filter: str = "ALL",
    source_filter: str = "ALL",
    bookmarked_only: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> list[dict]:
    """Fetches filtered user scan history ordered from newest to oldest."""
    init_db(db_path)
    query = "SELECT * FROM scan_history WHERE 1=1"
    params = []

    if search_query and search_query.strip():
        q = f"%{search_query.strip()}%"
        query += " AND (job_title LIKE ? OR company_name LIKE ? OR file_name LIKE ? OR extracted_text LIKE ? OR user_notes LIKE ?)"
        params.extend([q, q, q, q, q])

    if verdict_filter == "FAKE":
        query += " AND prediction = 1"
    elif verdict_filter == "REAL":
        query += " AND prediction = 0"
    elif verdict_filter in ("HIGH", "MODERATE", "CAUTION", "LOW"):
        query += " AND risk_tier = ?"
        params.append(verdict_filter)

    if source_filter and source_filter != "ALL":
        query += " AND source_type LIKE ?"
        params.append(f"%{source_filter}%")

    if bookmarked_only:
        query += " AND is_bookmarked = 1"

    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    conn = get_connection(db_path)
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def get_scan_by_id(scan_id: int, db_path: str = DEFAULT_DB_PATH) -> dict | None:
    init_db(db_path)
    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM scan_history WHERE id = ?", (int(scan_id),)).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


def toggle_bookmark(scan_id: int, db_path: str = DEFAULT_DB_PATH) -> bool:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        row = conn.execute("SELECT is_bookmarked FROM scan_history WHERE id = ?", (int(scan_id),)).fetchone()
        if not row:
            return False
        new_val = 0 if row["is_bookmarked"] else 1
        conn.execute("UPDATE scan_history SET is_bookmarked = ? WHERE id = ?", (new_val, int(scan_id)))
    conn.close()
    return bool(new_val)


def update_user_notes(scan_id: int, notes: str, db_path: str = DEFAULT_DB_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute("UPDATE scan_history SET user_notes = ? WHERE id = ?", (str(notes), int(scan_id)))
    conn.close()


def delete_scan(scan_id: int, db_path: str = DEFAULT_DB_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM scan_history WHERE id = ?", (int(scan_id),))
    conn.close()


def clear_all_history(db_path: str = DEFAULT_DB_PATH):
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute("DELETE FROM scan_history")
    conn.close()


def get_dashboard_stats(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Computes summary KPIs and distributions from the SQLite history database."""
    init_db(db_path)
    conn = get_connection(db_path)
    total = conn.execute("SELECT COUNT(*) FROM scan_history").fetchone()[0]
    if total == 0:
        conn.close()
        return {
            "total_scans": 0,
            "fake_count": 0,
            "real_count": 0,
            "bookmarked_count": 0,
            "avg_risk_score": 0.0,
            "avg_confidence": 0.0,
            "by_source": [],
            "by_risk_tier": [],
        }

    fake_cnt = conn.execute("SELECT COUNT(*) FROM scan_history WHERE prediction = 1").fetchone()[0]
    real_cnt = total - fake_cnt
    bm_cnt = conn.execute("SELECT COUNT(*) FROM scan_history WHERE is_bookmarked = 1").fetchone()[0]
    avg_risk = conn.execute("SELECT AVG(risk_score) FROM scan_history").fetchone()[0] or 0.0
    avg_conf = conn.execute("SELECT AVG(confidence_pct) FROM scan_history").fetchone()[0] or 0.0

    src_rows = conn.execute(
        "SELECT source_type, COUNT(*) as cnt, SUM(prediction) as fakes FROM scan_history GROUP BY source_type ORDER BY cnt DESC"
    ).fetchall()
    tier_rows = conn.execute(
        "SELECT risk_tier, COUNT(*) as cnt FROM scan_history GROUP BY risk_tier ORDER BY cnt DESC"
    ).fetchall()
    conn.close()

    return {
        "total_scans": int(total),
        "fake_count": int(fake_cnt),
        "real_count": int(real_cnt),
        "bookmarked_count": int(bm_cnt),
        "fraud_rate_pct": round((fake_cnt / total) * 100.0, 1),
        "avg_risk_score": round(float(avg_risk), 1),
        "avg_confidence": round(float(avg_conf), 1),
        "by_source": [dict(r) for r in src_rows],
        "by_risk_tier": [dict(r) for r in tier_rows],
    }



def seed_demo_history_if_empty(*args, **kwargs):
    """No-op: History starts completely clean and only records actual user actions."""
    pass
