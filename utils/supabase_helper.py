# utils/supabase_helper.py
"""
Robust supabase helper that supports:
 - REST (PostgREST) via SUPABASE_URL + SUPABASE_KEY (preferred)
 - Fallback direct Postgres via SUPABASE_DB_URL (psycopg2) if provided

Provides functions used by analysis_runner.py:
 - fetch_next_queued_job()
 - update_job_status(job_id, status, ...)
 - append_log(job_id, message, level="INFO")
 - insert_analysis_result(job_id, result_label, metrics=..., tables=..., plots=...)
 - load_dataset(table="patients", where_clause=None, select="*")
 - insert_job(payload_dict, priority=50, status="queued")
"""

import os
import json
import time
from datetime import datetime

# networking libs
import requests
import pandas as pd

# optional DB driver - only imported when needed
_psycopg2 = None

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")  # e.g. https://xyz.supabase.co
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role preferred
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")  # e.g. postgresql://postgres:pw@db....supabase.co:5432/postgres

# Determine mode
USE_REST = bool(SUPABASE_URL and SUPABASE_KEY)
USE_DB = bool(SUPABASE_DB_URL)

if not USE_REST and not USE_DB:
    raise RuntimeError("Supabase configuration missing: provide SUPABASE_URL+SUPABASE_KEY (preferred) or SUPABASE_DB_URL (fallback)")

# REST base
REST_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1" if USE_REST else None
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
} if USE_REST else None

# ---- Helper helpers -----------------------------------------------------
def _log_mode():
    if USE_REST:
        return f"REST mode (SUPABASE_URL={SUPABASE_URL})"
    else:
        return f"DB mode (SUPABASE_DB_URL={SUPABASE_DB_URL[:40]}...)"

def _call_get(path, params=None, timeout=30):
    url = REST_BASE + path
    r = requests.get(url, headers=HEADERS, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_post(path, payload, timeout=30):
    url = REST_BASE + path
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), params={"return": "representation"}, timeout=timeout)
    r.raise_for_status()
    return r.json()

def _call_patch(path, payload, timeout=30):
    url = REST_BASE + path
    r = requests.patch(url, headers=HEADERS, data=json.dumps(payload), params={"return": "representation"}, timeout=timeout)
    r.raise_for_status()
    return r.json()

# ---- DB helpers (lazy import) --------------------------------------------
def _ensure_psycopg2():
    global _psycopg2
    if _psycopg2 is None:
        try:
            import psycopg2
            import psycopg2.extras as _extras
            _psycopg2 = (psycopg2, _extras)
        except Exception as e:
            raise RuntimeError(f"psycopg2 is required for DB mode but not available: {e}")

def _get_db_conn():
    """
    Return a new psycopg2 connection using SUPABASE_DB_URL.
    Caller must close().
    """
    _ensure_psycopg2()
    psycopg2, _ = _psycopg2
    # use sslmode=require (Supabase needs ssl)
    return psycopg2.connect(SUPABASE_DB_URL, sslmode="require")

# ---- Public API used by analysis_runner.py -------------------------------

def fetch_next_queued_job():
    """
    Return one queued or pending job as dict, or None.
    Uses REST if available, otherwise DB.
    """
    if USE_REST:
        try:
            qs = "?status=in.(queued,pending)&select=*&order=priority.desc,created_at.asc&limit=1"
            rows = _call_get("/analysis_jobs" + qs)
            return rows[0] if rows else None
        except requests.HTTPError as e:
            raise RuntimeError(f"Supabase REST fetch_next_queued_job failed: {e}")
    else:
        # DB mode
        try:
            psycopg2, extras = _ensure_psycopg2() or _psycopg2
        except RuntimeError:
            _ensure_psycopg2()
        conn = _get_db_conn()
        try:
            cur = conn.cursor(cursor_factory=_psycopg2[1].RealDictCursor)
            q = """
            SELECT * FROM analysis_jobs
            WHERE status IN ('queued','pending')
            ORDER BY priority DESC, created_at ASC
            LIMIT 1
            """
            cur.execute(q)
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

def update_job_status(job_id, status, progress=None, started_at=None, finished_at=None, error_message=None):
    """
    Update a job row by job_id (tries REST then DB).
    """
    payload = {"status": status}
    if progress is not None:
        payload["progress"] = progress
    if started_at is not None:
        payload["started_at"] = started_at.isoformat() if isinstance(started_at, datetime) else started_at
    if finished_at is not None:
        payload["finished_at"] = finished_at.isoformat() if isinstance(finished_at, datetime) else finished_at
    if error_message is not None:
        payload["error_message"] = error_message

    if USE_REST:
        # try id then job_id
        for key in ("id", "job_id"):
            try:
                _call_patch(f"/analysis_jobs?{key}=eq.{job_id}", payload)
                return
            except requests.HTTPError:
                continue
        # if we get here it's an error
        raise RuntimeError("Failed to update job status via REST: no matching job id")
    else:
        conn = _get_db_conn()
        try:
            cur = conn.cursor()
            set_parts = []
            vals = []
            for k, v in payload.items():
                set_parts.append(f"{k} = %s")
                vals.append(v)
            vals.append(job_id)
            q = f"UPDATE analysis_jobs SET {', '.join(set_parts)} WHERE job_id = %s"
            cur.execute(q, tuple(vals))
            conn.commit()
        finally:
            conn.close()

def append_log(job_id, message, level="INFO"):
    """
    Insert a log row into analysis_logs (best-effort).
    """
    payload = {
        "job_id": job_id,
        "message": message,
        "level": level,
        "created_at": datetime.utcnow().isoformat()
    }
    if USE_REST:
        try:
            _call_post("/analysis_logs", payload)
        except requests.HTTPError:
            # ignore if logs table missing
            pass
    else:
        conn = _get_db_conn()
        try:
            cur = conn.cursor()
            q = "INSERT INTO analysis_logs (job_id, message, level, created_at) VALUES (%s,%s,%s,%s)"
            cur.execute(q, (job_id, message, level, datetime.utcnow()))
            conn.commit()
        finally:
            conn.close()

def insert_analysis_result(job_id, result_label, metrics=None, tables=None, plots=None):
    """
    Insert result row into analysis_results.
    """
    payload = {
        "job_id": job_id,
        "result_label": result_label,
        "metrics": metrics or {},
        "tables": tables or {},
        "plots": plots or {},
        "created_at": datetime.utcnow().isoformat()
    }
    if USE_REST:
        try:
            _call_post("/analysis_results", payload)
        except requests.HTTPError as e:
            raise RuntimeError(f"Failed to insert_analysis_result via REST: {e}")
    else:
        conn = _get_db_conn()
        try:
            cur = conn.cursor()
            q = """
            INSERT INTO analysis_results (job_id, result_label, metrics_json, tables_json, plots_json, created_at)
            VALUES (%s,%s,%s,%s,%s,%s)
            """
            cur.execute(q, (
                job_id,
                result_label,
                json.dumps(metrics or {}, default=str),
                json.dumps(tables or {}, default=str),
                json.dumps(plots or {}, default=str),
                datetime.utcnow()
            ))
            conn.commit()
        finally:
            conn.close()

def load_dataset(table="patients", where_clause=None, select="*"):
    """
    Return pandas DataFrame for the table. where_clause for REST should be PostgREST filter fragment
    (e.g. "age_years=gt.18"). For DB mode the where_clause is appended directly as SQL WHERE.
    """
    if USE_REST:
        path = f"/{table}?select={select}"
        params = None
        if where_clause:
            # if string, append as query fragment
            if isinstance(where_clause, str):
                if not where_clause.startswith("&"):
                    path += "&" + where_clause
            elif isinstance(where_clause, dict):
                params = where_clause
        rows = _call_get(path, params=params)
        return pd.DataFrame(rows)
    else:
        conn = _get_db_conn()
        try:
            cur = conn.cursor(cursor_factory=_psycopg2[1].RealDictCursor)
            q = f"SELECT {select} FROM {table}"
            if where_clause:
                q += f" WHERE {where_clause}"
            cur.execute(q)
            rows = cur.fetchall()
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        finally:
            conn.close()

def insert_job(payload_dict, priority=50, status="queued"):
    """
    Convenience: insert a job row and return created job id.
    Uses REST if available; DB otherwise.
    """
    payload = {
        "payload": payload_dict,
        "status": status,
        "priority": priority,
        "created_at": datetime.utcnow().isoformat()
    }
    if USE_REST:
        res = _call_post("/analysis_jobs", payload)
        # PostgREST returns an array of created rows when return=representation
        # try to extract id field 'job_id' or 'id'
        if isinstance(res, list) and len(res) > 0:
            row = res[0]
            return row.get("job_id") or row.get("id")
        return None
    else:
        conn = _get_db_conn()
        try:
            cur = conn.cursor()
            q = "INSERT INTO analysis_jobs (payload, status, priority, created_at) VALUES (%s,%s,%s,%s) RETURNING job_id"
            cur.execute(q, (json.dumps(payload_dict), status, int(priority), datetime.utcnow()))
            job_id = cur.fetchone()[0]
            conn.commit()
            return job_id
        finally:
            conn.close()
