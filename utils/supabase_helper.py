# utils/supabase_helper.py
import os
import json
import requests
import pandas as pd
from datetime import datetime

# REST API config (required)
SUPABASE_URL = os.getenv("SUPABASE_URL")        # e.g. https://<project>.supabase.co
SUPABASE_KEY = os.getenv("SUPABASE_KEY")        # anon or service_role key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in environment (Supabase REST access).")

REST_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}

# Optional direct DB connection string (postgres URI). Not required for REST job helpers.
DB_URL = os.getenv("SUPABASE_DB_URL")  # e.g. postgresql://postgres:pw@db.xxx.supabase.co:5432/postgres

# ---------- internal helpers for REST calls ----------
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

# ---------- optional DB connector ----------
def get_conn():
    """
    Return a psycopg2 connection using SUPABASE_DB_URL.
    This is optional — job pickup and most helpers use REST and do NOT need DB_URL.
    If you want to enable direct Postgres access, set SUPABASE_DB_URL in environment to:
      postgresql://<user>:<password>@db.<project>.supabase.co:5432/postgres
    Note: Supabase may be IPv6-only on some plans; if your host can't reach the DB you'll see 'Network is unreachable'.
    For hosted platforms like Render, prefer using Supabase REST or the Session Pooler (see Supabase docs) if direct connect fails.
    """
    try:
        import psycopg2
    except Exception:
        raise RuntimeError("psycopg2 is not installed in this environment. Add it to requirements if you need direct DB access.")
    if not DB_URL:
        raise RuntimeError(
            "SUPABASE_DB_URL not set. If you need direct DB access set SUPABASE_DB_URL to the full postgres URI.\n"
            "Get it from Supabase dashboard → Database → Connection string → Primary Database (URI).\n"
            "If you cannot connect due to network/IPv6 issues consider using the Supabase REST endpoints instead."
        )
    # psycopg2.connect will raise if unreachable; caller should handle/log
    return psycopg2.connect(DB_URL, sslmode="require")

# ---------- job helpers (use REST so we do not require DB_URL) ----------
def fetch_next_queued_job():
    """
    Fetch one queued or pending job from analysis_jobs table via Supabase REST.
    Returns a dict or None.
    """
    qs = "?status=in.(queued,pending)&select=*&order=priority.desc,created_at.asc&limit=1"
    try:
        rows = _call_get("/analysis_jobs" + qs)
        return rows[0] if rows else None
    except requests.HTTPError as e:
        raise RuntimeError(f"Supabase fetch_next_queued_job failed: {str(e)}")

def update_job_status(job_id, status, progress=None, started_at=None, finished_at=None, error_message=None):
    """
    Update a job row in analysis_jobs by id (tries id then job_id).
    Uses REST patch.
    """
    body = {"status": status}
    if progress is not None:
        body["progress"] = progress
    if started_at is not None:
        body["started_at"] = started_at if isinstance(started_at, str) else started_at.isoformat()
    if finished_at is not None:
        body["finished_at"] = finished_at if isinstance(finished_at, str) else finished_at.isoformat()
    if error_message is not None:
        body["error_message"] = error_message

    # try update by id, then job_id
    for key in ("id", "job_id"):
        try:
            _call_patch(f"/analysis_jobs?{key}=eq.{job_id}", body)
            return
        except requests.HTTPError:
            continue
    raise RuntimeError("Failed to update job status: no matching primary key (tried id, job_id)")

def append_log(job_id, message, level="INFO"):
    """
    Insert a small log row into analysis_logs (if table exists) via REST so UI can show progress.
    Non-fatal if table missing.
    """
    payload = {
        "job_id": job_id,
        "log_level": level,
        "message": message,
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        _call_post("/analysis_logs", payload)
    except requests.HTTPError:
        # ignore if logs table is not present
        pass

def insert_analysis_result(job_id, result_label, metrics=None, tables=None, plots=None):
    """
    Insert a result row into analysis_results via REST.
    """
    payload = {
        "job_id": job_id,
        "result_label": result_label,
        "metrics": metrics or {},
        "tables": tables or {},
        "plots": plots or {},
        "created_at": datetime.utcnow().isoformat(),
    }
    try:
        _call_post("/analysis_results", payload)
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to insert analysis_result: {e}")

# ---------- dataset loader (via REST) ----------
def load_dataset(table="patients", where_clause=None, select="*"):
    """
    Load a dataset table from Supabase into a pandas DataFrame via REST.
    where_clause may be a PostgREST query string (e.g. "age_years=gt.18") or a dict of params.
    """
    path = f"/{table}?select={select}"
    params = None
    if where_clause:
        if isinstance(where_clause, dict):
            params = where_clause
        else:
            # append string filters
            if not where_clause.startswith("&"):
                path += "&" + where_clause
            else:
                path += where_clause
            params = None
    try:
        rows = _call_get(path, params=params)
        df = pd.DataFrame(rows)
        return df
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to load dataset {table}: {e}")
