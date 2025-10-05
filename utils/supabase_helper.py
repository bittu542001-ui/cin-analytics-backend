# utils/supabase_helper.py
import os
import requests
import json
import pandas as pd
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    # If missing, raise early so logs show clear message
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY not set in environment")

# base REST endpoint for tables
REST_BASE = SUPABASE_URL.rstrip("/") + "/rest/v1"

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Accept": "application/json",
}


def _call_get(path, params=None):
    url = REST_BASE + path
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def _call_post(path, payload):
    url = REST_BASE + path
    r = requests.post(url, headers=HEADERS, data=json.dumps(payload), params={"return": "representation"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _call_patch(path, payload):
    url = REST_BASE + path
    r = requests.patch(url, headers=HEADERS, data=json.dumps(payload), params={"return": "representation"}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------- job helpers used by analysis_runner.py ----------

def fetch_next_queued_job():
    """
    Fetch one queued or pending job from analysis_jobs table.
    Assumes there is an `analysis_jobs` table with columns including 'status' and 'created_at'.
    Returns the first job dictionary or None.
    """
    # Query: status=queued or pending, order by priority desc then created_at asc, limit=1
    qs = "?status=in.(queued,pending)&select=*&order=priority.desc,created_at.asc&limit=1"
    try:
        rows = _call_get("/analysis_jobs" + qs)
        return rows[0] if rows else None
    except requests.HTTPError as e:
        # bubble up as RuntimeError so caller logs it
        raise RuntimeError(f"Supabase fetch_next_queued_job failed: {str(e)}")


def update_job_status(job_id, status, progress=None, started_at=None, finished_at=None, error_message=None):
    """
    Update a job row in analysis_jobs by id.
    job_id: value of the id column (if primary key column is named 'id' or 'job_id' ensure it matches DB)
    """
    # We assume your table's primary key is `id` or `job_id`. Try a couple ways.
    body = {"status": status}
    if progress is not None:
        body["progress"] = progress
    if started_at is not None:
        # ensure ISO string
        body["started_at"] = started_at if isinstance(started_at, str) else started_at.isoformat()
    if finished_at is not None:
        body["finished_at"] = finished_at if isinstance(finished_at, str) else finished_at.isoformat()
    if error_message is not None:
        body["error_message"] = error_message

    # Try update by id first, otherwise by job_id
    for key in ("id", "job_id"):
        try:
            _call_patch(f"/analysis_jobs?{key}=eq.{job_id}", body)
            return
        except requests.HTTPError:
            continue
    raise RuntimeError("Failed to update job status: no matching primary key (tried id, job_id)")


def append_log(job_id, message, level="INFO"):
    """
    Insert a small log row into analysis_logs (if table exists) so UI can show progress.
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
        # don't break main flow if logs table missing
        pass


def insert_analysis_result(job_id, result_label, metrics=None, tables=None, plots=None):
    """
    Insert a row into analysis_results table.
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


# ---------- dataset loader (returns pandas DataFrame) ----------

def load_dataset(table="patients", where_clause=None, select="*"):
    """
    Load a dataset table from Supabase into a pandas DataFrame via REST.
    - table: table name (string)
    - where_clause: simple filters in PostgREST format (e.g. "age_years=gt.18"), or None
    - select: columns to select, default "*"
    """
    path = f"/{table}?select={select}"
    if where_clause:
        # If user passed a raw SQL-like clause, try to translate minimally:
        # If where_clause is a dict with PostgREST params, pass them as query params
        if isinstance(where_clause, dict):
            # add each key= value as query param filters
            params = where_clause
        else:
            # assume string is already PostgREST filter expression in format "col=eq.val" etc.
            # append as query string
            if not where_clause.startswith("&"):
                path += "&" + where_clause
            else:
                path += where_clause
            params = None
    else:
        params = None
    try:
        rows = _call_get(path, params=params)
        df = pd.DataFrame(rows)
        return df
    except requests.HTTPError as e:
        raise RuntimeError(f"Failed to load dataset {table}: {e}")
