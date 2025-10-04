# utils/supabase_helper.py
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv("SUPABASE_DB_URL")

def get_conn():
    if not DB_URL:
        raise RuntimeError("SUPABASE_DB_URL not set")
    return psycopg2.connect(DB_URL, sslmode="require")

# JOB helpers
def fetch_next_queued_job():
    q = """
      SELECT * FROM analysis_jobs
      WHERE status IN ('queued','pending')
      ORDER BY priority DESC, created_at ASC
      LIMIT 1
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q)
            job = cur.fetchone()
            return job
    finally:
        conn.close()

def update_job_status(job_id, status, progress=None, started_at=None, finished_at=None, error_message=None):
    updates = []
    params = []
    updates.append("status = %s"); params.append(status)
    if progress is not None:
        updates.append("progress = %s"); params.append(progress)
    if started_at:
        updates.append("started_at = %s"); params.append(started_at)
    if finished_at:
        updates.append("finished_at = %s"); params.append(finished_at)
    if error_message is not None:
        updates.append("error_message = %s"); params.append(error_message)

    q = f"UPDATE analysis_jobs SET {', '.join(updates)} WHERE job_id = %s"
    params.append(job_id)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(q, params)
            conn.commit()
    finally:
        conn.close()

def insert_analysis_result(job_id, result_label, metrics, tables=None, arrays=None, plots=None, artifacts=None):
    q = """
      INSERT INTO analysis_results (job_id, result_label, metrics, tables, arrays, plots, artifacts)
      VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING result_id, created_at
    """
    conn = get_conn()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(q, (
                job_id,
                result_label,
                json.dumps(metrics) if metrics is not None else None,
                json.dumps(tables) if tables is not None else None,
                json.dumps(arrays) if arrays is not None else None,
                json.dumps(plots) if plots is not None else None,
                json.dumps(artifacts) if artifacts is not None else None,
            ))
            row = cur.fetchone()
            conn.commit()
            return row
    finally:
        conn.close()

def append_log(job_id, message, level="INFO"):
    q = "INSERT INTO analysis_logs (job_id, level, message) VALUES (%s,%s,%s)"
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(q, (job_id, level, message))
            conn.commit()
    finally:
        conn.close()

# convenience to load patients into pandas
import pandas as pd
def load_dataset(table='patients', where_clause=None, columns=None, limit=None):
    cols = "*" if not columns else ",".join(columns)
    q = f"SELECT {cols} FROM {table}"
    if where_clause:
        q += " WHERE " + where_clause
    if limit:
        q += f" LIMIT {int(limit)}"
    conn = get_conn()
    try:
        df = pd.read_sql(q, conn)
        return df
    finally:
        conn.close()
