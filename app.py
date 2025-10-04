# app.py (merge with existing file)
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
import analysis_runner
from utils import supabase_helper

app = FastAPI()

@app.get("/")
def root():
    return {"message": "CIN Analytics backend running!"}

@app.post("/analysis/run")
def run_analysis(background_tasks: BackgroundTasks):
    """
    Starts a background run of one queued job.
    Frontend should create a job row in analysis_jobs (status 'queued') first.
    """
    background_tasks.add_task(analysis_runner.run_one_job)
    return JSONResponse({"status": "started"})

@app.get("/analysis/next")
def next_job():
    job = supabase_helper.fetch_next_queued_job()
    return job or {}

@app.get("/analysis/{job_id}/results")
def fetch_results(job_id: str):
    # simple retrieval
    conn = supabase_helper.get_conn()
    try:
        import psycopg2.extras
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM analysis_results WHERE job_id=%s ORDER BY created_at DESC", (job_id,))
            rows = cur.fetchall()
            return {"results": rows}
    finally:
        conn.close()
