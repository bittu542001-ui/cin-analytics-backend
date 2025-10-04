# app.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import traceback
import json

# local modules (assumes these exist in your repo)
import analysis_runner
from utils import supabase_helper

# optional DB access for fetch_results
import psycopg2
import psycopg2.extras

app = FastAPI(title="CIN Analytics backend")

# ---------------------------
# Request model for POST /analysis/run
# ---------------------------
class AnalysisRequest(BaseModel):
    dataset_table: str
    selected_parameters: List[str]
    selected_tests: List[str]
    options: Optional[Dict] = None

# ---------------------------
# Root
# ---------------------------
@app.get("/")
def root():
    return {"message": "CIN Analytics backend running!"}

# ---------------------------
# POST /analysis/run
# Accepts JSON body (so Swagger shows editor) and schedules runner
# ---------------------------
@app.post("/analysis/run")
def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Starts a background run of one queued job.
    Either:
     - The frontend has already created a 'queued' job in analysis_jobs (DB) and runner will pick it up,
     - OR you can POST a request body describing dataset_table/parameters/tests and
       we can (optionally) insert a queued job in DB using supabase_helper.
    This endpoint also triggers the background runner task so it wakes and picks jobs.
    """
    received = request.dict()

    # Optionally: create a job row in DB from request payload so runner picks it up.
    # If you want automatic creation, uncomment these lines and ensure supabase_helper.create_job_from_request exists.
    # try:
    #     supabase_helper.create_job_from_request(received)
    # except Exception as e:
    #     # log but don't fail the request
    #     print("Warning: could not create DB job from request:", e)

    # Fire the runner in background (non-blocking)
    try:
        background_tasks.add_task(analysis_runner.run_one_job)
    except Exception as e:
        # don't fail if background schedule fails; we still return the parsed payload for debug
        print("Failed to schedule analysis_runner.run_one_job as background task:", e)
        traceback.print_exc()

    # return the parsed payload for confirmation
    return JSONResponse({"status": "started", "received": received})

# ---------------------------
# GET /analysis/next  -> return next queued job (for frontend to display)
# ---------------------------
@app.get("/analysis/next")
def next_job():
    try:
        job = supabase_helper.fetch_next_queued_job()
        return job or {}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# GET /analysis/{job_id}/results  -> fetch results rows for a given job
# ---------------------------
@app.get("/analysis/{job_id}/results")
def fetch_results(job_id: str):
    """
    Simple retrieval: returns results rows for a given job_id from analysis_results table.
    Uses supabase_helper.get_conn() to get a psycopg2 connection; adjust if your helper exposes a different API.
    """
    conn = None
    try:
        # attempt to get a connection via supabase helper (if present)
        conn = supabase_helper.get_conn()
    except Exception as e:
        print("Could not get DB connection from supabase_helper.get_conn():", e)
        traceback.print_exc()
        # fallback: raise an error so user knows
        raise HTTPException(status_code=500, detail="DB connection error")

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM analysis_results WHERE job_id = %s ORDER BY created_at DESC",
                (job_id,),
            )
            rows = cur.fetchall()
            # convert to serializable objects (RealDictCursor returns dicts -> JSON serializable)
            return {"results": rows}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            conn.close()
        except Exception:
            pass
