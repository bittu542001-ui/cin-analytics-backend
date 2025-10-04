# analysis_runner.py
import os
import time
import traceback
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from utils import supabase_helper, plot_helper, r_interface

def run_one_job():
    job = supabase_helper.fetch_next_queued_job()
    if not job:
        print("No queued jobs.")
        return None

    job_id = job["job_id"]
    supabase_helper.append_log(job_id, "Picked up job")
    supabase_helper.update_job_status(job_id, "running", progress=1, started_at=datetime.utcnow())

    try:
        # parse job payload
        dataset_table = job.get("dataset_table", "patients")
        dataset_filter = job.get("dataset_filter")  # optional JSON filter; if string use directly
        selected_parameters = job.get("selected_parameters") or []
        selected_tests = job.get("selected_tests") or []
        options = job.get("options") or {}

        # Simple where clause support: if dataset_filter is JSON with 'where' key
        where_clause = None
        if isinstance(dataset_filter, dict) and dataset_filter.get("where"):
            where_clause = dataset_filter.get("where")
        elif isinstance(dataset_filter, str) and dataset_filter.strip():
            where_clause = dataset_filter

        # load data
        supabase_helper.append_log(job_id, f"Loading dataset {dataset_table}")
        df = supabase_helper.load_dataset(table=dataset_table, where_clause=where_clause)
        supabase_helper.append_log(job_id, f"Loaded {len(df)} rows")

        if len(df) == 0:
            raise ValueError("Dataset empty for job")

        # default: if no params selected, take common predictors and outcome
        if not selected_parameters:
            # exclude meta columns heuristically
            exclude = {"patient_id","created_at","updated_at","job_id"}
            selected_parameters = [c for c in df.columns if c not in exclude]
        # assume user specifies outcome in options or default 'meets_cin_esur' or 'aki_outcome'
        outcome_col = options.get("outcome_col") or ("meets_cin_esur" if "meets_cin_esur" in df.columns else ("aki_outcome" if "aki_outcome" in df.columns else None))
        if not outcome_col:
            raise ValueError("No outcome column detected; set options.outcome_col")

        # simple descriptive stats
        desc = df[selected_parameters].describe(include='all').to_dict()

        # select numeric predictors for logistic regression (drop NA)
        Xcols = [c for c in selected_parameters if pd.api.types.is_numeric_dtype(df[c])]
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column {outcome_col} not in dataset")
        # drop rows with NA in Xcols or outcome
        usecols = Xcols + [outcome_col]
        sub = df[usecols].dropna()
        if len(sub) < 10:
            supabase_helper.append_log(job_id, f"Too few complete rows ({len(sub)}), skipping modeling", "WARNING")
            metrics = {"n": len(df)}
            supabase_helper.insert_analysis_result(job_id, "descriptive_only", metrics=metrics, tables={"desc": desc})
            supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
            return {"status":"completed","reason":"too few rows for modeling"}

        X = sub[Xcols].values
        y = sub[outcome_col].astype(int).values

        # simple logistic regression (L2)
        supabase_helper.append_log(job_id, "Fitting logistic regression")
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        probs = model.predict_proba(X)[:,1]
        auc = roc_auc_score(y, probs)

        # save ROC plot
        plot_path, auc_val = plot_helper.save_roc_plot(y, probs)
        supabase_helper.append_log(job_id, f"ROC saved to {plot_path}")

        # build metrics & table
        metrics = {"n_total": len(df), "n_model": len(sub), "auc": float(auc)}
        coef_table = [{"feature": Xcols[i], "coef": float(model.coef_[0][i])} for i in range(len(Xcols))]
        # save result
        res = supabase_helper.insert_analysis_result(
            job_id=job_id,
            result_label="logistic_regression",
            metrics=metrics,
            tables={"coefficients": coef_table},
            plots={"roc": plot_path}
        )

        supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
        supabase_helper.append_log(job_id, "Job completed", "INFO")
        return {"job_id": job_id, "result": res}
    except Exception as e:
        tb = traceback.format_exc()
        supabase_helper.append_log(job_id, f"Job failed: {str(e)}\n{tb}", "ERROR")
        supabase_helper.update_job_status(job_id, "failed", progress=100, error_message=str(e), finished_at=datetime.utcnow())
        return {"job_id": job_id, "error": str(e)}
