# analysis_runner.py
import os
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# local helpers - make sure these exist in utils/
from utils import supabase_helper, plot_helper, r_interface

# -------------------------------------------------------------------------------------------------
# Helper: call an R script and store result (centralized)
def call_r_and_store(job_id, label, script_path, payload):
    """
    Calls an R script using utils.r_interface.run_r_script and stores result into analysis_results.
    Returns dict with keys: success(bool), result(dict) or error(str).
    """
    try:
        supabase_helper.append_log(job_id, f"Calling R script {os.path.basename(script_path)}", level="INFO")
        r_out = r_interface.run_r_script(script_path, payload)
        if r_out.get("error"):
            err = r_out.get("stderr") or r_out.get("stdout") or "Rscript returned non-zero"
            supabase_helper.append_log(job_id, f"R script {script_path} failed: {err}", level="ERROR")
            # store a failed result row so UI can see attempt
            supabase_helper.insert_analysis_result(job_id, label, metrics={"error": True, "message": err})
            return {"success": False, "error": err}
        # successful: r_out["result"] contains parsed JSON (or raw_stdout)
        res = r_out.get("result", {"raw_stdout": r_out.get("stdout")})
        supabase_helper.insert_analysis_result(job_id, label, metrics=res)
        supabase_helper.append_log(job_id, f"R script {script_path} succeeded", level="INFO")
        return {"success": True, "result": res}
    except Exception as e:
        tb = traceback.format_exc()
        supabase_helper.append_log(job_id, f"Exception running R script {script_path}: {str(e)}\n{tb}", level="ERROR")
        supabase_helper.insert_analysis_result(job_id, label, metrics={"error": True, "exception": str(e)})
        return {"success": False, "error": str(e)}

# -------------------------------------------------------------------------------------------------
# Main runner: pick one queued job and run analyses (python + R)
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
        dataset_filter = job.get("dataset_filter")  # optional SQL WHERE string or dict with "where"
        selected_parameters = job.get("selected_parameters") or []
        selected_tests = job.get("selected_tests") or []   # e.g. ["logistic_regression","hosmer","dca",...]
        options = job.get("options") or {}

        # determine where clause
        where_clause = None
        if isinstance(dataset_filter, dict) and dataset_filter.get("where"):
            where_clause = dataset_filter.get("where")
        elif isinstance(dataset_filter, str) and dataset_filter.strip():
            where_clause = dataset_filter

        supabase_helper.append_log(job_id, f"Loading dataset {dataset_table}")
        df = supabase_helper.load_dataset(table=dataset_table, where_clause=where_clause)
        supabase_helper.append_log(job_id, f"Loaded {len(df)} rows")

        if len(df) == 0:
            raise ValueError("Dataset empty for job")

        # default selected_parameters: exclude meta columns
        if not selected_parameters:
            meta = {"patient_id", "created_at", "updated_at", "job_id"}
            selected_parameters = [c for c in df.columns if c not in meta]

        # decide outcome column: either provided in options or common defaults
        outcome_col = options.get("outcome_col") or ("meets_cin_esur" if "meets_cin_esur" in df.columns else ("aki_outcome" if "aki_outcome" in df.columns else None))
        if not outcome_col:
            raise ValueError("No outcome column detected; set options.outcome_col in the job.")

        # Save descriptive stats always
        try:
            desc = df[selected_parameters].describe(include='all').to_dict()
            supabase_helper.insert_analysis_result(job_id, "descriptive_stats", metrics={"desc": desc, "n_rows": len(df)})
        except Exception as e:
            supabase_helper.append_log(job_id, f"Descriptive stats failed: {str(e)}", level="WARNING")

        # numeric predictors for modeling
        Xcols = [c for c in selected_parameters if pd.api.types.is_numeric_dtype(df[c])]
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column {outcome_col} not in dataset")

        # drop rows with NA in predictors or outcome
        usecols = Xcols + [outcome_col]
        sub = df[usecols].dropna()
        supabase_helper.append_log(job_id, f"Rows with complete data for modeling: {len(sub)}")

        if len(sub) < 10:
            # too few rows - store message and finish with descriptive-only
            supabase_helper.append_log(job_id, f"Too few complete rows ({len(sub)}), skipping modeling", level="WARNING")
            supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
            return {"status": "completed", "reason": "too few rows for modeling"}

        X = sub[Xcols].values
        y = sub[outcome_col].astype(int).values

        # ------------ Python logistic regression (always run if requested) --------------
        if (not selected_tests) or ("logistic_regression" in selected_tests):
            supabase_helper.append_log(job_id, "Fitting logistic regression")
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, probs)

            # save python ROC plot
            plot_path, auc_val = plot_helper.save_roc_plot(y, probs)
            supabase_helper.append_log(job_id, f"ROC saved to {plot_path}")

            metrics = {"n_total": len(df), "n_model": len(sub), "auc": float(auc)}
            coef_table = [{"feature": Xcols[i], "coef": float(model.coef_[0][i])} for i in range(len(Xcols))]

            supabase_helper.insert_analysis_result(
                job_id=job_id,
                result_label="logistic_regression",
                metrics=metrics,
                tables={"coefficients": coef_table},
                plots={"roc": plot_path}
            )
        else:
            # if logistic not requested, still compute probs if needed for R scripts
            model = None
            probs = None

        # ----------------- Prepare common arrays for R scripts -------------------------
        # Provide y and p (probabilities) arrays for R calls. If model wasn't run,
        # try to use existing probability column name from job options (options.prob_col)
        if 'probs' not in locals() or probs is None:
            # try to find a predicted probability column in df
            prob_col = options.get("prob_col")
            if prob_col and prob_col in df.columns:
                probs = df[prob_col].astype(float).fillna(0).values
            else:
                # fallback: if model is None and no probs, create basic score (mean)
                probs = np.repeat(0.0, len(sub))

        # common payload for R
        # use only the rows used for modeling (sub) to align y & p lengths
        try:
            y_for_r = list(sub[outcome_col].astype(int).values)
            p_for_r = list((probs if len(probs) == len(sub) else (model.predict_proba(sub[Xcols].values)[:,1] if model is not None else [0]*len(sub))))
        except Exception:
            # fallback: build from sub and try again
            y_for_r = list(sub[outcome_col].astype(int).values)
            p_for_r = [0]*len(sub)

        # ------------------ Call R scripts in sequence (if requested) -----------------
        # we call each R script only if user requested it in selected_tests OR if selected_tests empty (means run all)
        run_all = (not selected_tests or len(selected_tests) == 0)

        # 1) Hosmer-Lemeshow
        if run_all or "hosmer_lemeshow" in selected_tests or "hosmer" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r, "g": int(options.get("hosmer_groups", 10))}
            call_r_and_store(job_id, "hosmer_lemeshow", "r_scripts/hosmer_lemeshow.R", payload)

        # 2) Decision Curve Analysis (DCA)
        if run_all or "dca" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r}
            call_r_and_store(job_id, "decision_curve", "r_scripts/dca.R", payload)

        # 3) Calibration belt / calibration summary
        if run_all or "calibration_belt" in selected_tests or "calibration" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r}
            call_r_and_store(job_id, "calibration_belt", "r_scripts/calibration_belt.R", payload)

        # 4) NRI / IDI (if user supplied p_old and p_new in options)
        if run_all or "nri_idi" in selected_tests:
            p_old = options.get("p_old")  # arrays expected if present
            p_new = options.get("p_new")
            if p_old and p_new:
                payload = {"y": y_for_r, "p_old": p_old, "p_new": p_new}
                call_r_and_store(job_id, "nri_idi", "r_scripts/nri_idi.R", payload)
            else:
                supabase_helper.append_log(job_id, "Skipping NRI/IDI: p_old or p_new not provided in job options", level="INFO")

        # 5) Little's MCAR (missingness) - pass a small sample of the dataset or full dataset (be careful with size)
        if run_all or "little_mcar" in selected_tests:
            # prepare a small JSON-able dataset: convert first 500 rows of the selected columns to dicts
            max_rows = int(options.get("mcar_max_rows", 500))
            small_df = df[selected_parameters].head(max_rows)
            payload = {"data": small_df.where(pd.notnull(small_df), None).to_dict(orient="records")}
            call_r_and_store(job_id, "little_mcar", "r_scripts/little_mcar.R", payload)

        # 6) MICE imputation (if requested)
        if run_all or "mice_impute" in selected_tests:
            max_rows = int(options.get("mice_max_rows", 200))
            small_df = df[selected_parameters].head(max_rows)
            payload = {"data": small_df.where(pd.notnull(small_df), None).to_dict(orient="records"), "m": int(options.get("mice_m", 5))}
            call_r_and_store(job_id, "mice_impute", "r_scripts/mice_impute.R", payload)

        # --------------------------------- Finish ------------------------------------
        supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
        supabase_helper.append_log(job_id, "Job completed", level="INFO")
        return {"job_id": job_id, "status": "completed"}

    except Exception as e:
        tb = traceback.format_exc()
        supabase_helper.append_log(job_id, f"Job failed: {str(e)}\n{tb}", "ERROR")
        supabase_helper.update_job_status(job_id, "failed", progress=100, error_message=str(e), finished_at=datetime.utcnow())
        return {"job_id": job_id, "error": str(e)}
