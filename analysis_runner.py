# analysis_runner.py
import os
import traceback
from datetime import datetime
import uuid
import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import KaplanMeierFitter, CoxPHFitter

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
            supabase_helper.insert_analysis_result(job_id, label, metrics={"error": True, "message": err})
            return {"success": False, "error": err}
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
# Small helpers for Python tests
def safe_insert(job_id, label, metrics=None, tables=None, plots=None):
    try:
        supabase_helper.insert_analysis_result(job_id=job_id, result_label=label, metrics=metrics or {}, tables=tables, plots=plots)
    except Exception as e:
        supabase_helper.append_log(job_id, f"Failed to insert result {label}: {e}", level="ERROR")

def try_run(job_id, label, fn, *args, **kwargs):
    try:
        res = fn(*args, **kwargs)
        safe_insert(job_id, label, metrics=res if isinstance(res, dict) else {"result": str(res)})
    except Exception as e:
        tb = traceback.format_exc()
        supabase_helper.append_log(job_id, f"{label} failed: {e}\n{tb}", level="ERROR")
        safe_insert(job_id, label, metrics={"error": True, "exception": str(e)})

# ------------------------- statistical test implementations -------------------------
def descriptive_stats(df, cols):
    out = {}
    for c in cols:
        s = df[c]
        out[c] = {
            "n": int(s.dropna().shape[0]),
            "missing": int(s.isna().sum())
        }
        if pd.api.types.is_numeric_dtype(s):
            out[c].update({
                "mean": float(s.dropna().mean()) if s.dropna().shape[0] else None,
                "median": float(s.dropna().median()) if s.dropna().shape[0] else None,
                "std": float(s.dropna().std(ddof=1)) if s.dropna().shape[0] else None,
                "min": float(s.dropna().min()) if s.dropna().shape[0] else None,
                "max": float(s.dropna().max()) if s.dropna().shape[0] else None,
                "iqr": float(s.dropna().quantile(0.75) - s.dropna().quantile(0.25)) if s.dropna().shape[0] else None
            })
        else:
            out[c].update({"freq": s.value_counts(dropna=False).to_dict()})
    return out

def shapiro_wilk(df, col):
    s = df[col].dropna()
    if len(s) < 3:
        return {"error": "insufficient_n"}
    stat, p = ss.shapiro(s)
    return {"statistic": float(stat), "pvalue": float(p), "n": int(len(s))}

def kolmogorov_smirnov_normal(df, col):
    s = df[col].dropna()
    if len(s) < 3:
        return {"error": "insufficient_n"}
    mu, sd = float(s.mean()), float(s.std(ddof=1))
    stat, p = ss.kstest(s, 'norm', args=(mu, sd))
    return {"statistic": float(stat), "pvalue": float(p), "mu": mu, "sd": sd}

def skew_kurtosis(df, col):
    s = df[col].dropna()
    if len(s) < 3: return {"error": "insufficient_n"}
    return {"skewness": float(ss.skew(s)), "kurtosis": float(ss.kurtosis(s))}

def qq_plot_and_save(df, col, job_id):
    s = df[col].dropna()
    if len(s) < 3: return {"error":"insufficient_n"}, None
    fig_path = plot_helper.save_qq_plot(col, s, job_id=job_id)  # assumes plot_helper has save_qq_plot(name, series, job_id)
    return {"n": int(len(s))}, fig_path

def levene_test(df, col, group_col):
    groups = [g.dropna() for _, g in df.groupby(group_col)[col]]
    if len(groups) < 2: return {"error": "need_>=2_groups"}
    stat, p = ss.levene(*groups)
    return {"statistic": float(stat), "pvalue": float(p)}

def bartlett_test(df, col, group_col):
    groups = [g.dropna() for _, g in df.groupby(group_col)[col]]
    if len(groups) < 2: return {"error": "need_>=2_groups"}
    stat, p = ss.bartlett(*groups)
    return {"statistic": float(stat), "pvalue": float(p)}

def t_test_independent(df, col, group_col):
    gdf = df[[col, group_col]].dropna()
    groups = gdf[group_col].unique()
    if len(groups) != 2: return {"error": "need_exactly_2_groups"}
    a = gdf[gdf[group_col]==groups[0]][col]
    b = gdf[gdf[group_col]==groups[1]][col]
    if len(a) < 2 or len(b) < 2: return {"error": "insufficient"}
    t, p = ss.ttest_ind(a, b, equal_var=False)
    return {"t": float(t), "p": float(p), "n_group0": int(len(a)), "n_group1": int(len(b))}

def t_test_paired(df, col1, col2):
    s = df[[col1, col2]].dropna()
    if len(s) < 3: return {"error": "insufficient_pairs"}
    t, p = ss.ttest_rel(s[col1], s[col2])
    return {"t": float(t), "p": float(p), "n": int(len(s))}

def wilcoxon_signed_rank(df, col1, col2):
    s = df[[col1, col2]].dropna()
    if len(s) < 3: return {"error": "insufficient_pairs"}
    stat, p = ss.wilcoxon(s[col1], s[col2])
    return {"statistic": float(stat), "pvalue": float(p), "n": int(len(s))}

def mann_whitney_u(df, col, group_col):
    gdf = df[[col, group_col]].dropna()
    groups = gdf[group_col].unique()
    if len(groups) != 2: return {"error": "need_exactly_2_groups"}
    a = gdf[gdf[group_col]==groups[0]][col]; b = gdf[gdf[group_col]==groups[1]][col]
    if len(a) < 1 or len(b) < 1: return {"error":"insufficient"}
    u, p = ss.mannwhitneyu(a, b, alternative='two-sided')
    return {"u": float(u), "p": float(p)}

def anova_oneway(df, col, group_col):
    groups = [g.dropna() for _, g in df.groupby(group_col)[col]]
    if len(groups) < 2: return {"error": "need_>=2_groups"}
    f, p = ss.f_oneway(*groups)
    # post-hoc Tukey if requested - return raw arrays so UI can run or call pairwise_tukeyhsd here
    return {"F": float(f), "p": float(p)}

def tukey_posthoc(df, col, group_col):
    sub = df[[col, group_col]].dropna()
    try:
        res = pairwise_tukeyhsd(sub[col], sub[group_col])
        # convert result table
        res_df = pd.DataFrame(data=res._results_table.data[1:], columns=res._results_table.data[0])
        return {"tukey": res_df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

def kruskal_wallis(df, col, group_col):
    groups = [g.dropna() for _, g in df.groupby(group_col)[col]]
    if len(groups) < 2: return {"error": "need_>=2_groups"}
    h, p = ss.kruskal(*groups)
    return {"H": float(h), "p": float(p)}

def pearson_corr(df, col1, col2):
    sub = df[[col1, col2]].dropna()
    if len(sub) < 3: return {"error": "insufficient"}
    r, p = ss.pearsonr(sub[col1], sub[col2])
    return {"r": float(r), "p": float(p), "n": int(len(sub))}

def spearman_corr(df, col1, col2):
    sub = df[[col1, col2]].dropna()
    if len(sub) < 3: return {"error": "insufficient"}
    rho, p = ss.spearmanr(sub[col1], sub[col2])
    return {"rho": float(rho), "p": float(p), "n": int(len(sub))}

def cohens_d(df, col, group_col):
    gdf = df[[col, group_col]].dropna()
    groups = gdf[group_col].unique()
    if len(groups) != 2: return {"error": "need_exactly_2_groups"}
    a = gdf[gdf[group_col]==groups[0]][col]; b = gdf[gdf[group_col]==groups[1]][col]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2: return {"error":"insufficient"}
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1-1)*s1*s1 + (n2-1)*s2*s2) / (n1+n2-2))
    d = (a.mean() - b.mean()) / pooled
    return {"cohens_d": float(d)}

def chi_square_test(df, col1, col2):
    tab = pd.crosstab(df[col1], df[col2])
    if tab.size == 0: return {"error": "empty_table"}
    chi, p, dof, exp = ss.chi2_contingency(tab)
    return {"chi2": float(chi), "p": float(p), "dof": int(dof), "expected": pd.DataFrame(exp, index=tab.index, columns=tab.columns).to_dict()}

def fisher_exact_test(df, col1, col2):
    tab = pd.crosstab(df[col1], df[col2])
    if tab.size != 4: return {"error": "need_2x2"}
    odds, p = ss.fisher_exact(tab.values)
    return {"odds_ratio": float(odds), "p": float(p)}

def multiple_testing_correction(pvals, method="fdr_bh"):
    rej, p_adj, _, _ = multipletests(pvals, method=method)
    return {"rejected": list(map(bool, rej)), "p_adjusted": p_adj.tolist(), "method": method}

def vif_check(df, covariates):
    X = df[covariates].dropna()
    Xc = sm.add_constant(X)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    out = {}
    for i, col in enumerate(Xc.columns):
        if col == 'const': continue
        try:
            out[col] = float(variance_inflation_factor(Xc.values, i))
        except Exception as e:
            out[col] = str(e)
    return out

def bootstrap_mean_ci(df, col, n_boot=500, alpha=0.05):
    arr = df[col].dropna().values
    if len(arr) < 2: return {"error":"insufficient"}
    boot_means = []
    for _ in range(n_boot):
        samp = np.random.choice(arr, size=len(arr), replace=True)
        boot_means.append(np.mean(samp))
    lower = np.percentile(boot_means, 100*(alpha/2))
    upper = np.percentile(boot_means, 100*(1-alpha/2))
    return {"boot_mean": float(np.mean(boot_means)), "ci": [float(lower), float(upper)], "n_boot": n_boot}

def kaplan_meier_plot(df, time_col, event_col, group_col=None, job_id=None):
    kmf = KaplanMeierFitter()
    fig_path = None
    try:
        # plot using plot_helper (expects DataFrame + columns)
        fig_path = plot_helper.save_km_plot(df, time_col, event_col, group_col, job_id=job_id)
        return {"plot": fig_path}
    except Exception as e:
        return {"error": str(e)}

def cox_ph_summary(df, time_col, event_col, covariates):
    d = df[[time_col, event_col] + covariates].dropna()
    if d.shape[0] < len(covariates) + 5:
        return {"error": "insufficient"}
    cph = CoxPHFitter()
    cph.fit(d, duration_col=time_col, event_col=event_col)
    return {"summary": cph.summary.reset_index().to_dict(orient="records")}

# ------------------------- End of statistical functions -------------------------

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
            desc = descriptive_stats(df, selected_parameters)
            supabase_helper.insert_analysis_result(job_id, "descriptive_stats", metrics={"desc": desc, "n_rows": len(df)})
        except Exception as e:
            supabase_helper.append_log(job_id, f"Descriptive stats failed: {str(e)}", level="WARNING")

        # run a set of python-only tests if requested (or run_all)
        run_all = (not selected_tests or len(selected_tests) == 0)

        # Helper to decide if to run
        def should_run(test_name):
            return run_all or (test_name in selected_tests)

        # 1) Normality & diagnostics
        numeric_cols = [c for c in selected_parameters if pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            if should_run("shapiro_wilk"):
                try_run(job_id, f"shapiro_{col}", shapiro_wilk, df, col)
            if should_run("kolmogorov_smirnov"):
                try_run(job_id, f"ks_normal_{col}", kolmogorov_smirnov_normal, df, col)
            if should_run("skew_kurtosis"):
                try_run(job_id, f"skew_kurtosis_{col}", skew_kurtosis, df, col)
            if should_run("qqplot"):
                metrics, plot_path = qq_plot_and_save(df, col, job_id)
                safe_insert(job_id, f"qqplot_{col}", metrics=metrics, plots={"qq": plot_path} if plot_path else None)

        # 2) Group comparisons for a selected group_col (if provided)
        group_col = options.get("group_col")
        if group_col and group_col in df.columns:
            for col in numeric_cols:
                if should_run("levene_test"):
                    try_run(job_id, f"levene_{col}_by_{group_col}", levene_test, df, col, group_col)
                if should_run("bartlett_test"):
                    try_run(job_id, f"bartlett_{col}_by_{group_col}", bartlett_test, df, col, group_col)
                if should_run("t_test_independent"):
                    try_run(job_id, f"t_test_{col}_by_{group_col}", t_test_independent, df, col, group_col)
                if should_run("mann_whitney_u"):
                    try_run(job_id, f"mannwhitney_{col}_by_{group_col}", mann_whitney_u, df, col, group_col)
            # categorical comparisons
            cat_cols = [c for c in selected_parameters if not pd.api.types.is_numeric_dtype(df[c])]
            for ccol in cat_cols:
                for ccol2 in cat_cols:
                    if ccol == ccol2: continue
                    if should_run("chi_square"):
                        try_run(job_id, f"chi2_{ccol}_vs_{ccol2}", chi_square_test, df, ccol, ccol2)
                    if should_run("fisher_exact"):
                        try_run(job_id, f"fisher_{ccol}_vs_{ccol2}", fisher_exact_test, df, ccol, ccol2)

        # 3) Pairwise / paired tests if explicitly requested
        if should_run("paired_t_test") and options.get("paired_cols"):
            for pair in options.get("paired_cols", []):
                c1, c2 = pair[0], pair[1]
                try_run(job_id, f"paired_t_{c1}_vs_{c2}", t_test_paired, df, c1, c2)

        if should_run("wilcoxon_signed_rank") and options.get("paired_cols"):
            for pair in options.get("paired_cols", []):
                c1, c2 = pair[0], pair[1]
                try_run(job_id, f"wilcoxon_{c1}_vs_{c2}", wilcoxon_signed_rank, df, c1, c2)

        # 4) ANOVA / Kruskal
        if group_col and should_run("anova"):
            for col in numeric_cols:
                try_run(job_id, f"anova_{col}_by_{group_col}", anova_oneway, df, col, group_col)
                if should_run("anova_posthoc_tukey"):
                    try_run(job_id, f"tukey_{col}_by_{group_col}", tukey_posthoc, df, col, group_col)
        if group_col and should_run("kruskal"):
            for col in numeric_cols:
                try_run(job_id, f"kruskal_{col}_by_{group_col}", kruskal_wallis, df, col, group_col)

        # 5) Correlations
        if should_run("pearson") and len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    try_run(job_id, f"pearson_{numeric_cols[i]}_{numeric_cols[j]}", pearson_corr, df, numeric_cols[i], numeric_cols[j])
        if should_run("spearman") and len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    try_run(job_id, f"spearman_{numeric_cols[i]}_{numeric_cols[j]}", spearman_corr, df, numeric_cols[i], numeric_cols[j])

        # 6) Effect sizes & multiple testing
        if group_col and should_run("cohens_d"):
            for col in numeric_cols:
                try_run(job_id, f"cohens_d_{col}_by_{group_col}", cohens_d, df, col, group_col)

        # multiple testing: if pvals provided in options or collect p-values from previous results (simple behavior)
        if should_run("multiple_testing"):
            pvals = options.get("pvals") or []
            if pvals:
                try_run(job_id, "multiple_testing_adjust", multiple_testing_correction, pvals, options.get("p_adjust_method", "fdr_bh"))
            else:
                supabase_helper.append_log(job_id, "No p-values provided for multiple testing", level="INFO")

        # 7) VIF / multicollinearity
        if should_run("vif") or should_run("multicollinearity"):
            covs = options.get("covariates") or [c for c in selected_parameters if c != outcome_col and pd.api.types.is_numeric_dtype(df[c])]
            if covs:
                try_run(job_id, "vif_check", vif_check, df, covs)

        # 8) Bootstrap example (mean CI) if requested
        if should_run("bootstrap") and options.get("bootstrap_cols"):
            for col in options.get("bootstrap_cols", []):
                try_run(job_id, f"bootstrap_{col}", bootstrap_mean_ci, df, col, options.get("bootstrap_iters", 500), options.get("alpha", 0.05))

        # 9) Survival analysis if requested
        if should_run("kaplan_meier") and options.get("time_col") and options.get("event_col"):
            try_run(job_id, "kaplan_meier", kaplan_meier_plot, df, options.get("time_col"), options.get("event_col"), options.get("group_col"), job_id)
        if should_run("cox_ph") and options.get("time_col") and options.get("event_col"):
            covs = options.get("covariates") or [c for c in selected_parameters if c not in (options.get("time_col"), options.get("event_col"))][:6]
            try_run(job_id, "cox_ph", cox_ph_summary, df, options.get("time_col"), options.get("event_col"), covs)

        # ---------------- Python logistic regression (if requested) --------------------
        # numeric predictors for modeling
        Xcols = [c for c in selected_parameters if pd.api.types.is_numeric_dtype(df[c])]
        if outcome_col not in df.columns:
            raise ValueError(f"Outcome column {outcome_col} not in dataset")

        usecols = Xcols + [outcome_col]
        sub = df[usecols].dropna()
        supabase_helper.append_log(job_id, f"Rows with complete data for modeling: {len(sub)}")

        if len(sub) < 10:
            supabase_helper.append_log(job_id, f"Too few complete rows ({len(sub)}), skipping modeling", level="WARNING")
            supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
            return {"status": "completed", "reason": "too few rows for modeling"}

        X = sub[Xcols].values
        y = sub[outcome_col].astype(int).values

        if should_run("logistic_regression"):
            supabase_helper.append_log(job_id, "Fitting logistic regression")
            model = LogisticRegression(max_iter=1000)
            model.fit(X, y)
            probs = model.predict_proba(X)[:, 1]
            auc = roc_auc_score(y, probs)

            # save python ROC plot
            try:
                plot_path, auc_val = plot_helper.save_roc_plot(y, probs)  # must return (path, auc)
            except Exception:
                plot_path = None
            supabase_helper.append_log(job_id, f"ROC saved to {plot_path}")

            metrics = {"n_total": int(len(df)), "n_model": int(len(sub)), "auc": float(auc)}
            coef_table = [{"feature": Xcols[i], "coef": float(model.coef_[0][i])} for i in range(len(Xcols))]

            supabase_helper.insert_analysis_result(
                job_id=job_id,
                result_label="logistic_regression",
                metrics=metrics,
                tables={"coefficients": coef_table},
                plots={"roc": plot_path} if plot_path else None
            )
        else:
            model = None
            probs = None

        # ----------------- Prepare common arrays for R scripts -------------------------
        if probs is None:
            prob_col = options.get("prob_col")
            if prob_col and prob_col in df.columns:
                probs_full = df[prob_col].astype(float).fillna(0).values
                # align with sub by index if possible
                try:
                    probs = list(df.loc[sub.index, prob_col].astype(float).fillna(0).values)
                except Exception:
                    probs = list(probs_full[:len(sub)])
            else:
                probs = [0.0] * len(sub)

        try:
            y_for_r = list(sub[outcome_col].astype(int).values)
            p_for_r = list(probs if len(probs) == len(sub) else (model.predict_proba(sub[Xcols].values)[:, 1].tolist() if model is not None else [0]*len(sub)))
        except Exception:
            y_for_r = list(sub[outcome_col].astype(int).values)
            p_for_r = [0] * len(sub)

        # ------------------ Call R scripts in sequence (if requested) -----------------
        # 1) Hosmer-Lemeshow
        if run_all or "hosmer_lemeshow" in selected_tests or "hosmer" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r, "g": int(options.get("hosmer_groups", 10))}
            call_r_and_store(job_id, "hosmer_lemeshow", "r_scripts/hosmer_lemeshow.R", payload)

        # 2) Decision Curve Analysis (DCA)
        if run_all or "dca" in selected_tests or "decision_curve_analysis" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r}
            call_r_and_store(job_id, "decision_curve", "r_scripts/dca.R", payload)

        # 3) Calibration belt / calibration summary
        if run_all or "calibration_belt" in selected_tests or "calibration" in selected_tests or "calibration_plot" in selected_tests:
            payload = {"y": y_for_r, "p": p_for_r}
            call_r_and_store(job_id, "calibration_belt", "r_scripts/calibration_belt.R", payload)

        # 4) NRI / IDI (if user supplied p_old and p_new in options)
        if run_all or "nri_idi" in selected_tests or "net_reclassification_index_nri" in selected_tests:
            p_old = options.get("p_old")  # arrays expected if present
            p_new = options.get("p_new")
            if p_old and p_new:
                payload = {"y": y_for_r, "p_old": p_old, "p_new": p_new}
                call_r_and_store(job_id, "nri_idi", "r_scripts/nri_idi.R", payload)
            else:
                supabase_helper.append_log(job_id, "Skipping NRI/IDI: p_old or p_new not provided in job options", level="INFO")

        # 5) Little's MCAR (missingness)
        if run_all or "little_mcar" in selected_tests:
            max_rows = int(options.get("mcar_max_rows", 500))
            small_df = df[selected_parameters].head(max_rows)
            payload = {"data": small_df.where(pd.notnull(small_df), None).to_dict(orient="records")}
            call_r_and_store(job_id, "little_mcar", "r_scripts/little_mcar.R", payload)

        # 6) MICE imputation (if requested)
        if run_all or "mice_impute" in selected_tests or "mice_multiple_imputation" in selected_tests:
            max_rows = int(options.get("mice_max_rows", 200))
            small_df = df[selected_parameters].head(max_rows)
            payload = {"data": small_df.where(pd.notnull(small_df), None).to_dict(orient="records"), "m": int(options.get("mice_m", 5))}
            call_r_and_store(job_id, "mice_impute", "r_scripts/mice_impute.R", payload)

        # 7) Time-dependent ROC (placeholder)
        if run_all or "time_dependent_roc" in selected_tests:
            payload = {"data": sub.to_dict(orient="records")}
            call_r_and_store(job_id, "time_dependent_roc", "r_scripts/time_dependent_roc.R", payload)

        # 8) Propensity score procedures
        if run_all or "propensity_score" in selected_tests or "propensity_score_matching" in selected_tests:
            payload = {"data": df[selected_parameters].where(pd.notnull(df[selected_parameters]), None).to_dict(orient="records"), "treatment_col": options.get("treatment_col")}
            call_r_and_store(job_id, "propensity_score", "r_scripts/propensity_score.R", payload)

        # 9) Subgroup analysis (R template)
        if run_all or "subgroup_analysis" in selected_tests:
            payload = {"data": df[selected_parameters].where(pd.notnull(df[selected_parameters]), None).to_dict(orient="records"), "subgroup_by": options.get("subgroup_by", [])}
            call_r_and_store(job_id, "subgroup_analysis", "r_scripts/subgroup_analysis.R", payload)

        # --------------------------------- Finish ------------------------------------
        supabase_helper.update_job_status(job_id, "completed", progress=100, finished_at=datetime.utcnow())
        supabase_helper.append_log(job_id, "Job completed", level="INFO")
        return {"job_id": job_id, "status": "completed"}

    except Exception as e:
        tb = traceback.format_exc()
        supabase_helper.append_log(job_id, f"Job failed: {str(e)}\n{tb}", "ERROR")
        supabase_helper.update_job_status(job_id, "failed", progress=100, error_message=str(e), finished_at=datetime.utcnow())
        return {"job_id": job_id, "error": str(e)}
