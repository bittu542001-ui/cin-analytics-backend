# utils/plot_helper.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats as ss

# ---------- helpers ----------
def _safe_tmp_path(prefix="plot", ext="png"):
    pid = os.getpid()
    return f"/tmp/{prefix}_{pid}.{ext}"

# ---------- ROC ----------
def save_roc_plot(y_true, y_score, fname=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve")
    plt.legend(loc="lower right")

    if not fname:
        fname = _safe_tmp_path(prefix="roc")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname, roc_auc

# ---------- QQ plot (normal Q-Q) ----------
def save_qq_plot(col_name, series, job_id=None, fname=None):
    s = series.dropna()
    if len(s) < 3:
        return None  # caller will handle

    plt.figure()
    ss.probplot(s, dist="norm", plot=plt)
    plt.title(f"QQ Plot - {col_name}")

    if not fname:
        fname = _safe_tmp_path(prefix=f"qq_{col_name}")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname

# ---------- Kaplan–Meier plot ----------
def save_km_plot(df, time_col, event_col, group_col=None, job_id=None, fname=None):
    from lifelines import KaplanMeierFitter

    plt.figure()
    kmf = KaplanMeierFitter()

    if group_col and group_col in df.columns:
        for grp, d in df[[time_col, event_col, group_col]].dropna().groupby(group_col):
            if len(d) == 0:
                continue
            kmf.fit(d[time_col], event_observed=d[event_col], label=str(grp))
            kmf.plot_survival_function(ci_show=True)
    else:
        d = df[[time_col, event_col]].dropna()
        if len(d) == 0:
            return None
        kmf.fit(d[time_col], event_observed=d[event_col], label="All")
        kmf.plot_survival_function(ci_show=True)

    plt.xlabel("Time")
    plt.ylabel("Survival probability")
    title = "Kaplan–Meier Survival"
    if group_col:
        title += f" by {group_col}"
    plt.title(title)

    if not fname:
        fname = _safe_tmp_path(prefix="km")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    return fname
