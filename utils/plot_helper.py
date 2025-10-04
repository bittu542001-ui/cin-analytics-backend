# utils/plot_helper.py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

def save_roc_plot(y_true, y_score, fname=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0,1],[0,1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if not fname:
        fname = f"/tmp/roc_{os.getpid()}.png"
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    return fname, roc_auc
