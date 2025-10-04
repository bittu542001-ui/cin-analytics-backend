#!/usr/bin/env Rscript
# nri_idi.R
# Input: { "y": [...], "p_old": [...], "p_new": [...] }
args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])
y <- inp$y
p_old <- inp$p_old
p_new <- inp$p_new

# compute simple IDI and NRI (category-free)
library(pROC)
# IDI = (mean new predicted for events - mean old predicted for events) - (mean new pred for non-events - mean old pred for non-events)
idi <- (mean(p_new[y==1]) - mean(p_old[y==1])) - (mean(p_new[y==0]) - mean(p_old[y==0]))
# NRI (continuous) = 2*(AUC_new - AUC_old) approx (simplistic)
auc_old <- auc(y, p_old)
auc_new <- auc(y, p_new)
nri_cont <- as.numeric(auc_new - auc_old)
res <- list(idi=idi, auc_old=as.numeric(auc_old), auc_new=as.numeric(auc_new), nri_approx=nri_cont)
cat(toJSON(res, auto_unbox=TRUE))
