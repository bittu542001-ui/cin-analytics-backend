#!/usr/bin/env Rscript
# calibration_belt.R
# Input: { "y": [...], "p": [...] }
args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])
y <- inp$y
p <- inp$p

# create calibration data (grouped)
library(pROC)
df <- data.frame(y=y, p=p)
df$grp <- cut(df$p, breaks = seq(0,1,by=0.1), include.lowest=TRUE)
cal <- aggregate(cbind(mean_obs=y, mean_pred=p) ~ grp, data=df, FUN=mean)
res <- list(calibration = cal)
cat(toJSON(res, auto_unbox=TRUE))
