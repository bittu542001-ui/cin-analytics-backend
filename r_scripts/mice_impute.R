#!/usr/bin/env Rscript
# mice_impute.R
# Input: { "data": [ {col1: val, col2: val}, ... ], "m": 5 }
# Output: { success: TRUE, completed: TRUE }
args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
library(mice)

if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])
df <- as.data.frame(inp$data)
m <- ifelse(is.null(inp$m), 5, inp$m)

res <- tryCatch({
  imp <- mice(df, m = m, maxit = 5, printFlag = FALSE)
  completed <- complete(imp, 1)
  list(success=TRUE, completed = completed)
}, error=function(e) {
  list(success=FALSE, error=as.character(e))
})

cat(toJSON(res, auto_unbox=TRUE))
