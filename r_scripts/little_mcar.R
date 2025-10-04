#!/usr/bin/env Rscript
# little_mcar.R
# Input JSON: { "data": [ {col1: val, col2: val}, ... ] }
# Output: result JSON (Little's MCAR not native — we will use a simple placeholder using mice::md.pattern)

args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
library(mice)

if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])

df <- as.data.frame(inp$data)
# return missingness pattern summary
mp <- tryCatch({
  pat <- md.pattern(df, plot = FALSE)
  list(success=TRUE, md_pattern=pat)
}, error=function(e){ list(success=FALSE, error=as.character(e)) })

cat(toJSON(mp, auto_unbox=TRUE))
