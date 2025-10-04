#!/usr/bin/env Rscript
# hosmer_lemeshow.R
# Usage: Rscript hosmer_lemeshow.R /path/to/input.json
# Input JSON: { "y": [0,1,...], "p": [0.12, 0.9, ...], "g": 10 }
# Output JSON printed to stdout.

args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
library(ResourceSelection) # hoslem.test
if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])

y <- inp$y
p <- inp$p
g <- ifelse(is.null(inp$g), 10, inp$g)

res <- tryCatch({
  ht <- hoslem.test(y, p, g = g)
  list(success=TRUE,
       statistic = ht$statistic,
       df = ht$parameter,
       p_value = ht$p.value,
       details = list(observed = ht$observed, expected = ht$expected))
}, error=function(e) {
  list(success=FALSE, error=as.character(e))
})

cat(toJSON(res, auto_unbox=TRUE))
