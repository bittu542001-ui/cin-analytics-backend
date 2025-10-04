#!/usr/bin/env Rscript
# dca.R
# Input JSON: { "y": [0,1,...], "p": [0.12, 0.9, ...] }
# Output: basic net benefit at thresholds (placeholder)

args <- commandArgs(trailingOnly = TRUE)
library(jsonlite)
if (length(args) < 1) stop("Missing input json path")
inp <- fromJSON(args[1])

y <- inp$y
p <- inp$p
# thresholds 0.01..0.99 step 0.01
th <- seq(0.01,0.99,by=0.01)
nb <- sapply(th, function(t) {
  predicted <- as.numeric(p >= t)
  tp <- sum(predicted==1 & y==1)
  fp <- sum(predicted==1 & y==0)
  n <- length(y)
  pb <- t/(1-t)
  nb_val <- (tp/n) - (fp/n)*pb
  nb_val
})
res <- list(thresholds=th, net_benefit=nb)
cat(toJSON(res, digits=6, auto_unbox=TRUE))
