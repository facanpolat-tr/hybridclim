# Reference values for tests/test_rfa_reference.py, computed with Hosking's own R packages.
#
#   Rscript tests/data/make_lmomrfa_reference.R
#
# Data: the 'Cascades' annual precipitation region of Hosking & Wallis (1997), shipped with
# lmomRFA (19 sites; record length n and sample L-moment ratios t, t_3, t_4).

suppressPackageStartupMessages({library(lmomRFA); library(jsonlite)})
data(Cascades)
set.seed(20261122)
tst <- regtst(Cascades, nsim = 20000)
rm <- regavlmom(Cascades)                       # record-length weighted (1, t, t3, t4, t5)
kap <- pelkap(c(1, rm[2], rm[3], rm[4]))
gev <- pelgev(c(1, rm[2], rm[3]))
T <- c(2, 5, 10, 25, 50, 100, 500)
out <- list(
  source = sprintf("lmomRFA %s, lmom %s, R %s", packageVersion("lmomRFA"), packageVersion("lmom"),
                   getRversion()),
  sites = data.frame(name = as.character(Cascades$name), n = Cascades$n, t = Cascades$t,
                     t3 = Cascades$t_3, t4 = Cascades$t_4),
  regional = list(t = rm[[2]], t3 = rm[[3]], t4 = rm[[4]]),
  D = unname(tst$D), Dcrit = unname(tst$Dcrit[1]),
  H = unname(tst$H[1:2]), vobs = unname(tst$vobs[1:2]), vbar = unname(tst$vbar[1:2]),
  vsd = unname(tst$vsd[1:2]),
  Z = as.list(tst$Z), t4fit = as.list(tst$t4fit), nsim = 20000,
  kappa = as.list(kap), gev = as.list(gev),
  gev_growth = list(T = T, x_T = unname(quagev(1 - 1 / T, gev)))
)
writeLines(toJSON(out, auto_unbox = TRUE, digits = 10, pretty = TRUE),
           "tests/data/lmomrfa_cascades.json")
cat("written tests/data/lmomrfa_cascades.json\n")
