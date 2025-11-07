from statsmodels.stats.proportion import proportion_confint

ci_low, ci_high = proportion_confint(count=0, nobs=150, alpha=0.05, method="beta")
print(ci_low, ci_high)
