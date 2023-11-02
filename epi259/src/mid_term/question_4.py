# Total number of tests
total_tests = 150
# Probability of a significant result in a single test (under the null hypothesis)
p_sig = 0.01
# Calculate the probability of no significant correlations among all tests
prob_no_sig = (1 - p_sig) ** total_tests
# Calculate the p-value (probability of at least one significant correlation)
p_value = 1 - prob_no_sig
print("P-value:", p_value)

import numpy as np
def rr(p_ref:float, oddsratio:float) -> float:
    return np.round(oddsratio/((1-p_ref)+(p_ref*oddsratio)), 2)

rate_ratios = {}
for oddsratio in [1.5, 4.24, 4.50]:
    rate_ratios[oddsratio] = rr(p_ref=17.2/100, oddsratio=oddsratio)
