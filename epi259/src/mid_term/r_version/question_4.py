# Total number of tests
total_tests = 150
# Probability of a significant result in a single test (under the null hypothesis)
p_sig = 0.01
# Calculate the probability of no significant correlations among all tests
prob_no_sig = (1 - p_sig) ** total_tests
# Calculate the p-value (probability of at least one significant correlation)
p_value = 1 - prob_no_sig
print("P-value:", p_value)