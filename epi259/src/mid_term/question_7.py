"""
    Author: Giorgio Ricciardiello
    Code question 6
Simulation of an experiment with Binomial distribution with n=24 and p=0.5
"""

from scipy.stats import binom
n = 8  # Total number of games
k = 8  # Observed number of correct predictions
p = 0.5  # Probability of correct prediction by chance

# Calculate the two-sided p-value
p_value = binom.cdf(k, n, p) + (1 - binom.cdf(k - 1, n, p))
print("Two-sided p-value:", p_value)