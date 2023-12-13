# Question 1
# correlated observations
contingency_table <- matrix(c(11, 3, 14, 32), nrow = 2, byrow = TRUE)

# McNemar's Test with no continuity correction
result_no_correction <- mcnemar.test(contingency_table, correct = FALSE)
print(result_no_correction)

# McNemar's Test with continuity correction
result_with_correction <- mcnemar.test(contingency_table, correct = TRUE)
print(result_with_correction)
