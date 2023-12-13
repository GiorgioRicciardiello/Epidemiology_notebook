import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

if __name__ == '__main__':
    #%% Question 1
    # correlated observations
    contingency_table = np.array([[11, 3], [14, 32]])
    # McNemar's Test with no continuity correction
    print(mcnemar(contingency_table, exact=False))

    # McNemar's Test with continuity correction
    print(mcnemar(contingency_table, exact=False, correction=False))