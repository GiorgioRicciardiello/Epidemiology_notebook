import numpy as np

def effect_size(means_pg, means_cg, std_pool):
    effect_size = (means_pg - means_cg) / std_pool
    return effect_size


def pooled_std(n1, s1, n2, s2):
    num = (n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2
    den = n1 + n2 - 2
    return np.sqrt(num / den)

# Get the baseline and last PG measures
n1 = 10
baseline_pg_measure_mean = 24.9
baseline_pg_measure_std = 1.1
last_pg_measure_mean = 29.4
last_pg_measure_std = 1.0


# Get the baseline and last CG measure
n2 = 10
baseline_cg_measure_mean = 27.1
baseline_cg_measure_std = 1.0
last_cg_measure_mean = 25.2
last_cg_measure_std = 0.8

# poold standard deviation from last observation in pg and cg group
std_pooled = pooled_std(
    n1=n2,
    s1=last_pg_measure_std,
    n2=n2,
    s2=last_cg_measure_std,
)
# mean change is the change from baseline to final
effect_size_value = effect_size(means_pg=np.diff([baseline_pg_measure_mean,
                                                  last_pg_measure_mean]),
                                means_cg=np.diff([baseline_cg_measure_mean,
                                                  last_cg_measure_mean]),
                                std_pool=std_pooled)
print(effect_size_value)