import pandas as pd
import scipy.stats as stats
import numpy as np
from tabulate import tabulate

if __name__ == '__main__':
    # %% Question 3
    # The outcome in Campo is the Drop Jump (DJ).
    def effect_size(means_pg, means_cg, std_pool):
        effect_size = (means_pg - means_cg) / std_pool
        return effect_size


    def pooled_standard_deviation(std_devs1: pd.Series, std_devs2: pd.Series, ):
        n1 = len(std_devs1)
        n2 = len(std_devs2)
        weighted_var1 = sum(((n1 - 1) * std ** 2) for std in std_devs1) / (n1 - 1)
        weighted_var2 = sum(((n2 - 1) * std ** 2) for std in std_devs2) / (n2 - 1)

        pooled_var = ((n1 - 1) * weighted_var1 + (n2 - 1) * weighted_var2) / (n1 + n2 - 2)
        pooled_std_dev = np.sqrt(pooled_var)
        return pooled_std_dev

    def pooled_std(n1, s1, n2, s2):
        num = (n1-1)*s1**2 + (n2-1)*s2**2
        den = n1+n2-2
        return np.sqrt(num/den)

    # measured data for the PG and the CG
    campo_drop_dj = {
        'Group': ['PG'] * 4 + ['CG'] * 4,
        'T_measure': [1, 2, 3, 4] * 2,
        'Mean': [24.9, 28.2, 28.9, 29.4, 27.1, 24.6, 25.6, 25.2],
        'Std': [1.1, 0.9, 0.8, 1.0, 1.0, 0.8, 0.9, 0.8]
    }
    campo_drop_dj_df = pd.DataFrame(campo_drop_dj)
    print(tabulate(campo_drop_dj_df, headers='keys', tablefmt='psql'))

    # Calculate mean change (difference within measures) for each group
    mean_change_pg = campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'PG',
                                                            'Mean'].diff().dropna().values
    mean_change_cg = campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'CG',
                                                            'Mean'].diff().dropna().values

    # Calculate pooled standard deviation using all the measures
    # std_pool = pooled_standard_deviation(std_devs1=campo_drop_dj_df.loc[
    #                                         campo_drop_dj_df['Group'] == 'PG', 'Std'],
    #                                      std_devs2=campo_drop_dj_df.loc[
    #                                          campo_drop_dj_df['Group'] == 'CG', 'Std']
    #                                      )
    # pooled standard deviation uses the standard deviation from the final measurement.
    std_pool = pooled_std(
        n1=10,
        s1=campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'PG', 'Std'].iloc[-1],
        n2=10,
        s2=campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'CG', 'Std'].iloc[-1],
    )

    # Calculate the effect size
    effect_size_value = effect_size(means_pg=np.mean(mean_change_pg),
                                    means_cg=np.mean(mean_change_cg),
                                    std_pool=std_pool)

    # confidence interval for the effect size
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha / 2, df=len(mean_change_pg) - 1)
    margin_of_error = t_critical * np.sqrt((std_pool ** 2) / len(mean_change_pg))

    confidence_interval = (effect_size_value - margin_of_error, effect_size_value + margin_of_error)
    confidence_interval = tuple(round(value, 3) for value in confidence_interval)

    print(f"Effect Size: {np.round(effect_size_value, 3)}; CI(95%) [{confidence_interval}]")

    #%% Simulate the error made by Campo
    def effect_size(means_pg, means_cg, std_pool):
        effect_size = (means_pg - means_cg) / std_pool
        return effect_size
    def pooled_std(n1, s1, n2, s2):
        num = (n1-1)*s1**2 + (n2-1)*s2**2
        den = n1+n2-2
        return np.sqrt(num/den)

    n1 = 10
    baseline_pg_measure_mean = 24.9
    baseline_pg_measure_std = 1.1
    baseline_pg_measure_se = baseline_pg_measure_std/np.sqrt(n1)
    last_pg_measure_mean = 29.4
    last_pg_measure_std = 1.0
    last_pg_measure_se = last_pg_measure_std/np.sqrt(n1)

    n2 = 10
    baseline_cg_measure_mean = 27.1
    baseline_cg_measure_std = 1.0
    baseline_cg_measure_se = baseline_cg_measure_std/np.sqrt(n2)
    last_cg_measure_mean = 25.2
    last_cg_measure_std = 0.8
    last_cg_measure_se = last_cg_measure_std/np.sqrt(n2)

    # poold from last observation in pg and cg group
    std_pooled = pooled_std(
        n1=n2,
        s1=last_pg_measure_std,
        n2=n2,
        s2=last_cg_measure_std,
    )
    # mean change is the change from baseline to final
    effect_size_value = effect_size(means_pg=np.diff([baseline_pg_measure_mean, last_pg_measure_mean]),
                                    means_cg=np.diff([baseline_cg_measure_mean, last_cg_measure_mean]),
                                    std_pool=std_pooled)
    print(effect_size_value)




    effect_size_value = effect_size(means_pg=np.mean(mean_change_pg),
                                    means_cg=np.mean(mean_change_cg),
                                    std_pool=std_pool)
