import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from typing import Union, Optional, Tuple
from numpy import ndarray
import random
from tabulate import tabulate
import numpy as np

if __name__ == '__main__':
    # %% Question 5 - violations of the homogeneity of variances assumption
    # compare the cholesterol levels of two groups with two-sample T-Test with a pooled variance
    def randomize_trial(n_distr_a: int, n_distr_b: int, std_dev_a: Union[int, float], std_dev_b: Union[int, float],
                        mean_dist_a: Union[int, float], mean_dist_b: Union[int, float],
                        seed: Optional[int] = 25, distribution: Optional[str] = 'normal') -> Union[ndarray, ndarray]:
        """
        if mean_cases == mean_controls we are assuming H0 is True. In both groups the true mean is the same
        :param num_controls:
        :param num_cases:
        :param std_dev:
        :param mean_cases:
        :param mean_controls:
        :param seed:
        :param distribution: type of distribution
        :return:
        """
        if distribution == 'normal':
            measure_cases = np.random.normal(loc=mean_dist_a, scale=std_dev_a, size=n_distr_a)
            measure_controls = np.random.normal(loc=mean_dist_b, scale=std_dev_b, size=n_distr_b)
        return measure_cases, measure_controls


    # Structure the scenarios
    scenarios = {
        'n_a': [50, 50, 50, 50, 50],
        'n_b': [50, 50, 100, 100, 100],
        'sd_a': [20, 10, 20, 10, 50],
        'sd_b': [20, 50, 20, 50, 10],
        'effect_size': np.nan,
    }
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df['pvalues'] = np.nan

    # T-Test with a pooled variance
    equal_variance = True
    # null_hypothesis else alternative
    SIMULATION_TYPE = 'null_hypothesis'
    sign_lv = 0.05
    n_trials = 1000

    if SIMULATION_TYPE == 'null_hypothesis':
        scenarios_df.effect_size = 0  # we assume Ho is true, no difference in means
        scenarios_df['Type_i_error'] = np.nan
        scenarios_df['ground_truth'] = 'Ho True'
    else:
        # assume artificial effect size between 10 and 40 (integers)
        scenarios_df.effect_size = [random.randint(10, 40) for _ in range(scenarios_df.n_a.shape[0])]
        scenarios_df['Type_ii_error'] = np.nan
        scenarios_df['ground_truth'] = 'H1 True'
        scenarios_df['power'] = np.nan

    scenarios_df['sign_lv'] = sign_lv

    # fig, axs = plt.subplots(nrows=scenarios_df.shape[0], ncols=1, figsize=(8, 16))

    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.5)
    # plt.suptitle("P-value distribution each scenario", fontsize=18, y=0.95)

    nrows = 3
    ncols = 2
    for idx_, scenario_ in scenarios_df.iterrows():
        # print(scenario_.n_a)
        p_values = []
        ax = plt.subplot(nrows, ncols, idx_ + 1)
        for _ in range(0, n_trials):
            distribution_a, distribution_b = randomize_trial(
                n_distr_a=scenario_.n_a,
                n_distr_b=scenario_.n_b,
                std_dev_a=scenario_.sd_a,
                std_dev_b=scenario_.sd_b,
                mean_dist_a=220,
                mean_dist_b=220 + scenario_.effect_size,
            )

            # compute the t-test
            p_value = stats.ttest_ind(
                a=distribution_a,
                b=distribution_b,
                alternative='two-sided',
                equal_var=equal_variance
            ).pvalue
            p_values.append(p_value)
        # scenario_.pvalues = str(p_values)
        scenarios_df.loc[idx_, 'pvalues'] = str(p_values)
        if SIMULATION_TYPE == 'null_hypothesis':
            # under H0 True
            # percent of times that we got a p-value bellow alpha -> significant when it should not
            scenarios_df.loc[idx_, 'Type_i_error'] = len([val for val in p_values if val < sign_lv]) / len(
                p_values)

        else:
            # under H1 True
            # percent of times that we got a p-value above alpha -> non-significant when it should
            scenarios_df.loc[idx_, 'Type_ii_error'] = len([val for val in p_values if val > sign_lv]) / len(p_values)
            scenarios_df.loc[idx_, 'power'] = (1 - scenario_.Type_ii_error) * 100

        # Plot histograms for p-value frequency in each scenario
        ax.hist(p_values, bins=len(np.unique(p_values)), color='skyblue', edgecolor='black')
        ax.set_xlim([0, max(p_values)])
        ax.set_title(f'Scenario {idx_ + 1}')
        ax.set_xlabel('P-values')
        ax.set_ylabel('Frequency')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    # Print the results of each scenario in a single table
    result = scenarios_df.drop(columns='pvalues')
    print(tabulate(result, headers='keys', tablefmt='psql'))