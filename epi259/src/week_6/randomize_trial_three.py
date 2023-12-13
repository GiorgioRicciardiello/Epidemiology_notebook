from scipy.stats import t
import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from numpy import ndarray
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pyplot as plt


# def randomize_trial(num_controls: int, num_cases: int, std_dev: Union[int, float],
#                     mean_cases: Union[int, float], mean_controls: Union[int, float],
#                     seed_cases: Optional[int] = 25, seed_controls: Optional[int] = 30,
#                     distribution: Optional[str] = 'normal') -> Union[np.ndarray, np.ndarray]:
#     """
#     Generate random data for control and case groups with different seeds.
#
#     :param num_controls: Number of data points for the control group
#     :param num_cases: Number of data points for the case group
#     :param std_dev: Standard deviation for the distributions
#     :param mean_cases: Mean for the case group
#     :param mean_controls: Mean for the control group
#     :param seed_cases: Seed for the case group distribution
#     :param seed_controls: Seed for the control group distribution
#     :param distribution: Type of distribution
#     :return: Tuple containing data arrays for the case and control groups
#     """
#     if distribution == 'normal':
#         np.random.seed(seed_cases)
#         measure_cases = np.random.normal(loc=mean_cases, scale=std_dev, size=num_cases)
#         np.random.seed(seed_controls)
#         measure_controls = np.random.normal(loc=mean_controls, scale=std_dev, size=num_controls)
#         return measure_cases, measure_controls
#     else:
#         raise ValueError("Distribution type not supported")


def randomize_trial(num_controls:int, num_cases:int, std_dev:Union[int, float],
                    mean_cases:Union[int, float], mean_controls:Union[int, float],
                    seed:Optional[int]=25, distribution:Optional[str]='normal') -> Union[ndarray, ndarray]:
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
        measure_cases = np.random.normal(loc=mean_cases, scale=std_dev, size=num_controls)
        measure_controls = np.random.normal(loc=mean_controls, scale=std_dev, size=num_cases)
    return measure_cases, measure_controls


if __name__ == "__main__":
    # %% input data
    scenarios = {
        'effect_size': [10, 15, 10, 10, 10],
        'sample_size': [60, 60, 60, 150, 60],
        'std': [25, 25, 15, 25, 25],
        'sign_lvl': [0.05, 0.05, 0.05, 0.05, 0.01]
    }
    scenarios_df = pd.DataFrame(scenarios)
    scenarios_df['power'] = np.nan
    scenarios_df['Type_i_error'] = np.nan
    scenarios_df['Type_ii_error'] = np.nan
    scenarios_df['ground_truth'] = 0
    scenarios_df['pvalues'] = 0
    # %% Under which hypothesis we run the simulation
    SIMULATION_TYPE = 'alternative_hypothesis'
    n_trials = 1000
    # plot the p value distribution on each
    fig, axs = plt.subplots(nrows=scenarios_df.shape[0], ncols=1, figsize=(8, 16))
    # Calculate statistical power for each scenario
    for scenario_ in scenarios_df.index:
        # for each scenario we store the p-values
        p_values = []
        alpha = scenarios_df.loc[scenario_, 'sign_lvl']
        for _ in range(0, n_trials):
            cases, controls = randomize_trial(
                num_controls=scenarios_df.loc[scenario_, 'sample_size'],
                num_cases=scenarios_df.loc[scenario_, 'sample_size'],
                std_dev=scenarios_df.loc[scenario_, 'std'],
                mean_cases = 220,  # simulation_type[SIMULATION_TYPE][0],
                mean_controls=220 + scenarios_df.loc[scenario_, 'effect_size'],  # simulation_type[SIMULATION_TYPE][1]
            )
            cases = np.round(cases, 2)
            controls = np.round(controls, 2)

            # Perform t-test (two distributions)
            p_value = ttest_ind(cases,
                                controls,
                                alternative='two-sided').pvalue

            # Perform t-test (distributions (difference) vs constant (effect size) )
            # difference = cases - controls
            # p_value = ttest_1samp(a=difference,
            #                       popmean=scenarios_df.loc[scenario_, 'effect_size'],
            #                       alternative='two-sided'
            #                       ).pvalue

            p_values.append(p_value)
        scenarios_df.loc[scenario_, 'pvalues'] = str(p_values)
        if SIMULATION_TYPE == 'null_hypothesis':
            scenarios_df['ground_truth'] = 'Ho True'
            # percent of times that we got a p-value bellow alpha -> significant when it should not
            scenarios_df.at[scenario_, 'Type_i_error'] = len([val for val in p_values if val < alpha]) / len(p_values)

        elif SIMULATION_TYPE == 'alternative_hypothesis':
            scenarios_df['ground_truth'] = 'H1 True'
            # percent of times that we got a p-value above alpha -> non-significant when it should
            scenarios_df.at[scenario_, 'Type_ii_error'] = len([val for val in p_values if val > alpha]) / len(p_values)
            scenarios_df.at[scenario_, 'power'] = (1 - scenarios_df.at[scenario_, 'Type_ii_error']) * 100
        else:
            raise ValueError(f'Undefined simulation hypothesis')
        # Plot histograms for p-value frequency in each scenario
        axs[scenario_].hist(p_values, bins=len(np.unique(p_values)), color='skyblue', edgecolor='black')
        axs[scenario_].set_xlim([0, max(p_values)])
        axs[scenario_].set_title(f'Scenario {scenario_ + 1}')
        axs[scenario_].set_xlabel('P-values')
        axs[scenario_].set_ylabel('Frequency')
        axs[scenario_].grid(True)
    plt.tight_layout()
    plt.show()
