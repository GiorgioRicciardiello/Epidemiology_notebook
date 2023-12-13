import numpy as np
import pandas as pd
from typing import Union, Optional
from numpy import ndarray
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib.pyplot as plt


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
    scenarios = {
        'effect_size': [10, 15, 10, 10, 10],
        'sample_size': [60, 60, 60, 150, 60],
        'std': [25, 25, 15, 25, 25],
        'sign_lvl': [0.05, 0.05, 0.05, 0.05, 0.01]
    }
    scenarios_df = pd.DataFrame(scenarios)
    simulation_type = {
        'null_hypothesis': [250, 250],  # same mean
        'alternative_hypothesis': [250, 270]  # difference in means
    }
    assert simulation_type['null_hypothesis'][0] == simulation_type['null_hypothesis'][1]

    SIMULATION_TYPE = 'alternative_hypothesis'
    TEST = 'one_sample'
    alpha = 0.05
    n_trials = 1000

    one_sided_pvals = {}
    two_sided_pvals = {}
    for scenario in scenarios_df.index:
        two_sided_pvals[scenario] = []
        one_sided_pvals[scenario] = {'less': [], 'greater': []}
        for _ in range(0, n_trials):
            cases, controls = randomize_trial(
                num_controls=scenarios_df.loc[scenario, 'sample_size'],
                num_cases=scenarios_df.loc[scenario, 'sample_size'],
                std_dev=scenarios_df.loc[scenario, 'std'],
                mean_cases=simulation_type[SIMULATION_TYPE][0],
                mean_controls=simulation_type[SIMULATION_TYPE][1]
            )
            cases = np.round(cases, 2)
            controls = np.round(controls, 2)
            differences = cases - controls

            if TEST == 'one_sample':
                # one sample t test
                _, p_value = ttest_1samp(a=differences,
                                         popmean=scenarios_df.loc[scenario, 'effect_size'],
                                         alternative='two-sided')
                two_sided_pvals[scenario].append(p_value)

                _, p_value_less = ttest_1samp(a=differences,
                                                 popmean=scenarios_df.loc[scenario, 'effect_size'],
                                                 alternative='less')
                one_sided_pvals[scenario]['less'].append(p_value_less)

                _, p_value_greater = ttest_1samp(a=differences,
                                              popmean=scenarios_df.loc[scenario, 'effect_size'],
                                              alternative='greater')
                one_sided_pvals[scenario]['greater'].append(p_value_greater)

            else:
                # two sample t test
                _, p_value = ttest_ind(cases, controls)
                two_sided_pvals[scenario].append(p_value)

                # Performing one-sided t-tests (less and greater)
                _, p_value_less = ttest_ind(cases, controls,
                                            alternative='less')
                one_sided_pvals[scenario]['less'].append(p_value_less)

                _, p_value_greater = ttest_ind(cases, controls,
                                               alternative='greater')

                one_sided_pvals[scenario]['greater'].append(p_value_greater)



    # Creating subplots with the number of scenarios
    fig, axs = plt.subplots(nrows=len(two_sided_pvals), ncols=1, figsize=(8, 16))
    # Plotting histograms for p-value frequency in each scenario
    for i in range(len(two_sided_pvals)):
        axs[i].hist(two_sided_pvals[i], bins=50, color='skyblue', edgecolor='black')
        axs[i].set_title(f'Scenario {scenarios_df.index[i]+1}')
        axs[i].set_xlabel('P-values')
        axs[i].set_ylabel('Frequency')
        axs[i].grid(0.7)
    plt.tight_layout()
    plt.show()

    PRINT = False
    if SIMULATION_TYPE == 'null_hypothesis':
        # Type I error rate, only possible when Ho is true
        scenarios_df['Type I Err'] = np.nan
        for scenario_ in scenarios_df.index:
            type_one = len([val for val in two_sided_pvals[scenario_] if val < alpha])/ len(two_sided_pvals[scenario_])
            type_one = np.round(type_one, 2)  # percent of times we got a wrong value
            scenarios_df.loc[scenario_, 'Type I Err'] = type_one
            if PRINT:
                print(f'When testing for H0 (non-significant)'
                      f'\nAbout {type_one*100}% of the time we get a p-value < {alpha} '
                      f'(significant)')
                print(f'Erroneous rejection of Ho')
                print(f'Type I error rate {type_one*100}')

    else:
        # Type II error rate,  only possible when H1 is true
        # errors result when the p-value shows no significance
        scenarios_df['Type II Err'] = np.nan
        scenarios_df['Power'] = np.nan
        for scenario_ in scenarios_df.index:
            type_two = len([val for val in two_sided_pvals[scenario_] if val > alpha]) /len(two_sided_pvals[scenario_])
            type_two = type_two  # percent of times we got a wrong value
            power = (1 - type_two)*100
            scenarios_df['Type II Err'] = type_two
            scenarios_df['Power'] = power
            if PRINT:
                print(f'When testing for H1 (significance)'
                      f'\nAbout {type_two*100}% of the time we get a p-value > {alpha} '
                      f'(non-significant)')
                print(f'Fail to reject Ho when we should had')
                print(f'Type I error rate {type_two*100}')
                print(f'The power of the study is  {power}%')

                # sample size affects Type II error