import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pandas as pd
import scipy.stats as stats
from typing import Union, Optional, Tuple
from numpy import ndarray
import random
from tabulate import tabulate
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

    # %% Question 2
    data = pd.read_excel(
        r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\finaldata.xlsx')

    # Report the mean, median, standard deviation, IQR, and N for all three variables
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
    # Plot boxplots for each variable
    for i, var in enumerate(data.columns):
        sns.boxplot(y=data[var], ax=axes[i])
        axes[i].set_title(f'Boxplot of {var}', fontsize=18)
        axes[i].grid()
        # first order statistics
        values = data[var]
        mean_val = np.mean(values)
        median_val = np.median(values)
        std_dev = np.std(values)
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        count = len(values)
        text = f'\nSamples: {count}\nMean: {mean_val:.2f}\nmedian: {median_val:.2f}\nstd: {std_dev:.2f}\nIQR: {iqr:.2f}'
        axes[i].text(0.5, -0.30, text, ha='center', transform=axes[i].transAxes, fontsize=20)

    plt.tight_layout()
    plt.show()
    # Evaluate whether age is related to depression
    # two continuous variables of different units, we can do correlation to determine if there is a linear relation
    corr_age_dep = data['age'].corr(data['depression'])

    # Fit the linear regression model
    X = sm.add_constant(data['age'])
    model = sm.OLS(data['depression'], X)
    results = model.fit()
    results.summary()

    # Evaluate whether depression is related to social media use
    corr_age_socm = data['age'].corr(data['socialmedia'])
    # Fit the linear regression model
    X = sm.add_constant(data['age'])
    model = sm.OLS(data['socialmedia'], X)
    results = model.fit()
    results.summary()

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


    # measured data for the PG and the CG
    campo_drop_dj = {
        'Group': ['PG'] * 4 + ['CG'] * 4,
        'T_measure': [1, 2, 3, 4] * 2,
        'Mean': [24.9, 28.2, 28.9, 29.4, 27.1, 24.6, 25.6, 25.2],
        'Std': [1.1, 0.9, 0.8, 1.0, 1.0, 0.8, 0.9, 0.8]
    }
    campo_drop_dj_df = pd.DataFrame(campo_drop_dj)

    # Calculate mean change (difference within measures) for each group
    mean_change_pg = campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'PG', 'Mean'].diff().dropna().values
    mean_change_cg = campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'CG', 'Mean'].diff().dropna().values

    # Calculate pooled standard deviation
    std_pool = pooled_standard_deviation(std_devs1=campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'PG', 'Std'],
                                         std_devs2=campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'CG', 'Std']
                                         )
    # std_pool = np.sqrt(
    #     (
    #             (campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'PG', 'Std'] ** 2).sum() +
    #             (campo_drop_dj_df.loc[campo_drop_dj_df['Group'] == 'CG', 'Std'] ** 2).sum()
    #     ) / (2 * len(mean_change_pg) - 2)
    # )

    # Calculate the effect size
    effect_size_value = effect_size(np.mean(mean_change_pg), np.mean(mean_change_cg), std_pool)

    # confidence interval for the effect size
    alpha = 0.05
    t_critical = stats.t.ppf(1 - alpha / 2, df=len(mean_change_pg) - 1)
    margin_of_error = t_critical * np.sqrt((std_pool ** 2) / len(mean_change_pg))

    confidence_interval = (effect_size_value - margin_of_error, effect_size_value + margin_of_error)
    confidence_interval = tuple(round(value, 3) for value in confidence_interval)

    print(f"Effect Size: {np.round(effect_size_value, 3)}; CI(95%) [{confidence_interval}]")

    # %% Question 4
    # Generate social isolation scores
    social = np.linspace(-3, 3, 100)

    x1_high, y1_high = -2, -0.8
    x2_high, y2_high = 2, 0.8
    m_high = (y2_high - y1_high) / (x2_high - x1_high)
    b_high = y1_high - m_high * x1_high
    y_hih_c = m_high * social + b_high

    # Given points for low complexity line
    x1_low, y1_low = -2, -1.5
    x2_low, y2_low = 2, 1.5
    m_low = (y2_low - y1_low) / (x2_low - x1_low)
    b_low = y1_low - m_low * x1_low
    y_low_c = m_low * social + b_low

    plt.plot(social, y_hih_c, label='y_hih_c')
    plt.plot(social, y_low_c, label='y_low_c', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Linear Regression Line')
    plt.legend()
    plt.grid()
    plt.show()

    # Create a DataFrame with predictors
    data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social]),
        'Cognitive_Change_Score': np.concatenate([y_hih_c, y_low_c])
    })

    # Fit multiple linear regression model
    X = sm.add_constant(data[['High_Low_Complexity', 'Social_Isolation_Score']])
    y = data['Cognitive_Change_Score']
    model = sm.OLS(y, X).fit()

    # Display regression results
    print(model.summary())

    # Create a DataFrame with predictors for predictions
    observed_data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social])
    })

    # Add a constant term for predictions
    observed_data = sm.add_constant(observed_data)

    # Use the fitted model to predict 'Cognitive_Change_Score' for observed_data
    predicted_values = model.predict(observed_data)

    plt.plot(observed_data.Social_Isolation_Score, predicted_values,
             label='Linear Regression Model', )
    plt.plot(social, y_hih_c, label='y_high_c')
    plt.plot(social, y_low_c, label='y_low_c', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Predicted Linear Regression Line')
    plt.legend()
    plt.grid()
    plt.show()


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

    fig, axs = plt.subplots(nrows=scenarios_df.shape[0], ncols=1, figsize=(8, 16))
    for idx_, scenario_ in scenarios_df.iterrows():
        # print(scenario_.n_a)
        p_values = []
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
        axs[idx_].hist(p_values, bins=len(np.unique(p_values)), color='skyblue', edgecolor='black')
        axs[idx_].set_xlim([0, max(p_values)])
        axs[idx_].set_title(f'Scenario {idx_ + 1}')
        axs[idx_].set_xlabel('P-values')
        axs[idx_].set_ylabel('Frequency')
        axs[idx_].grid(True)

    plt.tight_layout()
    plt.show()
    # Print the results of each scenario in a single table
    result = scenarios_df.drop(columns='pvalues')
    print(tabulate(result, headers='keys', tablefmt='psql'))


