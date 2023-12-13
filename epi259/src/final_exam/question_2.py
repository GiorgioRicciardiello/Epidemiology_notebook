import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import  Path
import scipy.stats as stats
if __name__ == '__main__':
    # %% Question 2
    root = Path(__file__).parents[2]
    data_path = root.joinpath(r'rawdata\finaldata.xlsx')
    data = pd.read_excel(data_path)
    data['age'] = pd.to_numeric(data['age'], errors='coerce')
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

    # evaluate the
    sns.pairplot(data)
    plt.show()

    # Evaluate whether age is related to depression
    # age is categorical and depression is continuous -> kruskal
    corr_age_dep = data['age'].corr(data['depression'])
    age_lvls = data['age'].unique()
    statistic, p_value = stats.kruskal(data['depression'][data['age'] == age_lvls[0]],
                                       data['depression'][data['age'] == age_lvls[1]],
                                       data['depression'][data['age'] == age_lvls[2]],
                                       data['depression'][data['age'] == age_lvls[3]])

    # Fit ordinal logistic regression model
    model = sm.MNLogit.from_formula(' age ~ depression', data=data)
    result = model.fit()
    print(result.summary())

    # Evaluate whether depression is related to social media use
    corr_dep_socm = data['depression'].corr(data['socialmedia'])
    # Fit the linear regression model
    X = sm.add_constant(data['depression'])
    model = sm.OLS(data['socialmedia'], X)
    results = model.fit()
    results.summary()
    print(results.summary())
    with open('../../output/regression_results_dep_socialmedia.tex', 'w') as f:
        f.write(results.summary().as_latex())
