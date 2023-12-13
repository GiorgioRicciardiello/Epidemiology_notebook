import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, ranksums


# Given data
mean_treated = 7.1197
mean_control = 5.6055
std_treated = 1.672
std_control = 1.582
n_treated = 20
n_control = 20
t_value = 2.0  # T-value for a 95% confidence interval with 38 degrees of freedom

# Calculate the standard error of the difference in means, we use the std of each group
std_error_diff = np.sqrt((std_treated**2 / n_treated) + (std_control**2 / n_control))

# Calculate the margin of error
margin_of_error = t_value * std_error_diff

diff = mean_treated - mean_control
# Calculate the confidence interval
confidence_interval = (diff - margin_of_error,
                       diff + margin_of_error)

print("95% Confidence Interval:", confidence_interval)

data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\classdata.xlsx'
data = pd.read_excel(data_path)

data.loc[data['exercise'] > 3.25, 'highex'] = 1
data.loc[(data['exercise'] >= 0) & (data['exercise'] <3.25), 'highex'] = 0


# data.loc[:, ['wakeup', 'highex']]

# Assuming 'data' is your DataFrame
plt.figure(figsize=(8, 6))
sns.boxplot(x='highex', y='wakeup', data=data)
plt.title('highex vs wakeup')
plt.xlabel('wakeup')
plt.ylabel('highex')
# plt.xticks(ticks=[0, 1], labels=['e', 'Fracture'])
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()

#%% observed diffdrence in means measures in minutes
diff = data.loc[data['highex'] == 1, 'wakeup'].reset_index(drop=True) - data.loc[data['highex'] == 0, 'wakeup'].reset_index(drop=True)
np.mean(diff)*60

#%% t-test to compare the mean waku-up time between high exercisers and low exercises
t_statistic, t_p_value = ttest_ind(a=data.loc[data['highex'] == 1, 'wakeup'],
                                 b=data.loc[data['highex'] == 0, 'wakeup'],
                                 alternative='two-sided')

#%% Perform Wilcoxon rank-sum test
ranksums_statistic, ranksums_p_value = ranksums(x=data.loc[data['highex'] == 1, 'wakeup'],
                                y=data.loc[data['highex'] == 0, 'wakeup'],
                                alternative='two-sided')

# ranksums_statistic - > effect size and the p-value tells us the extent on which is significant
# Create a DataFrame with the results
results_df = pd.DataFrame({
    'Test': ['t-test', 'Wilcoxon Rank-Sum Test'],
    'Statistic': [t_statistic, t_p_value],
    'P-Value': [ranksums_statistic, ranksums_p_value]
})


# Print the DataFrame
print(results_df)