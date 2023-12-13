from sklearn.metrics import cohen_kappa_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import scipy.stats as stats


data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi255\data\EPI225_Midterm_DataforQ23.xlsx'
data = pd.read_excel(data_path)

data['self_rep_score'] = data['self_rep_score'].astype(int)


# Calculate Cohen's Kappa
kappa_score = cohen_kappa_score(y1=data.loc[:, 'self_rep_score'],
                                        y2=data.loc[:, 'totalQscore'],
                                        labels=range(1, 21))

# median_self = np.median(data.loc[:, 'self_rep_score'])
# median_qscore = np.median(data.loc[:, 'totalQscore'])
#
#
#
# t_stat, p_value = ttest_ind(a=data.loc[:, 'self_rep_score'],
#                             b=data.loc[:, 'totalQscore'],
#                             equal_var=False
#                             )
#
#
# print(f"t-statistic: {t_stat}")
# print(f"P-value: {p_value}")

# Compute Pearson correlation coefficient between 'self_rep_score' and 'totalQscore'
correlation = data['self_rep_score'].corr(data['totalQscore'])

# Print the correlation coefficient
print("Pearson correlation coefficient:", correlation)

# Create a Bland-Altman plot
# we expect the mean difference to be zero and the 95% of the difference to be within 2 times the std
num_std = 2
# Create a Bland-Altman plot
mean_score = data.mean(axis=1)
diff = data['self_rep_score'] - data['totalQscore']
mean_diff = np.mean(diff)
std_diff = np.std(diff, ddof=1)

# Calculate the upper and lower limits
lower_limit = mean_diff - num_std * std_diff
upper_limit = mean_diff + num_std * std_diff

plt.scatter(mean_score, diff)
plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Diff.')
plt.axhline(lower_limit, color='gray', linestyle='--', label='Mean-2*std')
plt.axhline(0, color='black', linestyle='--', label='Baseline')
plt.axhline(upper_limit, color='gray', linestyle='--', label='Mean+2*std')

plt.xlabel('Mean Score')
plt.ylabel('Difference between two measures')
plt.title('Bland-Altman Plot')
plt.grid(alpha=0.7)
plt.legend()
plt.show()

# # 95% CI of the bland-Altman plot
# mean_diff = np.mean(diff)
# std_diff = np.std(diff, ddof=1)
# se_diff = std_diff / np.sqrt(len(diff))
# df = len(diff) - 1
# t_critical = stats.t.ppf(0.975, df)
# margin_of_error = t_critical * se_diff
# lower_limit = mean_diff - margin_of_error
# upper_limit = mean_diff + margin_of_error
# print(f"95% Confidence Interval for Limits of Agreement: ({lower_limit}, {upper_limit})")

