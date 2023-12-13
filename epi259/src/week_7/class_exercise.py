import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#%% read data
data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\Stanford_Epidemiology\Courses\1_year\259 - Intro_probability_statistics_Epi\3_exercies\7_unit\Unit7Data.xlsx'
data = pd.read_excel(data_path)

#%% 1. How many athletes have had a stress fracture
num_fractures = data.loc[data['fracture'] == 1, 'fracture'].sum()
num_no_fractures = data.shape[0] - num_fractures
print(f'Number of fractures {num_fractures} and non frctrues {num_no_fractures}')


#%% 2.	How many athletes were classified as having low BMD?
num_lowbmd = data.loc[data['lowbmd'] == 1, 'lowbmd'].sum()
num_no_lowbmd = data.shape[0] - num_lowbmd
print(f'Number of low BMD {num_lowbmd} and non low BMD {num_no_lowbmd}')


#%% 3.	Are the observations in this dataset independent or correlated?
# observations are independent
#%% 4.	What are the mean and standard deviation of BMD Z scores in athletes who have fractured?
frac_bmd = data.loc[data['fracture'] == 1, 'bmdzscore']
frac_bmd_mean = np.mean(frac_bmd)
frac_bmd_std = np.std(frac_bmd)
print(f'BMD Z-score with factures: {np.round(frac_bmd_mean, 3)} +/- {np.round(frac_bmd_std, 3)}')

non_frac_bmd = data.loc[data['fracture'] == 0, 'bmdzscore']
non_frac_bmd_mean = np.mean(non_frac_bmd)
non_frac_bmd_std = np.std(non_frac_bmd)
print(f'BMD Z-score with factures: {np.round(non_frac_bmd_mean, 3)} +/- {np.round(non_frac_bmd_std, 3)}')

#%% 5. plot that compares the BMD Z scores of fractured athletes to non-fractured athletes.
# Using Seaborn to create a box plot
plt.figure(figsize=(8, 6))
sns.boxplot(x='fracture', y='bmdzscore', data=data)
plt.title('Box plot of bmdzscore based on fracture')
plt.xlabel('Fracture')
plt.ylabel('bmdzscore')
plt.xticks(ticks=[0, 1], labels=['Non Fracture', 'Fracture'])
plt.grid(alpha=0.6)
plt.tight_layout()
plt.show()
#%% 6. 6.	If you want to formally compare the BMD Z scores of fractured athletes to non-fractured athletes,
# what statistical test could you use?
# Perform independent t-test
t_stat, p_value = stats.ttest_ind(non_frac_bmd, frac_bmd)

# Print the results
print("T-Statistic for non_frac_bmd vs non_frac_bmd", t_stat)
print("P-Value:", p_value)


