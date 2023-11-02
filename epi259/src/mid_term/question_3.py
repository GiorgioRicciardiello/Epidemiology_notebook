"""
    Author: Giorgio Ricciardiello
    Code question 3

Write a simulation in SAS or R that proves that the marginal  probability for the 30th person is indeed 1 in 10.
Please provide your code below. Use at least 100,000 repeats in your final simulation.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import  Path

# get the file from path
root = Path(__file__).parents[2]
data_path = root.joinpath(r'rawdata\report_unintentional_fall_deaths_rates_per_100000.csv')
assert data_path.exists()

# read the file
data = pd.read_csv(data_path)
# select the non-nan rows and the desired columns
data_plot = data.loc[data['Crude Rate'].notna(),
['Year', 'Crude Rate', 'Age-Adjusted Rate']]
# remove the total count of mortality
data_plot = data_plot.loc[data_plot['Year'] != 'Total', :]
# from strings to integers to use the x limits
data_plot['Year'] = data_plot['Year'].astype(int)
# generate plot
lbl_fontsize = 22
ticks_fontsize = 20
legend_fontsize = 18
plt.figure(figsize=(16, 6))
sns.lineplot(data=data_plot,
             x='Year',
             y='Crude Rate',
             color='blue',
             label='Crude Rate')
sns.lineplot(data=data_plot,
             x='Year',
             y='Age-Adjusted Rate',
             color='red',
             label='Age-Adjusted Rate')
plt.xlabel('Years', fontsize=lbl_fontsize)
plt.ylabel('Mortality per 100,000', fontsize=lbl_fontsize)
plt.xticks(data_plot['Year'].unique(),
           fontsize=ticks_fontsize,
           rotation=45)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(data_plot['Year'].min(), data_plot['Year'].max())
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



