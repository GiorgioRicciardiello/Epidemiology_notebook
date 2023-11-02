"""
Objective: Identy error in two columns of the data set. The dataset was adultered.

The data is perfectly sorted, expected for a few observations in the cheated column.
They changed the value to make the cheater count higher and make it significant the difference
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_test

data_path = r"C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\Stanford_Epidemiology\Courses\1_year\EPI 259 - Introduction to Probability and Statistics for Epidemiology\exercies\1_unit\data DAC Study_4 PS.csv"
data = pd.read_csv(data_path)

data = data.loc[:, ['cheated', 'Numberofresponses']]


cheated = data[data.loc[:, 'cheated'] == 1]
cheated.drop(columns='cheated', inplace=True)

non_cheated = data[data.loc[:, 'cheated'] == 0]
non_cheated.drop(columns='cheated', inplace=True)



# compute the mean
mean = data.mean()
std = data.std()

print(f'Observations; {cheated.shape[0]}')
print(f'Cheated:  {cheated.mean().values[0]} ({cheated.std().values[0]})')
print(f'Non Cheated:  {non_cheated.mean().values[0]} ({non_cheated.std().values[0]})')


data_two_col = pd.concat([cheated, non_cheated], axis=1, ignore_index=True)

row_index = 139 # Replace with the index you want to access
fixed_cheated = cheated.copy()

fixed_cheated.loc[row_index, 'Numberofresponses'] = 5
fixed_cheated.loc[140, 'Numberofresponses'] = 5


print(f'Fixed Cheated:  {fixed_cheated.mean().values[0]} ({fixed_cheated.std().values[0]})')



# The data
