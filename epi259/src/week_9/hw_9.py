import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\classdata_unit_9.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
# data.dropna(inplace=True)

#%% Make a plot that corresponds to the following regression model:
# Expected optimism = intercept + alcohol (drinks/week) + Varsity sports (1=played varsity sports in high school, 0=did not) + Varsity*alcohol

# make the interaction term
data['Varsity_alcohol'] = data['varsity'] * data['alcohol']

# Fit the regression model
X = sm.add_constant(data[['alcohol', 'varsity', 'Varsity_alcohol']])
y = data['optimism']
model = sm.OLS(y, X).fit()


# Plot the regression lines
sns.lmplot(x='alcohol', y='optimism', hue='varsity', data=data, ci=None)
plt.title('Regression Plot for Expected Optimism')
plt.xlabel('Alcohol (drinks/week)')
plt.ylabel('Expected Optimism')
plt.tight_layout()
plt.show()

#%% Fit the following linear regression model:
#Expected optimism = intercept + alcohol (drinks/week) + Varsity sports (1=played varsity sports in high school, 0=did not) + Varsity*alcohol
#What is the resulting beta coefficient for the interaction term?

summary = model.summary()

# Extract the coefficients
coefficients = pd.DataFrame(summary.tables[1].data)
interaction_coefficient = float(coefficients.loc[0, 'coef'])  # Assuming 'Varsity_alcohol' is the third row

# Print the result
print(f"The beta coefficient for the interaction term is: {interaction_coefficient:.2f}")

