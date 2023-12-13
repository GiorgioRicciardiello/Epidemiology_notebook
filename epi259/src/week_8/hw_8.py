import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from scipy.stats import shapiro

data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\classdata.xlsx'
data = pd.read_excel(data_path)
data.fillna(0, inplace=True)
#%% 1. superimpose a Loess smoothing line with mild curvature
lowess = sm.nonparametric.lowess(endog=data['bushjr'],  # The y-values of the observed points
                                 exog=data['bushsr'],  # The x-values of the observed points
                                 frac=0.5 # Adjust the 'frac' parameter for curvature control
                                 )

sns.scatterplot(data=data, y= 'bushjr', x= 'bushsr', label='Data points')
sns.lineplot(x=lowess[:, 0], y=lowess[:, 1], color='red', label='Loess smoothing')
plt.xlabel('bushjr')
plt.ylabel('bushsr')
plt.title('Scatter Plot with Loess Smoothing')
plt.legend()
plt.show()

#%% Pearson correlation

x = data.loc[~data['bushjr'].isna(), 'bushjr'].values
y = data.loc[~data['bushsr'].isna(), 'bushsr'].values
correlation_coefficient, p_value = pearsonr(x, y)

#%% What is the beta coefficient from a linear regression in which ratings of Bush Sr. is the predictor and ratings of Bush Jr. is the outcome? Round to two decimal places.
X_data = data.loc[:, 'bushsr'].values.reshape(-1, 1)
y = data.loc[:, 'bushjr'].values
reg = LinearRegression().fit(X_data, y)
reg.coef_

#%% Add the variable politics to the linear regression model that you ran in (3). Which of the following is true?
X_data = data.loc[:, ['bushsr', 'politics']].values
y = data.loc[:, 'bushjr'].values
reg = LinearRegression().fit(X_data, y)
reg.coef_

# Predict the rating for someone with politics=50 and bushsr=50
new_data = np.array([[50, 50]])
predicted_rating = reg.predict(new_data)
predicted_rating_whole_number = int(round(predicted_rating[0]))


#%% residuals and leverage
X_data = data.loc[:, ['bushsr', 'politics']].values
y = data.loc[:, 'bushjr'].values

# Fit the OLS model
X_data = sm.add_constant(X_data)  # Add a constant term to the predictor
model = sm.OLS(y, X_data).fit()

# Get residuals and leverage
residuals = model.resid
leverage = model.get_influence().hat_matrix_diag

# Perform Shapiro-Wilk test
statistic, p_value = shapiro(residuals)

# Print the test results
print(f"Shapiro-Wilk test statistic: {statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the results
if p_value > 0.05:
    print("The residuals appear to be normally distributed.")
else:
    print("The residuals do not appear to be normally distributed.")

# Create a bar plot of absolute residuals
plt.bar(range(len(residuals)), np.abs(residuals), color='red')
plt.title('Absolute Residuals')
plt.xlabel('Observation')
plt.ylabel('Absolute Residual')
plt.show()

# Plot residuals vs fitted values
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title('Residuals vs Fitted Values')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()

# Plot leverage
sns.regplot(x=leverage, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title('Residuals vs Leverage')
plt.xlabel('Leverage')
plt.ylabel('Residuals')
plt.show()

#%%
formula = "obama ~ bushjr + politics"
model2 = sm.formula.glm(formula=formula, data=data, family=sm.families.Binomial()).fit()
print(model2.summary())
