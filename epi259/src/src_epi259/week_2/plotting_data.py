import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\reports-data-export.csv'

data = pd.read_csv(data_path)

# Convert the 'Year' column to numeric (errors='coerce' will convert non-numeric values to NaN)
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')

# drop nan rows
data.dropna(how='all', inplace=True)

# Convert the 'Year' column from float to int and assign it back to the DataFrame
data['Year'] = data['Year'].astype(int)

# Drop rows where 'Year' is NaN
data.dropna(subset=['Year'], inplace=True)

data.loc[:, 'Year'].astype(int)

# Sort the DataFrame by the 'Year' column in ascending order
data_sorted = data.sort_values(by='Year')

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=data,
             x='Year',
             y='Age-Adjusted Rate',
             label='Age-Adjusted Rate')
sns.scatterplot(data=data,
                x='Year',
                y='Age-Adjusted Rate',)
# Set x-axis ticks at exact year values
plt.xticks(data_sorted['Year'])
# Add labels and a legend
ax.set_xlabel('Year')
ax.set_ylabel('Mortality Rate')
ax.set_title('Line Plot and Scatter Plot')
ax.legend()
plt.grid(.7)
plt.tight_layout()
plt.show()


# %% 2. The indidence of homocides from terrorirms was higher than the incidence of homocides from fierarms?

# Download: All Intents Terrorism Deaths and Rates per 100,000
#           Data Years: 2001, United States, All Ages, Both Sexes, All Races, All Ethnicities
terrorism_data_path = r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\Epidemiology_notebook\epi259\rawdata\terrorism_reports-data-export.csv'
terrorism_data = pd.read_csv(terrorism_data_path)
