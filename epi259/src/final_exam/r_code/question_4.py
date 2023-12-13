import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # %% Question 4
    # Generate social isolation scores
    social = np.linspace(-3, 3, 100)


    # Given points for high complexity line
    x1_high, y1_high = -2, -0.08
    x2_high, y2_high = 2, 0.08
    m_high = (y2_high - y1_high) / (x2_high - x1_high)
    b_high = y1_high - m_high * x1_high  # intercept is zero
    y_hih_c = m_high * social + b_high

    # Given points for low complexity line
    x1_low, y1_low = -2, -0.15
    x2_low, y2_low = 2, 0.15
    m_low = (y2_low - y1_low) / (x2_low - x1_low)
    b_low = y1_low - m_low * x1_low  # intercept is zero
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

    # Add interaction term since one model that generates two lines
    data['InteractionTerm'] = data['High_Low_Complexity'] * data['Social_Isolation_Score']

    # Fit multiple linear regression model with the interaction term
    X = sm.add_constant(data[['High_Low_Complexity', 'Social_Isolation_Score', 'InteractionTerm']])
    y = data['Cognitive_Change_Score']
    model = sm.OLS(y, X).fit()
    print(model.summary())
    with open('../../output/question_4_ols.tex', 'w') as f:
        f.write(model.summary().as_latex())

    # Create a DataFrame with predictors for predictions
    observed_data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social])
    })
    # Add the interaction term for predictions
    observed_data['InteractionTerm'] = observed_data['High_Low_Complexity'] * observed_data['Social_Isolation_Score']
    # Add a constant term for predictions
    observed_data = sm.add_constant(observed_data)

    # Use the fitted model to predict 'Cognitive_Change_Score' for observed_data
    predicted_values = model.predict(observed_data)


    plt.plot(social, y_hih_c, label='y_high_c')
    plt.plot(observed_data.Social_Isolation_Score, predicted_values,
             label='Linear Regression Model', )
    plt.plot(social, y_low_c, label='y_low_c', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Predicted Linear Regression Line')
    plt.legend()
    plt.grid()
    plt.show()



    #%%
    # Create a DataFrame with predictors
    data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social]),
        'Cognitive_Change_Score': np.concatenate([y_hih_c, y_low_c])
    })

    # Add interaction term
    data['InteractionTerm'] = data['High_Low_Complexity'] * data['Social_Isolation_Score']

    # Fit multiple linear regression model with the interaction term
    X = sm.add_constant(data[['High_Low_Complexity', 'Social_Isolation_Score', 'InteractionTerm']])
    y = data['Cognitive_Change_Score']
    model = sm.OLS(y, X).fit()
    print(model.summary())

    # Make predictions for both High and Low Complexity
    observed_data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social])
    })

    # # Make predictions for both High and Low Complexity
    # observed_data = pd.DataFrame({
    #     'High_Low_Complexity': [1] * len(social),
    #     'Social_Isolation_Score': social
    # })


    # Add the interaction term for predictions
    observed_data['InteractionTerm'] = observed_data['High_Low_Complexity'] * observed_data['Social_Isolation_Score']

    # Add a constant term for predictions
    observed_data = sm.add_constant(observed_data)

    # Use the fitted model to predict 'Cognitive_Change_Score' for observed_data
    predicted_values = model.predict(observed_data)

    # plt.plot(social, y_hih_c, label='Original High Complexity Line')
    plt.plot(observed_data['Social_Isolation_Score'], predicted_values, label='Predicted Linear Regression Model')
    # plt.plot(social, y_low_c, label='Original Low Complexity Line', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Predicted Linear Regression Line')
    plt.legend()
    plt.grid()
    plt.show()

    #%%
