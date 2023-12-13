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

    # Create a DataFrame with predictors, social and the response of each is known
    data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social]),
        'Cognitive_Change_Score': np.concatenate([y_hih_c, y_low_c])
    })

    # Add interaction term since one model that generates two lines
    data['InteractionTerm'] = data['High_Low_Complexity'] * data['Social_Isolation_Score']

    # Fit multiple linear regression model with the interaction term
    X = data[['High_Low_Complexity', 'Social_Isolation_Score', 'InteractionTerm']]
    # X = sm.add_constant(X)  # add a constant term, but the intercept is zero, so we did not add it
    # we want to fit the model to generate the cognitive change score
    y = data['Cognitive_Change_Score']

    model = sm.OLS(y, X).fit()
    print(model.summary())

    with open('../../output/question_4_ols.tex', 'w') as f:
        f.write(model.summary().as_latex())

    # observed data, predict
    observed_data = pd.DataFrame({
        'High_Low_Complexity': [1] * len(social) + [0] * len(social),
        'Social_Isolation_Score': np.concatenate([social, social])
    })
    # Add the interaction term for predictions
    observed_data['InteractionTerm'] = observed_data['High_Low_Complexity'] * observed_data['Social_Isolation_Score']

    # observed_data_high_complex
    observed_data_high_complex = observed_data.loc[observed_data['High_Low_Complexity'] == 1, :]
    # observed_data = sm.add_constant(observed_data)
    pred_high_complex = model.predict(observed_data_high_complex)

    # plt.plot(social, y_hih_c, label='y_high_c')
    plt.plot(observed_data_high_complex.Social_Isolation_Score, pred_high_complex, label='pred_high_complex', )
    plt.plot(social, y_low_c, label='y_low_c', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Linear Regression with Complexity = 1')
    plt.legend()
    plt.grid()
    plt.show()

    # observed_data_low_complex
    observed_data_low_complex = observed_data.loc[observed_data['High_Low_Complexity'] == 0, :]
    # observed_data = sm.add_constant(observed_data)
    pred_low_complex = model.predict(observed_data_low_complex)
    plt.plot(observed_data_low_complex.Social_Isolation_Score, pred_low_complex,
             label='pred_low_complex', )
    # plt.plot(social, y_low_c, label='pred_low_complex', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Linear Regression with Complexity = 0')
    plt.legend()
    plt.grid()
    plt.show()

    # let's plot now both
    plt.plot(observed_data_high_complex.Social_Isolation_Score, pred_high_complex,
             label='Complexity = 1', )
    plt.plot(observed_data_low_complex.Social_Isolation_Score, pred_low_complex,
             label='Complexity = 0', )
    # plt.plot(social, y_low_c, label='pred_low_complex', linestyle='--')
    plt.xlabel('Social Isolation Score')
    plt.ylabel('Cognitive Change Score')
    plt.title('Predicted Linear Regression Line')
    plt.legend()
    plt.grid()
    plt.show()

