"""
    Author: Giorgio Ricciardiello
    Code question 6
Simulation of an experiment with Binomial distribution with n=24 and p=0.5
"""
import numpy as np
from scipy.stats import shapiro, binom
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import probplot

def binomial_probability(x: int, n: int, p: float) -> float:
    """
    Compute the binomial distribution for binary events
    :param x: Number of successful events
    :param n: Total number of trials
    :param p: Probability of success for each trial
    :return: Probability of the event
    :return:
        probability of the event
    """
    # return binom_pro * (p) ** x * (1 - p) ** (n - x)
    return binom.pmf(x, n, p)

if __name__ == "__main__":
    #%% Evaluate if a Binomal experiment has real of fake data
    recorded_data = {
        'infantid': [id for id in range(1, 21)],
        'num_right_gaze' : [12, 11, 13, 12, 11, 12, 13, 12, 11, 11, 12, 12, 13, 14, 12, 10, 12, 12, 11, 13],
    }
    recorded_data['num_left_gaze'] = [20-right_ for right_ in recorded_data['num_right_gaze']]
    df = pd.DataFrame(recorded_data)

    # run a simulation of what its is expected, two possible outcome per event, with equal prob, Binomal distribution
    n = 25  # Total number of records
    p_right = 0.5  # Probability of each outcome occurring
    p_left = 0.5  # Probability of each outcome occurring
    assert p_right + p_left == 1
    # Simulating the control experiment - generating NumRightGaze data
    n_trials = 1000

    # method 1: calculating the theoretical probability mass function for the observed data based on the binomial
    # distribution parameters. We want to simulate the drawn of 25 events, where each have a binary outcome of
    # probability p
    simulation_num_right_gaze = np.random.binomial(n=n, p=p_right, size=n_trials)
    simulation_num_left_gaze = np.random.binomial(n=n, p=p_left, size=n_trials)

    #%% Plotting the binomial distributions for right and left gaze
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(18, 10))
    sns.kdeplot(data=df, x='num_right_gaze', label='Observed Right Gaze', color='orange', ax=axs[0, 0])
    axs[0, 0].set_title('Right Gaze Distribution')
    axs[0, 0].set_xticks(np.arange(len(recorded_data['infantid'])))
    axs[0, 0].set_xlabel('Number of Successes')
    axs[0, 0].set_ylabel('Probability')
    axs[0, 0].grid(0.7)
    axs[0, 0].text(0.5, -0.2,
                   f'Mean: {np.mean(recorded_data["num_right_gaze"]):.2f}\nSD: {np.std(recorded_data["num_right_gaze"]):.2f}',
                   horizontalalignment='center', verticalalignment='center', transform=axs[0, 0].transAxes)
    axs[0, 0].legend()
    axs[0, 0].set_xlim(min(recorded_data['infantid']), max(recorded_data['infantid']))


    sns.kdeplot(data=df, x='num_left_gaze', label='Observed Left Gaze', ax=axs[0, 1])
    axs[0, 1].set_title('Left Gaze Distribution')
    axs[0, 1].set_xticks(np.arange(len(recorded_data['num_right_gaze'])))
    axs[0, 1].set_xlabel('Number of Successes')
    axs[0, 1].set_ylabel('Probability')
    axs[0, 1].grid(0.7)
    axs[0, 1].text(0.5, -0.2,
                   f'Mean: {np.mean(recorded_data["num_left_gaze"]):.2f}\nSD: {np.std(recorded_data["num_left_gaze"]):.2f}',
                   horizontalalignment='center', verticalalignment='center', transform=axs[0, 1].transAxes)
    axs[0, 1].legend()
    axs[0, 1].set_xlim(min(recorded_data['infantid']), max(recorded_data['infantid']))


    axs[1, 0].hist(simulation_num_right_gaze, bins=np.arange(n + 2) - 0.5, alpha=0.7,
                   edgecolor='black', color='orange',
                   label=f'Simulation Right Gaze Binomial (n={n}, p={p_right})')
    axs[1, 0].set_title(f'Simulated NumRightGaze Binomial Distribution {n_trials} trials')
    axs[1, 0].set_xlabel('Number of Successes')
    axs[1, 0].set_ylabel('Probability')
    axs[1, 0].set_xticks(np.arange(n + 1))
    axs[1, 0].grid(0.7)
    axs[1, 0].text(0.5, -0.2,
                   f'Mean: {np.mean(simulation_num_right_gaze):.2f}\nSD: {np.std(simulation_num_right_gaze):.2f}',
                   horizontalalignment='center', verticalalignment='center', transform=axs[1, 0].transAxes)
    axs[1, 0].legend()
    axs[1, 0].set_xlim(min(recorded_data['infantid']), max(recorded_data['infantid']))


    axs[1, 1].hist(simulation_num_left_gaze, bins=np.arange(n + 2) - 0.5, alpha=0.7, edgecolor='black',
                   label=f'Simulation Left Gaze Binomial (n={n}, p={p_left})')
    axs[1, 1].set_title(f'Simulated NumLeftGaze Binomial Distribution {n_trials} trials')
    axs[1, 1].set_xlabel('Number of Successes')
    axs[1, 1].set_ylabel('Probability')
    axs[1, 1].set_xticks(np.arange(n + 1))
    axs[1, 1].grid(0.7)
    axs[1, 1].text(0.5, -0.2,
                   f'Mean: {np.mean(simulation_num_left_gaze):.2f}\nSD: {np.std(simulation_num_left_gaze):.2f}',
                   horizontalalignment='center', verticalalignment='center', transform=axs[1, 1].transAxes)
    axs[1, 1].legend()
    axs[1, 1].set_xlim(min(recorded_data['infantid']), max(recorded_data['infantid']))

    plt.tight_layout()
    plt.show()

    # Testing for Gaussian Distribution (Normality Test)
    stat, p = shapiro(df['num_right_gaze'])
    print("Shapiro-Wilk Test:")
    print(f"Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print("Data is normally distributed")
    else:
        print("Data is not normally distributed")

    # Generate a Q-Q plot for visual assessment of normality
    plt.figure(figsize=(6, 4))
    probplot(df['num_right_gaze'], dist="norm", plot=plt)
    plt.title("Q-Q plot")
    plt.show()

    # Empirical 68–95–99.7 rule for Gaussian Distribution test
    # If gaussian we expect that
    # - 68% of observations fall within the first standard deviation
    # - 95% within the first two standard deviations
    #  - 99.7% within the first three standard deviations
    mean = df['num_right_gaze'].mean()
    std_dev = df['num_right_gaze'].std()

    # Calculate percentages within 1, 2, and 3 standard deviations from the mean
    # One standard deviation (µ ± σ)
    within_one_std = np.mean(np.abs(df['num_right_gaze'] - mean) <= std_dev)
    # Two standard deviations (µ ± 2σ)
    within_two_std = np.mean(np.abs(df['num_right_gaze'] - mean) <= 2 * std_dev)
    # Three standard deviations (µ ± 3σ)
    within_three_std = np.mean(np.abs(df['num_right_gaze'] - mean) <= 3 * std_dev)

    # Print the percentages within each range
    print(f"Percentage within one standard deviation: {within_one_std * 100:.2f}% (Expected 68%)")
    print(f"Percentage within two standard deviations: {within_two_std * 100:.2f}% (Expected 95%)%")
    print(f"Percentage within three standard deviations: {within_three_std * 100:.2f}% (Expected 99.7%)%")

    # Calculate Q1 and Q3
    q1 = np.percentile(df['num_right_gaze'], 25)
    q3 = np.percentile(df['num_right_gaze'], 75)
