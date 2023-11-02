import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Any


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


def compute_expected(x: np.ndarray, px: np.ndarray) -> float:
    """Compute the expected value for distribution with unequal probabilities"""
    return np.array([x_ * px_ for x_, px_ in zip(x, px)]).sum()


def compute_std(x: np.ndarray, px: np.ndarray, ex: float) -> float:
    """Compute the STD value for distribution with unequal probabilities"""
    return np.sqrt(np.array([(x_ - ex) ** 2 * px_ for x_, px_ in zip(x, px)]).sum())


def simulate_trial(n_a: int, n_b: int, p_a: float, p_b: float, diff: str = 'a-b',
                   num_trials: int = 30000, plot_histogram: bool = False, return_proportions: bool = False) -> tuple[
    list[Union[Union[float, int], Any]], Any, Any]:
    """
    Simulate a randomized trial and calculate the difference in event occurrences or proportions between two groups.

    :param n_a: Number of individuals in group a
    :param n_b: Number of individuals in group b
    :param p_a: Probability of event in group a
    :param p_b: Probability of event in group b
    :param diff: 'a-b' for a minus b, 'b-a' for b minus a
    :param num_trials: Number of virtual trials to run
    :param plot_histogram: Whether to plot a histogram of trial results
    :param return_proportions: Whether to return proportions (True) or counts (False)
    :return: List of trial results (differences), mean, and standard deviation
    """
    trial_results = []

    for _ in range(0, num_trials):
        infections_a = np.random.binomial(n_a, p_a)
        infections_b = np.random.binomial(n_b, p_b)

        if return_proportions:
            proportion_a = infections_a / n_a
            proportion_b = infections_b / n_b

            if diff == 'a-b':
                trial_results.append(proportion_a - proportion_b)
            elif diff == 'b-a':
                trial_results.append(proportion_b - proportion_a)
        else:
            if diff == 'a-b':
                trial_results.append(np.abs(infections_a - infections_b))
            elif diff == 'b-a':
                trial_results.append(np.abs(infections_b - infections_a))

    if plot_histogram:
        x_label = f"{'Difference in Proportions' if return_proportions else 'Excess Infections'} ({'Control - Treatment' if diff == 'a-b' else 'Treatment - Control'})"
        plt.hist(trial_results, bins=30, color='blue', alpha=0.6)
        plt.xlabel(x_label)
        plt.ylabel("Frequency")
        plt.title(
            f"Histogram of {'Proportion Differences' if return_proportions else 'Excess Infections'} in {num_trials} Virtual Trials")
        plt.show()
    return trial_results, np.mean(trial_results), np.std(trial_results)


if __name__ == "__main__":
    print("sd")

    # HW4 - EX 6
    n = 8  # Total number of patients
    p = 0.25  # Probability of a patient responding to treatment
    x = 3
    total_probability_percentage =binomial_probability(x, n, p) * 100
    # HW4 - EX 7
    n = 8  # Total number of patients
    p = 0.25  # Probability of a patient responding to treatment
    x_values = [0, 1, 2]  # Number of patients responding (0, 1, or 2)
    total_probability = 0  # Initialize the total probability

    for x in x_values:
        probability_x = binomial_probability(x, n, p)
        total_probability += probability_x

    # Convert the total probability to a percentage
    total_probability_percentage = total_probability * 100

    # randomized_trial
    # how many more infections occurred in the placebo than in the vaccine group in a single trial
    results, mean, std_deviation = simulate_trial(n_a=1000, n_b=1000, p_a=0.15, p_b=0.05, diff='a-b',
                                                  num_trials=30000,
                                                  plot_histogram=True, return_proportions=False)
    print(f"Mean: {mean}, Standard Deviation: {std_deviation}")

    # HW4 - EX 8
    x = []