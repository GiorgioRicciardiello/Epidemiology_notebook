"""
    Author: Giorgio Ricciardiello
    Code question 1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Your dictionary and plot code
    dict_count = {
        3.5: 2,
        5: 3,
        5.5: 3.5,
        6: 5.2,
        6.5: 10.5,
        7: 17,
        7.5: 28,
        8: 16,
        8.5: 9,
        9: 4,
        9.5: 2
    }
    array = [key for key, val in dict_count.items() for _ in range(int(val))]
    # Create the bar plot
    plt.hist(array, bins=len(dict_count), edgecolor='black', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Bar Plot')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Set x-ticks to the keys in the dictionary
    plt.xticks(list(dict_count.keys()))
    plt.show()
    # Calculate statistics and print them
    print(f"Mean: {np.mean(array):.2f}")
    print(f"Median: {np.median(array):.2f}")
    print(f"Standard Deviation: {np.std(array):.2f}")
    print(f"15th Percentile: {np.percentile(array, 15):.2f}")
    print(f"95th Percentile: {np.percentile(array, 95):.2f}")

    # Calculate the 95% confidence interval
    mean = np.mean(array)
    std_dev = np.std(array, ddof=1)
    n = len(array)
    confidence_level = 0.95
    t_critical = t.ppf(1 - (1 - confidence_level) / 2, df=n - 1)
    margin_of_error = t_critical * (std_dev / np.sqrt(n))

    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error
    print(f"95% Confidence Interval for the Mean: ({lower_bound:.2f}, {upper_bound:.2f})")
