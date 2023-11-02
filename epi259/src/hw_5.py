"""You want to test the hypothesis that girls under age 10 are faster swimmers (on average) than boys under age 10.
You randomly sample 30 girl swimmers and 30 boy swimmers from local swim clubs and clock their 50-yard freestyle
times to test this hypothesis. You find that the girls are on average 5 seconds faster than boys.

Set up a computer simulation in which you assume that boys and girls are equally fast swimmers (=null hypothesis).
Assume that, in girls, swim times are normally distributed with a mean of 60 and a standard deviation of 10 seconds.
Assume the same distribution for boys. Randomly sample 30 boys and 30 girls and calculate the difference in the means
of the two groups. Repeat this a large number of times (e.g., 10,000 times). Plot a histogram of the difference in
means. Which of the following histogram results?
"""
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
num_simulations = 10000
sample_size = 30  # 30 samples per group
mean_girls = 60  # Mean swim time for girls
std_dev = 10  # Standard deviation for both boys and girls

differences = []  # List to store the differences in means

for _ in range(num_simulations):
    # Generate samples for boys and girls
    boys_sample = np.random.normal(loc=mean_girls, scale=std_dev, size=sample_size)
    girls_sample = np.random.normal(loc=mean_girls, scale=std_dev, size=sample_size)

    # Calculate the difference in means for the current sample
    mean_diff = np.mean(girls_sample) - np.mean(boys_sample)
    differences.append(mean_diff)

# Plotting the histogram
plt.figure(figsize=(8, 6))
plt.hist(differences, bins=30, alpha=0.7, color='skyblue')
plt.axvline(np.mean(differences), color='red', linestyle='dashed', linewidth=2, label='Mean Difference')
plt.xlabel('Difference in Means (Girls - Boys)')
plt.ylabel('Frequency')
plt.title('Histogram of Difference in Means between Girls and Boys')
plt.legend()
plt.grid(True)
plt.show()

print(f'What is the mean difference of the means {np.mean(differences)}')
print(f'What is the std of the difference of the means {np.std(differences)}')

# calculate the p-value by counting the number of simulations where the difference in means was equal to or greater
# than 5 or equal to or less than -5. The p-value is expressed as a percentage.
#
# The p-value indicates the likelihood of obtaining a difference in means as extreme as the observed difference (5
# seconds) between boys and girls under the assumption that there is no true difference between them. A lower p-value
# would suggest stronger evidence against the null hypothesis.
sim_diff_gr_five = [diff for diff in differences if diff >= 5]
sim_diff_lw_five = [diff for diff in differences if diff <= -5]

pvalue = (len(sim_diff_gr_five) + len(sim_diff_lw_five)) / num_simulations * 100

print(f"P-value: {pvalue:.1f}%")