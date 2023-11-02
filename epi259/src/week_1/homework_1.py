import numpy as np
import matplotlib.pyplot as plt

# %% Question One - frequency count
# Define the data
data = [5] * 3 + [6] * 8 + [7] * 20 + [8] * 11 + [9] * 6 + [10] * 2

assert len(data) == 50

# Calculate the mean
mean = np.mean(data)

# Calculate the standard deviation
std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

# Calculate the median
median = np.median(data)

# Calculate the interquartile range
q75, q25 = np.percentile(data, [75 ,25])
interquartile_range = q75 - q25

# Print the results
print("Mean:", round(mean, 1))
print("Standard Deviation:", round(std_dev, 2))
print("Median:", int(median))
print("Interquartile Range:", int(interquartile_range))


# Create a histogram
plt.hist(data, bins=5, edgecolor='black')  # You can adjust the number of bins as needed
plt.title("Histogram of Hours Slept per Night")
plt.xlabel("Hours per Night")
plt.ylabel("Frequency")
plt.show()


# question 6 - which bar plot has the largest variance

rect_histogram = {4.2:16, 6:16, 8:16, 20:16}



# %% Question Two - binary data
# Define the data
data = [0] * 47 + [1] * 53

assert len(data) == 100

# Calculate the mean
mean = np.mean(data)

# Calculate the standard deviation
std_dev = np.std(data, ddof=1)  # ddof=1 for sample standard deviation

# Calculate the 90th percentile
percentile_90 = np.percentile(data, 90)

# Print the results
print("Mean:", round(mean, 2))
print("Standard Deviation:", std_dev,)
print("90th Percentile:", int(percentile_90))


# question 3
# Descriptive statistics and analyses were performed according to whether the variable was continuous or
# categorical. Depending on distribution of
#   continuous data, variables were expressed as mean values ± standard deviation (SD)
#   or median values with interquartile range (IQR),
# and compared using the Students t-test (normally
# distributed data) or Wilcoxon Rank Sum (non-normally distributed data).
# Categorical data were expressed as
# frequencies and percentages and compared using the χ2 test or Fisher exact test.
#
# age: continuous
# bmi: continuous
# smoking: Ordinal
# waist cm: continuous
# cd4: continuous
# art: continuous
# viral supression:continuous

cd4_median = 529.5
cd4_q3 = 686.5
cd4_q1 = 372.0