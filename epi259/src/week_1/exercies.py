import numpy as np


median = np.median(a = [-8, -10, -12, -16, -18, -20, -21, -24, -26, -30, +4, +3, 0, -3, -4, -5, -11, -14, -15, -30])


np.std([4, 3, 0, -3, -4, -5, -11, -14, -15, -300], ddof=1)
print(median)

np.std([-8, -10, -12, -16, -18, -20, -21, -24, -26, -30], ddof=1)

# compute IQR
x = [4,3, 0, -3, -4, -5, -11, -14, -15, -300]
q75, q25 = np.percentile(x, [75 ,25])
iqr = q75 - q25

data = [4, 3, 0, -3, -4, -5, -11, -14, -15, -300]

# Calculate Q1 and Q3
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)

# Calculate IQR
iqr = q3 - q1

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", iqr)