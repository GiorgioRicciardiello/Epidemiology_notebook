"""
To find the
�
z value in Python for a given alpha (significance level) for both one-sided and two-sided tests using the standard normal distribution, you can use the scipy.stats module.

For a two-sided test, where you want to find the critical z-value for a given alpha level split equally between the tails of the distribution, you'd use norm.ppf from scipy.stats.

For a one-sided test, the z-value will differ depending on whether it's a left-tailed or right-tailed test. If it's a left-tailed test, you would use norm.ppf with alpha directly. For a right-tailed test, you'd subtract alpha from 1 and then use norm.ppf.
"""
from scipy.stats import norm

def two_sided_z_value(alpha):
    # Divide alpha by 2 for a two-sided test
    alpha /= 2
    # Calculate z-value
    z = norm.ppf(1 - alpha)
    return z

def left_sided_z_value(alpha):
    # For left-sided test, use alpha directly
    z = norm.ppf(alpha)
    return z

def right_sided_z_value(alpha):
    # For right-sided test, use (1 - alpha)
    z = norm.ppf(1 - alpha)
    return z

# Example usage:

if __name__ == "__main__":
    alpha = 0.05 # Set your significance level
    z_two_sided = two_sided_z_value(alpha)
    z_left_sided = left_sided_z_value(alpha)
    z_right_sided = right_sided_z_value(alpha)

    print(f"Two-sided Z-value for alpha = {alpha}: {z_two_sided}")
    print(f"Left-sided Z-value for alpha = {alpha}: {z_left_sided}")
    print(f"Right-sided Z-value for alpha = {alpha}: {z_right_sided}")
