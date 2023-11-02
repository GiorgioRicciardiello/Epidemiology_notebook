"""
    Author: Giorgio Ricciardiello
    Code question 6
Simulation of an experiment with Binomial distribution with n=24 and p=0.5
"""
library(ggplot2)

# Function to calculate binomial probability
binomial_probability <- function(x, n, p) {
  return(dbinom(x, size = n, prob = p))
}

# Simulated data
recorded_data <- data.frame(
  infantid = 1:20,
  num_right_gaze = c(12, 11, 13, 12, 11, 12, 13, 12, 11, 11, 12, 12, 13, 14, 12, 10, 12, 12, 11, 13)
)
recorded_data$num_left_gaze <- 20 - recorded_data$num_right_gaze

# Set parameters for the binomial distribution
n <- 25
p_right <- 0.5
p_left <- 0.5
n_trials <- 1000

# Simulate the experiment
simulation_num_right_gaze <- rbinom(n_trials, size = n, prob = p_right)
simulation_num_left_gaze <- rbinom(n_trials, size = n, prob = p_left)

# Visualizations
par(mfrow = c(2, 2))
hist(recorded_data$num_right_gaze, breaks = seq(0, n + 1) - 0.5, col = "orange", main = "Right Gaze Distribution")
hist(recorded_data$num_left_gaze, breaks = seq(0, n + 1) - 0.5, main = "Left Gaze Distribution")
hist(simulation_num_right_gaze, breaks = seq(0, n + 1) - 0.5, col = "orange", main = "Simulated Right Gaze")
hist(simulation_num_left_gaze, breaks = seq(0, n + 1) - 0.5, main = "Simulated Left Gaze")

# Normality Test
shapiro.test(recorded_data$num_right_gaze)

# Q-Q Plot
qqnorm(recorded_data$num_right_gaze)
qqline(recorded_data$num_right_gaze)

# Empirical 68-95-99.7 Rule
mean_val <- mean(recorded_data$num_right_gaze)
std_dev <- sd(recorded_data$num_right_gaze)
within_one_std <- mean(abs(recorded_data$num_right_gaze - mean_val) <= std_dev)
within_two_std <- mean(abs(recorded_data$num_right_gaze - mean_val) <= 2 * std_dev)
within_three_std <- mean(abs(recorded_data$num_right_gaze - mean_val) <= 3 * std_dev)

# Print the percentages within each range
print(paste("Percentage within one standard deviation:", sprintf("%.2f%%", within_one_std * 100), "(Expected 68%)"))
print(paste("Percentage within two standard deviations:", sprintf("%.2f%%", within_two_std * 100), "(Expected 95%)"))
print(paste("Percentage within three standard deviations:", sprintf("%.2f%%", within_three_std * 100), "(Expected 99.7%)"))

# Calculate Q1 and Q3
q1 <- quantile(recorded_data$num_right_gaze, 0.25)
q3 <- quantile(recorded_data$num_right_gaze, 0.75)
