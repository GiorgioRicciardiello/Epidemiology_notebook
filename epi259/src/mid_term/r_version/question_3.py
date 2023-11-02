"""
    Author: Giorgio Ricciardiello
    Code question 3

Write a simulation in SAS or R that proves that the marginal  probability for the 30th person is indeed 1 in 10.
Please provide your code below. Use at least 100,000 repeats in your final simulation.
"""
# Load the data (assuming it's a CSV file)
data <- read.csv("data_path")

# Select non-missing rows and desired columns
data_plot <- subset(data, !is.na(data$`Crude Rate`), select = c("Year", "Crude Rate", "Age-Adjusted Rate"))

# Remove the total count of mortality
data_plot <- subset(data_plot, Year != "Total")

# Convert 'Year' column to integers to use x-limits
data_plot$Year <- as.integer(data_plot$Year)

# Load necessary libraries
library(ggplot2)

# Create the plot
lbl_fontsize <- 22
ticks_fontsize <- 20
legend_fontsize <- 18

# Plot the data
ggplot(data = data_plot, aes(x = Year)) +
  geom_line(aes(y = `Crude Rate`, color = "Crude Rate")) +
  geom_line(aes(y = `Age-Adjusted Rate`, color = "Age-Adjusted Rate")) +
  labs(x = "Years", y = "Mortality per 100,000") +
  scale_x_continuous(breaks = unique(data_plot$Year), labels = unique(data_plot$Year), expand = c(0, 0)) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(size = ticks_fontsize, angle = 45, hjust = 1),
    axis.text.y = element_text(size = ticks_fontsize),
    axis.title.x = element_text(size = lbl_fontsize),
    axis.title.y = element_text(size = lbl_fontsize),
    legend.text = element_text(size = legend_fontsize)
  )




