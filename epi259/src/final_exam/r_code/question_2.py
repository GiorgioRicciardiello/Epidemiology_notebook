library(ggplot2)
library(readxl)
library(MASS)
library(broom)

# Question 2
# Load data
root <- dirname(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
data_path <- file.path(root, 'rawdata', 'finaldata.xlsx')
data <- read_excel(data_path)
data$age <- as.numeric(as.character(data$age))  # Convert 'age' to numeric

# Report the mean, median, standard deviation, IQR, and N for all three variables
par(mfrow = c(1, 3), mar = c(5, 4, 2, 2))
for (var in names(data)) {
  boxplot(data[[var]], main = paste("Boxplot of", var), ylab = var)
  stats <- summary(data[[var]])
  text(1.5, max(data[[var]]) + 0.5, paste("Mean: ", round(stats$mean, 2), "\n",
                                           "Median: ", round(stats$median, 2), "\n",
                                           "SD: ", round(stats$sd, 2), "\n",
                                           "IQR: ", round(IQR(data[[var]]), 2), "\n",
                                           "N: ", length(data[[var]])), adj = 0.5, cex = 0.8)
}
par(mfrow = c(1, 1))  # Reset to single plot layout

# Evaluate the pairplot
pairplot(data)

# Evaluate whether age is related to depression
# Age is categorical and depression is continuous -> Kruskal-Wallis test
age_lvls <- unique(data$age)
kruskal_result <- kruskal.test(data$depression ~ factor(data$age))
print(kruskal_result)

# Fit ordinal logistic regression model
ordinal_model <- polr(factor(age) ~ depression, data = data, method = "logistic")
summary(ordinal_model)

# Evaluate whether depression is related to social media use
corr_dep_socm <- cor(data$depression, data$socialmedia)
# Fit the linear regression model
linear_model <- lm(socialmedia ~ depression, data = data)
summary(linear_model)
