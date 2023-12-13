# Question 3
# The outcome in Campo is the Drop Jump (DJ).
# Define the effect size function
effect_size <- function(means_pg, means_cg, std_pool) {
  return((means_pg - means_cg) / std_pool)
}

# Define the pooled standard deviation function
pooled_std <- function(n1, s1, n2, s2) {
  num <- (n1 - 1) * s1^2 + (n2 - 1) * s2^2
  den <- n1 + n2 - 2
  return(sqrt(num / den))
}

# Measured data for the PG and the CG
campo_drop_dj <- data.frame(
  Group = rep(c('PG', 'CG'), each = 4),
  T_measure = rep(1:4, times = 2),
  Mean = c(24.9, 28.2, 28.9, 29.4, 27.1, 24.6, 25.6, 25.2),
  Std = c(1.1, 0.9, 0.8, 1.0, 1.0, 0.8, 0.9, 0.8)
)

# Print the dataframe
print(campo_drop_dj)

# Calculate mean change (difference within measures) for each group
mean_change_pg <- diff(subset(campo_drop_dj, Group == 'PG')$Mean)
mean_change_cg <- diff(subset(campo_drop_dj, Group == 'CG')$Mean)

# Calculate pooled standard deviation using the standard deviation from the final measurement.
std_pool <- pooled_std(
  n1 = 10,
  s1 = tail(subset(campo_drop_dj, Group == 'PG')$Std, 1),
  n2 = 10,
  s2 = tail(subset(campo_drop_dj, Group == 'CG')$Std, 1)
)

# Calculate the effect size
effect_size_value <- effect_size(
  means_pg = mean(mean_change_pg),
  means_cg = mean(mean_change_cg),
  std_pool = std_pool
)

# Confidence interval for the effect size
alpha <- 0.05
t_critical <- qt(1 - alpha / 2, df = length(mean_change_pg) - 1)
margin_of_error <- t_critical * sqrt((std_pool^2) / length(mean_change_pg))

confidence_interval <- c(
  effect_size_value - margin_of_error,
  effect_size_value + margin_of_error
)
confidence_interval <- round(confidence_interval, 3)

cat(paste("Effect Size:", round(effect_size_value, 3), "; CI(95%) [", confidence_interval, "]\n"))

# Simulate the error made by Campo
# Define the baseline and last measurement values for PG and CG
n1 <- 10
baseline_pg_measure_mean <- 24.9
baseline_pg_measure_std <- 1.1
baseline_pg_measure_se <- baseline_pg_measure_std / sqrt(n1)
last_pg_measure_mean <- 29.4
last_pg_measure_std <- 1.0
last_pg_measure_se <- last_pg_measure_std / sqrt(n1)

n2 <- 10
baseline_cg_measure_mean <- 27.1
baseline_cg_measure_std <- 1.0
baseline_cg_measure_se <- baseline_cg_measure_std / sqrt(n2)
last_cg_measure_mean <- 25.2
last_cg_measure_std <- 0.8
last_cg_measure_se <- last_cg_measure_std / sqrt(n2)

# Pooled from the last observation in PG and CG group
std_pooled <- pooled_std(
  n1 = n2,
  s1 = last_pg_measure_std,
  n2 = n2,
  s2 = last_cg_measure_std
)

# Mean change is the change from baseline to final
effect_size_value <- effect_size(
  means_pg = diff(c(baseline_pg_measure_mean, last_pg_measure_mean)),
  means_cg = diff(c(baseline_cg_measure_mean, last_cg_measure_mean)),
  std_pool = std_pooled
)

print(effect_size_value)
