# Question 5 - Violations of the homogeneity of variances assumption
# Compare the cholesterol levels of two groups with a two-sample T-Test with pooled variance

randomize_trial < - function(n_distr_a, n_distr_b, std_dev_a, std_dev_b, mean_dist_a, mean_dist_b,
                             distribution='normal')
{
if (distribution == 'normal')
{
    measure_cases < - rnorm(n_distr_a, mean_dist_a, std_dev_a)
measure_controls < - rnorm(n_distr_b, mean_dist_b, std_dev_b)
}
return (list(measure_cases=measure_cases, measure_controls=measure_controls))
}

# Structure the scenarios
scenarios < - data.frame(
n_a = c(50, 50, 50, 50, 50),
n_b = c(50, 50, 100, 100, 100),
sd_a = c(20, 10, 20, 10, 50),
sd_b = c(20, 50, 20, 50, 10),
effect_size = NA,
pvalues = NA,
Type_i_error = NA,
Type_ii_error = NA,
power = NA,
ground_truth = 'Ho True',
sign_lv = 0.05
)

# T-Test with pooled variance
equal_variance < - TRUE
SIMULATION_TYPE < - 'null_hypothesis'
sign_lv < - 0.05
n_trials < - 1000

if (SIMULATION_TYPE == 'null_hypothesis')
{
scenarios$effect_size < - 0  # We assume H0 is true, no difference in means
scenarios$Type_i_error < - NA
} else {
scenarios$effect_size < - sample(10: 40, nrow(
    scenarios), replace = TRUE)  # Assume artificial effect size between 10 and 40
scenarios$Type_ii_error < - NA
scenarios$ground_truth < - 'H1 True'
scenarios$power < - NA
}

scenarios$sign_lv < - sign_lv

# Set up the layout for the plots
par(mfrow=c(3, 2), mar=c(4, 4, 2, 2))

for (idx_ in seq(nrow(scenarios))) {
    p_values < - numeric(n_trials)
    for (trial in seq(n_trials)) {
        trial_data < - randomize_trial(
            n_distr_a=scenarios$n_a[idx_],
    n_distr_b = scenarios$n_b[idx_],
    std_dev_a = scenarios$sd_a[idx_],
    std_dev_b = scenarios$sd_b[idx_],
    mean_dist_a = 220,
    mean_dist_b = 220 + scenarios$effect_size[idx_]
    )

    p_value < - t.test(
    x = trial_data$measure_cases,
    y = trial_data$measure_controls,
    alternative = 'two.sided',
    var.equal = equal_variance
    )$p.value
    p_values[trial] < - p_value
    }

    scenarios$pvalues[idx_] < - toString(p_values)

    if (SIMULATION_TYPE == 'null_hypothesis') {
    # Under H0 True
    # Percent of times that we got a p-value below alpha -> significant when it should not be
    scenarios$Type_i_error[idx_] < - sum(p_values < sign_lv) /length(p_values)
    } else {
    # Under H1 True
    # Percent of times that we got a p-value above alpha -> non-significant when it should be
    scenarios$Type_ii_error[idx_] < - sum(p_values > sign_lv) /length(p_values)
    scenarios$power[idx_] < - (1 - scenarios$Type_ii_error[idx_]) * 100
    }

    # Plot histograms for p-value frequency in each scenario
    hist(p_values, col='skyblue', main=paste('Scenario', idx_ + 1), xlab='P-values', ylab='Frequency')
}

# Reset the layout
par(mfrow=c(1, 1))

# Print the results of each scenario in a single table
result < - scenarios[
    c('n_a', 'n_b', 'sd_a', 'sd_b', 'effect_size', 'Type_i_error', 'Type_ii_error', 'power', 'ground_truth', 'sign_lv')]
print(result)
