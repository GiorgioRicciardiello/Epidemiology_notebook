"""
    Author: Giorgio Ricciardiello
    Code question 1
Write a simulation in that proves that the marginal probability for the 30th person is indeed 1 in 10. Please provide
your code below. Use at least 100,000 repeats in your final
simulation.
"""
# Set the total number of Hershey's Kisses
N <- 200
n_stars <- 20
n_non_stars <- N - n_stars
n_students <- 30
n_repeats <- 10000
star_count <- 0

set.seed(123)  # Setting seed for reproducibility

for (i in 1:n_repeats) {
  # Create the bag of 200 items (1= stars, 0=no stars)
  bag <- c(rep(1, n_stars), rep(0, n_non_stars))
  # Shuffle the bag
  bag <- sample(bag)
  # Simulate 30 draws from the bag
  drawn_items <- sample(bag, n_students)
  # Check if the 30th person draws a star
  if (drawn_items[n_students] == 1) {
    star_count <- star_count + 1
  }
}

# Calculate the marginal probability
marginal_probability <- star_count / n_repeats
cat("Marginal Probability for the 30th student drawing a star: ",
    round(marginal_probability, digits = 1), "\n")




