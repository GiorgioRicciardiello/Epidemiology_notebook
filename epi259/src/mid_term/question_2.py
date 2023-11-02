"""
    Author: Giorgio Ricciardiello
    Code question 1
Write a simulation in that proves that the marginal probability for the 30th person is indeed 1 in 10. Please provide
your code below. Use at least 100,000 repeats in your final
simulation.
"""
import numpy as np
import random
if __name__ == "__main__":
    # get marginal probability of the last student drawing a star from the bag
    # Set the total number of Hershey's Kisses
    N = 200
    n_stars = 20
    n_non_stars = N - n_stars
    n_students = 30
    n_repeats = 10000
    star_count = 0
    for _ in range(n_repeats):
        # Create the bag of 200 items (1= stars, 0=no stars)
        bag = [1] * n_stars + [0] * n_non_stars
        # Shuffle the bag
        random.shuffle(bag)
        # simulate 30 draws from the bag
        drawn_items = random.sample(bag, n_students)
        # Check if the 30th person draws a star
        if drawn_items[-1] == 1:  # in the bag, 1= stars
            star_count += 1
    # Calculate the marginal probability
    marginal_probability = star_count / n_repeats
    print(f"Marginal Probability for the 30th student drawing a star: "
          f"{np.round(marginal_probability, 1)}")



