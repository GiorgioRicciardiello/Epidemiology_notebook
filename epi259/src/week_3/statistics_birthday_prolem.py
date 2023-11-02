"""
in a class of 100 students, what is the probability that at least 4 people
will share the same birthday?
"""
import numpy as np
import random
from math import comb

def birthday_probability(num_students, num_matches):
    if num_matches < 2:
        return 0.0

    # Calculate the probability that no two students share a birthday (factorial)
    probability = 1.0
    for i in range(num_students):
        probability *= (365 - i) / 365  # Probability that the next student has a different birthday

    # Calculate the probability of at least one shared birthday
    probability = 1 - probability

    return probability

if __name__ == "__main__":
    num_matches = 4
    num_students = 30
    probability_matches = birthday_probability(num_students, num_matches)

    print(f"The probability at least {num_matches} share  birthday in {num_students} students is: \n{probability_matches}")



