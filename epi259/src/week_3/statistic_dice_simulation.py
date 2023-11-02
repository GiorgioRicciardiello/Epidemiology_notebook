"""
if you roll 10 dices, what is the probability that the same number will come up at least 5 times
"""
import random

def roll_dice(num_dice_roll: int = 10) -> list[int]:
    return [random.randint(1, 6) for _ in range(num_dice_roll)]


def has_at_least_n_occurrences(rolls:list[int], num_occurrences:int=5):
    # Count the occurrences of each number from 1 to 6
    count = [rolls.count(i) for i in range(1, 7)]
    # Check if at least one number occurs n or more times
    return any(count[i] >= num_occurrences for i in range(6))


if __name__ == "__main__":
    # Number of simulations
    num_simulations = 100000
    num_dice = 10
    count_successes = 0

    for _ in range(num_simulations):
        # get the results of the 1 rolls
        rolls: list[int] = roll_dice(num_dice_roll=10)
        if has_at_least_n_occurrences(num_occurrences=5, rolls=rolls):
            count_successes += 1

    probability = count_successes / num_simulations

    print(f"Probability of the same number coming up at least 5 times in 10 dice rolls: {probability:.4f}")
