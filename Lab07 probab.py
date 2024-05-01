import pandas as pd
import numpy as np
from itertools import product, repeat
from math import comb;
import math

# Q-1
# Prob_drawing_fuse_bulb= 4/14
# Prob_drawing_good_bulb= 10/14
# n_draws = 3
# probability_dict = {}
# for i in range(n_draws+1):
#     binomial_coefficient = comb(n_draws, i)
#     probability = binomial_coefficient * (Prob_drawing_fuse_bulb ** i) * (Prob_drawing_good_bulb ** (n_draws-i))
#     probability_dict[i] = probability
# for k, probability in probability_dict.items():
#     print(f'P(X = {k}) = {probability:.4f}')

# Q-2
# prob_defective_component = 0.02
# n=7
# p_seven_defect = ((1 - n) ** (n - 1)) * prob_defective_component
# print("Probability of first defect caused by seventh component tested:",p_seven_defect)

# Expected_components = 1 / n
# print("Expected number of components:",Expected_components)

# Q-3
from scipy.stats import poisson
croissants_sold_per_hour = np.array([10, 15, 20, 25, 30])
days = np.array([2, 3, 4, 3, 2])
average_croissants = np.sum(croissants_sold_per_hour * days) / np.sum(days)
print("Average number of croissants sold per hour:", average_croissants)



lambda_parameter = average_croissants
prob_20_croissants = poisson.pmf(20, lambda_parameter)
print("Probability of selling exactly 20 croissants:", prob_20_croissants)



prob_15_croissants = 0
for i in range(15):
    prob_i_croissants = poisson.pmf(i, lambda_parameter)
    prob_15_croissants += prob_i_croissants
print("Probability of selling fewer than 15 croissants:", prob_15_croissants)
standard_deviation = np.sqrt(lambda_parameter)
print("Standard deviation of the number of croissants sold per hour:", standard_deviation)

# Q-4
from scipy.stats import poisson

rate = 4.5
prob_hold = 0.1

agents = 1

prob_excess = 1 - poisson.cdf(agents, rate)

while prob_excess > prob_hold:
    agents += 1
    prob_excess = 1 - poisson.cdf(agents, rate)

print("Minimum number of agents needed:", agents)







