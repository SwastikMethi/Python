import random as rand;
import numpy as np;
import statistics as stats;
from math import comb;
# Q-1
# list = ["I","am","Studying","In","BU"]
# n = np.random.choice(list , size= 1500)
# print(n)
# sample_list = []
# for i in range(len(n)):
#     if n[i] == "BU":
#         sample_list.append(n[i+1])
#         print(sample_list)
#         print(len(sample_list))

# dict = {}
# for i in sample_list:
#     if i in dict:
#         dict[i] += 1
#     else:
#         dict[i] = 1
# print(dict)

# for i in dict:
#     dict[i] = dict[i]/len(sample_list)
# print(dict)

# Q-2
def calculate_probability(x):
    total_outcomes = 2**4
    num_ways = comb(4, x)
    probability = num_ways / total_outcomes
    return probability

for x in range(5):
    probability = calculate_probability(x)
    print(f"p(X = {x})={probability}")
        

