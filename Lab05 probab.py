import numpy as np
import random 
import pandas as pd
from itertools import product, repeat
## Q-1
possible_outcomes = list(product([1,2,3,4,5,6] , repeat = 3))
probab_distribution = {}

for i in possible_outcomes:
        outcome_sum = sum(i)
        if outcome_sum in probab_distribution:
            probab_distribution[outcome_sum] += 1
        else:
            probab_distribution[outcome_sum] = 1
print(probab_distribution)

for i in probab_distribution:
     probab_distribution[i]=probab_distribution[i]/216
print(probab_distribution)

## Q-2
total_students = 50
Data = {0:3, 1:8, 2:12, 3:15, 4:9, 5:3}
arr = np.array([[0,1,2,3,4,5],[3,8,12,15,9,3]])
mean_data = np.mean(list(Data.keys()))
print(mean_data)
variance_data = sum((x-mean_data)**2* f/total_students for x,f in Data.items())
print(variance_data)
for i in Data:
    if(i>=3):
        sum = sum + Data[i]
    else:
        sum = 0
print(sum/total_students)

## Q-3
data = {0:0.25, 1:0.5, 2:0.125, 3:0.125}
cdf = {}
cum_probab = 0
for x, prob in data.items():
    cum_probab += prob
    cdf[x] = cum_probab

num_samples = 1500
random_samples = []
for _ in range(num_samples):
    random_number = random.random()
    for x, cum_prob in cdf.items():
        if random_number <= cum_prob:
            random_samples.append(x)
            break
print(random_samples)


