# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import mode
# import math

# agp_series = [(2+(n-1)*4)*5**(n-1) for n in range(200)]

# population = []
# for i in range(200):
#     population.extend([agp_series[i]] * (i+1))

# log_means = []
# log_medians = []
# log_modes = []

# for day in range(1, 51):
#     sample = np.random.choice(population, size=day*10)
#     log_mean = np.log(sample.mean()/10**130)
#     log_median = np.log(np.median(sample)/10**130)

#     unique_values, counts = np.unique(sample, return_counts=True)
#     mode_index = np.argmax(counts)
#     mode_value = unique_values[mode_index] / 10 ** 130
#     log_mode = round(np.log(mode_value), 2)
#     log_modes.append(log_mode)

#     log_means.append(log_mean)
#     log_medians.append(log_median)
#     log_modes.append(log_mode)

# plt.bar(range(1, 51), log_means)
# plt.xlabel('Day')
# plt.ylabel('Log Mean')
# plt.title('Probability Distribution of Log Means Over 50 Days')
# plt.show()

# plt.bar(range(1, 51), log_medians)
# plt.xlabel('Day')
# plt.ylabel('Log Median')
# plt.title('Probability Distribution of Log Medians Over 50 Days')
# plt.show()

# plt.subplot(133)
# plt.bar(day, log_modes)
# plt.title("Log Mode")

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import  seaborn
def f(x):
    if x < 0:
        return 1 / 100
    elif x < 30:
        return np.exp(-x / 10)
    else:
        return 20 * np.exp(-20 * (x - 30))

def simulate_random_numbers(n_samples):
    samples = np.random.uniform(-100, 100, n_samples)
    probabilities = np.zeros(n_samples)

    for i, x in enumerate(samples):
        if x < 0:
            probabilities[i] = 1 / 100
        elif x < 30:
            probabilities[i] = np.exp(-x / 10)
        else:
            probabilities[i] = 20 * np.exp(-20 * (x - 30))

    return probabilities

def compute_mean_probability(n_samples):
    probabilities = simulate_random_numbers(n_samples)
    mean_probability = np.mean(probabilities) / 1e8
    return mean_probability

mean_probabilities = np.zeros(200)
for i in range(200):
    mean_probabilities[i] = compute_mean_probability(100)

plt.figure(figsize=(12, 6))


seaborn.distplot(mean_probabilities)
plt.xlabel('Mean Probability Value')
plt.ylabel('Number of Runs')
plt.title('Bar Plot of Mean Probability Values')

plt.legend()

plt.show()

import numpy as np
import matplotlib.pyplot as plt

first_term = 2
common_ratio = 5
common_difference = 4
n = 200

agp_series = [
    first_term + (i - 1) * common_difference * ((common_ratio) ** (i - 1))
    for i in range(n)
]

population = []
for i, term in enumerate(agp_series, start=1):
    population.extend([term] * i)

log_means = []
log_medians = []
log_modes = []
num_days = 50

for day in range(1, num_days + 1):
    num_samples = 10 * day

    samples = np.random.choice(population, size=num_samples)

    mean = np.mean(samples) / 10 ** 130
    log_mean = round(np.log(mean), 2)
    log_means.append(log_mean)

    median = np.median(samples) / 10 ** 130
    log_median = round(np.log(median), 2)
    log_medians.append(log_median)

    unique_values, counts = np.unique(samples, return_counts=True)
    mode_index = np.argmax(counts)
    mode_value = unique_values[mode_index] / 10 ** 130
    log_mode = round(np.log(mode_value), 2)
    log_modes.append(log_mode)

days = list(range(1, 51))

plt.figure(figsize=(12, 6))

plt.subplot(131)
plt.bar(days, log_means)
plt.title("Log Mean")

plt.subplot(132)
plt.bar(days, log_medians)
plt.title("Log Median")

plt.subplot(133)
plt.bar(days, log_modes)
plt.title("Log Mode")

plt.tight_layout()
plt.show()

