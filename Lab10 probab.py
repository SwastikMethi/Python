import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

#Q-1
np.random.seed(100)
random_marks = np.random.normal(527,112,100)
probability_of_student_scoring_more_than_500 = np.sum(random_marks >= 500) / len(random_marks)
probability_above_500 = 1-stats.norm.cdf(500,527,112) 
# print("Probability of a student scoring more than 500: {:.2f}".format(probability_of_student_scoring_more_than_500))
print("Probability of a student scoring more than 500: {:.4f}".format(probability_above_500))
Top_5_percent = stats.norm.ppf(1-0.05,527,112)
print("Marks scored by the top 5% of the students: {:.4f}".format(Top_5_percent))
probability_of_an_individual_scoring_between_527_to_554= stats.norm.cdf(554,527,112) - stats.norm.cdf(527,527,112)
print("Probability of an individual scoring between 527 and 554: {:.4f}".format(probability_of_an_individual_scoring_between_527_to_554))

#Q-2
_lambda_ = 2
more_than_3 = 1 - stats.expon.cdf(3, scale=1/_lambda_)
at_least_1 = (1 - stats.expon.cdf(3, scale=1/_lambda_)) / (1 - stats.expon.cdf(2, scale=1/_lambda_))
print(f"Probability that shower will last more than three minutes: {more_than_3:.4f}")
print(f"Probability that shower will last at least one more minute given that it has already lasted for 2 minutes: {at_least_1:.4f}")

#Q-3
lambda_1 = 0.0003
lambda_2 = 0.00035
time = 10000
prob_10000_hours_1 = 1 - stats.expon.cdf(time, scale=1 / lambda_1)
prob_10000_hours_2 = 1 - stats.expon.cdf(time, scale=1 / lambda_2)
print(f"a) Proportion of fans which will give at least 10000 hours service for lambda = 0.0003: {prob_10000_hours_1:.4f}")
print(f"b) Proportion of fans which will give at least 10000 hours service for lambda = 0.00035: {prob_10000_hours_2:.4f}")

#Q-4
np.random.seed(100)
sample=np.random.normal(7,8,9000)
logged_sample = np.log(np.abs(sample))
logged_mean = np.mean(logged_sample)
logged_std = np.std(logged_sample)
plt.figure(figsize=(10, 6))
plt.hist(logged_sample, bins=30, color='red', edgecolor='cyan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
stat, p = stats.shapiro(sample)
print(f'Mean after log: {logged_mean:.4f}')
print(f'STD after log: {logged_std:.4f}')
if p > 0.05:
    print("The data appears to be normally distributed.")
else:
    print("The data does not appear to be normally distributed.")