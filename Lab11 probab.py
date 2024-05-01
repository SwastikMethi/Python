import numpy as np
import pandas as pd
import math
import scipy.stats as stats
import random
import matplotlib.pyplot as plt

#Q-1
dataset = pd.read_csv("G:\Course notes\dataset_porbab.csv")
num_of_samples = 1000
sample_size = 40
mean_of_samples = []
for i in range(num_of_samples):
    sample = np.random.randint(dataset.shape[0], size=sample_size)
    mean = np.mean(dataset.iloc[sample, 0])
    mean_of_samples.append(mean)

plt.hist(mean_of_samples, bins=30, alpha=0.5, edgecolor='black', color='cyan', label='Mean of samples')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

num_of_samples_02 = 2000
sample_size_02 = 150
mean_of_samples_02 = []
for i in range(num_of_samples_02):
    sample_02 = np.random.randint(dataset.shape[0], size=sample_size_02)
    mean_02 = np.mean(dataset.iloc[sample_02, 0])
    mean_of_samples_02.append(mean_02)

plt.hist(mean_of_samples_02, bins=30, alpha=0.5, edgecolor='black', color='cyan', label='Mean of samples')
plt.xlabel('Value')
plt.title('Distribution of mean of samples')
plt.ylabel('Frequency')
plt.show()



# Q-2
sample_size = 40
num_of_samples = 1000
mean_of_samples = []


for i in range(num_of_samples):
    normal_distribution = np.random.normal(18, 20, sample_size)
    mean = np.mean(normal_distribution)
    mean_of_samples.append(mean)

plt.hist(mean_of_samples, bins=30, alpha=0.5, edgecolor='black', color='cyan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Normal Distribution')
plt.show()


for i in range(num_of_samples):
    poisson_distribution = np.random.poisson(10, sample_size)
    mean = np.mean(poisson_distribution)
    mean_of_samples.append(mean)

plt.hist(mean_of_samples, bins=30, alpha=0.5, edgecolor='black', color='cyan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Poisson Distribution')
plt.show()


for i in range(num_of_samples):
    Exponential_distribution = np.random.exponential(20, sample_size)
    mean = np.mean(Exponential_distribution)
    mean_of_samples.append(mean)

plt.hist(mean_of_samples, bins=30, alpha=0.5, edgecolor='black', color='cyan')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Exponential Distribution')
plt.show()



# Q-3
mean = 75
std_dev = 25
sample_size = 110
sample_mean = 82
standard_error = std_dev / (sample_size ** 0.5)
a = (sample_mean - mean) / standard_error
probability = 1 - stats.norm.cdf(a)
print(probability)





    