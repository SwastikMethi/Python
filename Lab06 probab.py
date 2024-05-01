import numpy as np
import pandas as pd
import random
## Q-1
x_data = []
y_data = []
for i in range(100):
    x_data.append(random.randint(1,100))
for i in range(100):
    y_data.append(random.randint(1,100))

mean_x_data = np.mean(x_data)
mean_y_data = np.mean(y_data)
standard_daviation_x_data = np.std(x_data)
standard_daviation_y_data = np.std(y_data)
covarience_x_data = np.cov(x_data)
covarience_y_data = np.cov(y_data)

print("mean of x_data =", mean_x_data)
print("mean of y_data =", mean_y_data)
print("standard daviation of x_data =", standard_daviation_x_data)
print("standard daviation of y_data =", standard_daviation_y_data)
print("covarience of x_data=", covarience_x_data)
print("covarience of y_data=", covarience_y_data)
print("corelation of X_data=", np.corrcoef(x_data))
print("corelation of Y_data=", np.corrcoef(y_data))
corelation_x_data_v2= covarience_x_data/(standard_daviation_x_data*standard_daviation_x_data)
corelation_y_data_v2= covarience_y_data/(standard_daviation_y_data*standard_daviation_y_data)
print("corelation of X_data_v2=", corelation_x_data_v2)
print("corelation of Y_data_v2=", corelation_y_data_v2)


## Q-2
x_data = []
y_data = []
for i in range(1000):
    x_data.append(random.randint(1,1000))
for i in range(1000):
    y_data.append(random.randint(1,1000))
mean_x_data = np.mean(x_data)
mean_y_data = np.mean(y_data)
covarience_x_data = np.cov(x_data)
covarience_y_data = np.cov(y_data)
print("mean of x_data =", mean_x_data)
print("mean of y_data =", mean_y_data)
print("covarience of x_data=", covarience_x_data)
print("covarience of y_data=", covarience_y_data)

## Q-3
def y(x):
    return (x + 1)**2
x_pmf = {-2: 0.125,-1: 0.25,0: 0.25,1: 0.25,2: 0.25}
range_y = list(set(y(k) for k in x_pmf.keys()))
y_pmf = {}
for i in range_y:
    y_pmf[i] = 0
for j in x_pmf.keys():
    y_pmf[y(j)] += x_pmf[j]
print("Range of Y:", range_y)
print("PMF of Y:",y_pmf)



