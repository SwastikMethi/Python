# Q-1
p_y0 = p_y15 = 1/3
p_y5 = p_y10 = 1/5
p_x710 = (7*60 + 10 - 7*60)/(30*60 - 7*60)
p_x725 = (7*60 + 25 - 7*60)/(30*60 - 7*60)
p_x715 = (7*60 + 15 - 7*60)/(30*60 - 7*60)
p_at_least_12 = p_x715*p_y15 + (1 - p_x715)*p_y0
p_less_than_5 = p_x710*p_y0 + p_x725*p_y5 + (1 - p_x725)*p_y10
print("Probability of waiting less than 5 minutes for a bus:", p_at_least_12)
print("Probability of waiting at least 12 minutes for a bus:", p_less_than_5)

# Q-2
data = [0.25, 0.65, 0.10, 0.50, 0.80, 0.30, 0.70, 0.40, 0.20, 0.60]
min_value = min(data)
max_value = max(data)
pdf_min=1/(max_value-min_value)
cdf = []
for i in range(len(data)):
    cdf.append(pdf_min * (data[i] - min_value))
probability = cdf[6] - cdf[3]

print(f"Minimum value: {min_value:.2f}")
print(f"Maximum value: {max_value:.2f}")
print(f"PDF for values within the interval: {pdf_min:.2f}")
print(f"CDF for x = ",cdf)
print(f"Probability in the interval [0.4, 0.7]: {probability:.2f}")

# Q-3
import numpy as np
dp = [0.25, 0.65, 0.10, 0.50, 0.80, 0.30, 0.70, 0.40, 0.20, 0.60]
min_val = min(dp)
max_val = max(dp)
mean=(max_val+min_val)/2
variance=((max_val-min_val)**2)/12
outside_val=[]
for value in dp:
    if value < min_val or value > max_val:
        outside_val.append(value)
print(f"Mean value: {mean}")
print(f"Variance: {variance:.4f}")
print(f"Values in the dataset are outside the specified uniform distribution interval: {outside_val}")


















# def prob_less_than_5():
#     p_y0 = p_y15 = 1/4
#     p_y5 = p_y10 = 1/8
#     p_x710 = (7*60 + 10 - 7*60)/(30*60 - 7*60)
#     p_x725 = (7*60 + 25 - 7*60)/(30*60 - 7*60)
#     p_x = lambda x: 1/23 if 7*60 <= x <= 7*60 + 30 else 0
#     return p_x710*p_y0 + p_x725*p_y5 + (1 - p_x725)*p_y10

# def prob_at_least_12():
#     p_y0 = 1/4
#     p_y15 = 1/4
#     p_x715 = (7*60 + 15 - 7*60)/(30*60 - 7*60)
#     p_x = lambda x: 1/23 if 7*60 <= x <= 7*60 + 30 else 0
#     return p_x715*p_y15 + (1 - p_x715)*p_y0

# print("Probability of waiting less than 5 minutes:", prob_less_than_5())
# print("Probability of waiting at least 12 minutes:", prob_at_least_12())
