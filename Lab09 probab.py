import numpy as np
import pandas as pd

np.random.seed(150)
random_numbers = np.random.normal(12,19,8500)
standard_normal_numbers = (random_numbers - 12)/19
a1 = 12-0.325*19
a2 = 12+0.325*19
b1 = 12-0.5*19
b2 = 12+0.5*19
c1 = 12-0.275*19
c2 = 12+0.275*19
percentage_between_0_325_sigma = np.sum(np.abs(standard_normal_numbers) <= 0.325) / len(standard_normal_numbers) * 100
print("Percentage of numbers between i:", percentage_between_0_325_sigma, "%")
percentage_between_0_5_sigma = np.sum(np.abs(standard_normal_numbers) <= 0.5) / len(standard_normal_numbers) * 100
print("Percentage of numbers between ii:", percentage_between_0_5_sigma, "%")
percentage_between_0_275_sigma = np.sum(np.abs(standard_normal_numbers) <= 0.275) / len(standard_normal_numbers) * 100
print("Percentage of numbers between iii:", percentage_between_0_275_sigma, "%")



#Q-2
import math
p_printer_I = 0.4
p_printer_II = 0.6
mean_printer_I = 2
lower_bound_printer_II = 0
upper_bound_printer_II = 5
p_B_given_A = 1 - math.exp(-(1 / mean_printer_I) * 1)
p_B_given_not_A = 1 / (upper_bound_printer_II - lower_bound_printer_II) 
p_B = (p_B_given_A * p_printer_I) + (p_B_given_not_A * p_printer_II)
p_A_given_B = (p_B_given_A * p_printer_I) / p_B
print("probability that it was printed by Printer I : " , p_A_given_B)


#Q-3
import scipy.stats as stats
mean = 75
std_dev = 10
z_score = (80 - mean) / std_dev
proportion_more_than_80 = 1 - stats.norm.cdf(z_score)
percentage_more_than_80 = proportion_more_than_80 * 100
print("Proportion of students who have secured more than 80% marks: {:.2f}%".format(percentage_more_than_80))
