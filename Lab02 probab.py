import numpy as np;
import random;
import csv;
# random.seed(2)
# list = [random.randint(90,100) for i in range(1500)]
# # print(list)
# dict ={}
# for i in list:
#     if i in dict:
#         dict[i] += 1
#     else:
#         dict[i] = 1
# print(dict)

#create a function to compute probability of getting a number from an array

# array = [20,20,21,21,21,22,23,26,27,27]
# print(array)
# dict = {}
# for i in array:
#     if i in dict:
#         dict[i] += 1
#     else:
#         dict[i] = 1

# print(dict)

# for i in dict:
#     dict[i] = dict[i]/len(array)
# print(dict)

# # Q-3
# random.seed(30)
# list = ["I", "am", "writing", "a", "program"]
# newlist=np.random.choice(list , size=1500)
# print(newlist)
# dict ={}
# for i in newlist:
#      if i in dict:
#          dict[i] += 1
#      else:
#          dict[i] = 1
# print(dict)

random.seed(2)
list = [random.randint(10,200) for i in range(20)]
print(list)
with open('random.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Random Number"])
    for i in list:
        writer.writerow([i])

with open('random.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        print(row)









