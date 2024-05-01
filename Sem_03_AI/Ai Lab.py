import numpy as np;
import math;
import csv;
import pandas as pd;
import random;
import statistics;
import matplotlib.pyplot as plt;
# from scipy.stats import mode;

#Question-1
# a = np.array([345627,"raju"])
# b = np.array([[435,"Laxmi chit fund"]])
# c = np.array([[[1,"apple"],[2,"pie"],],[[1,"job"],[2,"startup"]],[[3,"mango"],[3,"man"]]])
# print("1st array")
# print(a)
# print(a.shape)
# print(a.dtype)
# print(a.ndim)
# print(a[0].size)
# print("2nd array")
# print(b)
# print(b.shape)
# print(b.dtype)
# print(b.ndim)
# print(b[0,0].size)
# print("3rd array")
# print(c)
# print(c.shape)
# print(c.dtype)
# print(c.ndim)
# print(c[0,1].size)
# arr= np.array([[2,3,4],[4,5,6]])
# newarr = arr.reshape(3,2)
# print(arr)
# print(newarr)
# print(newarr.shape)

#Question-2
# arr1=np.array([4,5,6,3,9,7])
# print(arr1[2:5])

# arr2=np.array([[1,2,3],[4,5,6]])
# print(arr2[0,0:2])

#Question-3 (a)
# ar1 = np.array([4,5,6,7])
# ar2 = np.array([3,4,1])
# concatinated_arr = np.concatenate([ar1,ar2])
# print(concatinated_arr)

# #(b)
# array1 = np.array([[1,2,3],[4,5,6]])
# array2 = np.array([[3,2,1],[6,5,4]])


# #(c)
# array1= np.array([4,5,6,3,9,10])
# array2= np.split(array1,3)
# print(array2)

# with open("G:\Course notes\wine+quality\winequality-red.csv", mode = 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)
#     for column in reader:
#         print(column)

# with open("G:\Course notes\wine+quality\winequality-white.csv", mode = 'r') as file:
#     reader = csv.reader(file)
#     for row in reader:
#         print(row)
#     for column in reader:
#         print(column)


#Lab - 02
data01 = pd.read_csv("G:\Course notes\wine+quality\winequality-red.csv", sep = ";")

data02 = pd.read_csv("G:\Course notes\wine+quality\winequality-white.csv", sep = ";")

XY_Red = data01.to_numpy()
XY_White = data02.to_numpy()
# print(XY_Red)
# print(XY_White)
# print(XY_Red.shape)
# print(XY_White.shape)
Y=XY_Red[:,-1]
X=XY_White[:,:-1]
# print(Y,Y.shape)
# print(X,X.shape)
np.random.shuffle(XY_Red)
aT = np.transpose(XY_Red)
print(aT)
print(aT.shape)
print(np.min(X,axis=0))
print(np.max(X,axis=0))
print(np.min(X,axis=1))
print(np.max(X,axis=1))
print(np.count_nonzero(Y == 5))


print(np.mean(X,axis=0))
mode = statistics.mode(Y)
print(mode)

stddev = np.std(X,axis=0)
print(stddev)



















    


