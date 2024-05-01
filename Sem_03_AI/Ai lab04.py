import math;
import numpy as np;
import pandas as pd;
import random;
# arrx=[1,3,8,10]
# arry=[2,5,12,15]
# print(np.mean(arrx))
# print(np.mean(arry))
# for i in range (4):
#     nr= (arrx[i]-np.mean(arrx))*(arry[i]-np.mean(arry))
#     dr= (arrx[i]-np.mean(arrx))**2
#     m=nr/dr
#     print("m = ",m)
#     c=np.mean(arry)-m*np.mean(arrx)
#     print("c = ",c)

dataset = pd.read_csv("G:\Course notes\dataset.csv")
print(dataset)
mean_head_size = np.mean(dataset['Head Size(cm^3)'])
mean_brain_weight = np.mean(dataset['Brain Weight(grams)'])
n=0
d=0
arr1=dataset['Head Size(cm^3)']
arr2=dataset['Brain Weight(grams)']
for i in range(237):
  n+=((arr1[i]-mean_head_size)*(arr2[i]-mean_brain_weight))
  d+=(arr1[i]-mean_head_size)**2
m=n/d
print(m)

c=mean_brain_weight-m*mean_head_size
print(c)



