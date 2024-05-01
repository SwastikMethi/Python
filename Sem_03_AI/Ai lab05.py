import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data = pd.read_csv("G:\\Course notes\\bank-full-final (1).csv")
print(data.shape)
# for col in data.columns:
#     print(col)
# data.info()
df=pd.DataFrame(data)
# print(df.isnull())


x_train,x_test=train_test_split(data,test_size=0.3,train_size =0.7,random_state=0)
y_train,y_test=train_test_split(data,test_size=0.2,train_size =0.7,random_state=0)
# print(x_test)
# print(x_train)
# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

logistic = LinearRegression()
logistic.fit(x_train,y_train)
logistic.score(x_test,y_test)
