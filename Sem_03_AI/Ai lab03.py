import numpy as np;
import matplotlib.pyplot as plt;
import csv;
import pandas as pd;
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
import copy
import random
from numpy import nan
Dataset = pd.read_csv("G:\Course notes\\Titanic\\train.csv")
# print(Dataset)
Dataset.info()
temp1 = copy.copy(Dataset)
mean_age = temp1['Age'].mean()
temp1['Age'] = temp1['Age'].fillna(mean_age)
# temp1.info()
temp2 = copy.copy(temp1)
mode_embarked = temp2['Embarked'].mode()
temp2['Embarked'] = temp2['Embarked'].fillna(mode_embarked[0])
# temp2.info()
# print(temp2['Embarked'])
temp3 = copy.copy(Dataset)
temp3['Age'] = temp3['Age'].fillna(random.randint(35,45))
temp3.info()
temp4 = copy.copy(Dataset)
# for i in range(0,891):
#     if(temp4['Age'][i] == np.nan):
#         temp4['Age']=temp4['Age'].fillna(random.randint(35,45))
#     else:
#         continue

for i in temp4['Age']:
    if(i == np.NAN):
        temp4['Age']=temp4['Age'].fillna(random.randint(35,45))
    else:
        continue
temp4.info()
