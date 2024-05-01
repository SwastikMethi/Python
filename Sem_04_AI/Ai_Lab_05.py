import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("g:\\Course notes\\breast-cancer.csv")
X = data.drop('texture_mean',axis=1)
y = data['id']
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)
## Linear Regression and Polynomial Regression models with degrees 2
model = LinearRegression()
model.fit(X_train, y_train)
print(model.coef_)
print(model.intercept_)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, y_pred, color='blue')
plt.show()

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly.fit(X_poly, y)
model = LinearRegression()
model.fit(X_poly, y)
print(model.coef_)







