import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("G:\Course notes\dataset.csv")
print(data.size)
print(data.shape)
print(data.ndim)

a = data["Age Range"]
b = data["Brain Weight(grams)"]
c = data["Head Size(cm^3)"]
plt.scatter(c,b)
plt.show()

plt.hist(c)
plt.show()

plt.bar(c,b)
plt.show()

plt.pie(a)
plt.show()

plt.plot(a,b)
plt.show()

