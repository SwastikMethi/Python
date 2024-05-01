import numpy as np;
import matplotlib.pyplot as plt;
import csv;
import pandas as pd;
Titanic_data = pd.read_csv("G:\Course notes\\Titanic\\train.csv")
print(Titanic_data)
passenger = Titanic_data['PassengerId'].value_counts()
print(passenger)
survived = Titanic_data['Survived'].value_counts()
print(survived)
no_male_psng = Titanic_data['Sex'].value_counts()
print(no_male_psng)

# plt.pie(survived,labels = ['survived','not survived'],autopct = '%1.1f%%',shadow = True,explode = (0.1,0), startangle = 90)
# plt.show()

# plt.bar(passenger.index,passenger.values)
# plt.ylabel('No. of passengers')
# plt.xlabel('Passenger ID')
# plt.title(passenger.name)
# plt.show()

# plt.plot(Titanic_data['Age'])
# plt.xlabel('Number of passengers')
# plt.ylabel('Age of passengers')
# plt.title('Age of passengers')
# plt.show()

# plt.hist(Titanic_data['Age'],bins = 10)
# plt.xlabel('Age of passengers')
# plt.ylabel('No. of passengers')
# plt.title('Age of passengers')
# plt.show()

# Titanic_data.boxplot(column = 'Age', by='Sex')
# plt.show()

# plt.scatter(Titanic_data['Age'],Titanic_data['Fare'])
# plt.xlabel('Age of passengers')
# plt.ylabel('Fare of passengers')
# plt.title('Age vs Fare')
# plt.show()