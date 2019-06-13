#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values


#Splitting the dataset into training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#Simple linear regression using gradient descent
theta0 = 0
theta1 = 0

#function to return  derivative of cost function.


def averageCost0():
    total = 0
    for (i,j) in zip(X_train,Y_train):
        total = total + theta0 + theta1*i - j
    return total/len(X_train)

def averageCost1():
    total = 0
    for (i,j) in zip(X_train,Y_train):
        total = total + (theta0 + theta1*i - j)*i
    return total/len(X_train)

cost0 = averageCost0()
cost1 = averageCost1()

#Here loop stops when cost0 and cost1 becomes approximately to 0
while((cost0 <-0.01 or cost0 > 0.01 ) and (cost1 <-0.01 or cost1 > 0.01 )):
    temp0 = theta0 - 0.01*cost0
    temp1 = theta1 - 0.01*cost1
    theta0 = temp0
    theta1 = temp1
    
    #print(cost0,cost1)

    cost0 = averageCost0()
    cost1 = averageCost1()

print(cost0,cost1)    
print(theta0,theta1)

error = 0

    
for i,j in zip(X_test,Y_test):
    error = error + abs((theta0 + theta1*i - j)*100/j)
error = error/len(X_train)
print(error/len(X_train))

temp = [] #It contanis predicted values

for i,j in zip(X_train,Y_train):
    temp.append(theta0 + theta1*i)
print(len(temp),len(X_train))

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, temp, color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, temp, color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
