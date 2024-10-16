# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Start the Program

Step 2: Import the required libraries.

Step 3: Upload and read the dataset

Step 4: Check for any null values using the isnull() function.

Step 5: From sklearn.tree import DecisionTreeRegressor.

Step 6: Find the accuracy of the model and predict the required values by importing the required module from sklearn. 

Step 7: End the program

## Program:


#### Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
#### Developed by: JANANI.V.S
#### RegisterNumber: 212222230050
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics


data= pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

le=LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
y=data["Salary"]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 2)

dt =DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

mse= metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])


```

## Output:

### data
![image](https://github.com/user-attachments/assets/39efac5c-e618-4678-b001-578e199de5a6)

### data.info() and data.isnull().sum()
![image](https://github.com/user-attachments/assets/92de4c5a-d46c-4363-ad4d-0f62dadfbb6a)


### Mean Squared Error
![image](https://github.com/user-attachments/assets/c84f4a98-ca23-4d09-b8c7-f34881596b66)

### R2 Score
![image](https://github.com/user-attachments/assets/2f70c445-4914-43e1-9944-d191af4db25d)

### Prediction
![image](https://github.com/user-attachments/assets/249abefb-d4c8-4500-b30a-e9fa079f14a5)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
