# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 17:21:09 2019

@author: kdandebo
"""
#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#Importing data
df = pd.read_csv("C:/Users/kdandebo/Desktop/Models/Python excercise/Capstone/housing.csv")
#df=D1.parse("Calculations")

#a = df['Produce Name'].head(30)
#print(df['Produce Name'].head(5))
#b = df['Actual Boxes'].head(30)
F = df.head(500)
#print(F)
#print(df.columns)
#df.info()

# The independent variables are denoted as X
X = F[['lotsize', 'bedrooms', 'bathrms',
               'stories', 'garagepl']]

# the dependent variable which is denoted by Y, has been assinged to price , as it depends on all the variables mentioned above

y = F['price']

#   trying to create a model which generalises our data, so we divide our data into traiing data and test data

from sklearn.model_selection import train_test_split
# we Apply our linear regression algorithm on the training data and test it on our test data
# test data = 0.5 represents the data ratio , 50% to traninig and 50% to test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
# It fits the linear regression model on the training data.
lm.fit(X_train,y_train)

# Now lets predict on the test data, which is the main purpose of regression
predictions = lm.predict(X_test)
plt.scatter(y_test,predictions)
plt.show()
