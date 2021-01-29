import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("/Users/nithinkyatham/Downloads/50_Startups.csv")

X=dataset.iloc[:,:-1].values

Y=dataset.iloc[:,4].values

#Handling Categorical Data

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransfer = ColumnTransformer([("State", OneHotEncoder(), [3])], remainder = 'passthrough')
X = columntransfer.fit_transform(X)
X = X.astype('float64')
X = X[:,1:]

#Data Splitting to Train and Test Data

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


sc_y = StandardScaler()
Y_train = np.array(Y_train).reshape(len(Y_train),1)
Y_train = sc_y.fit_transform(Y_train)

#Training Regression Model Algorithm using Train Data

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Making Predictions for Test Data

Y_pred = regressor.predict(X_test)

#Reversing the Feature Scaling

Y_pred = sc_y.inverse_transform(Y_pred)

b = regressor.intercept_
print(b)

c = regressor.coef_
print(c)

import math
from sklearn.metrics import mean_squared_error

# Mean Squared Error
MSE = mean_squared_error(Y_test, Y_pred)

# Root Mean Squared Error
RMSE = math.sqrt(MSE)

"""# Implementing Backward Elimination to find impactful columns

import statsmodels.regression.linear_model as sm

# Adding a column of 1's to existing matrix of IV

X = np.append(arr=np.ones(shape=(50,1)), values=X, axis=1)

#Creating a new matrix of IV which will be gradually optimized

X_opt = X[:,[0,3,5]]

# Creating a new type of regressor object from OLS method of statsmodels

regressor_OLS = sm.OLS(exog=X_opt,endog=Y).fit()

regressor_OLS.summary()"""


