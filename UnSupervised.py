import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split


dataset = pd.read_csv("/Users/nithinkyatham/Downloads/Data.csv")

X=dataset.iloc[:,:-1].values

Y=dataset.iloc[:,3].values

'''#Handling Missing Values

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])


#Handling Categorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

columntransfer = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = columntransfer.fit_transform(X)
X = X.astype('float64')

labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)'''

#Splitting the Dataset into Training and Testing sets

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


#Feature Scaling

sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


