import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("/Users/nithinkyatham/Downloads/Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values

Y = dataset.iloc[:,4].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# Implementing Classification using K Nearest Neighbour Algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=19)
classifier.fit(X_train, Y_train)

# Predict the data by passing Test Data
Y_pred = classifier.predict(X_test)

# Calculating the Prediction Accuracy using Confusion-Matrix Functionality
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
