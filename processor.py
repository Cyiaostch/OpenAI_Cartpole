import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

features=np.load("features.npy")
labels=np.load("labels.npy")
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
with open('logistic.pkl', 'wb') as f:
    pickle.dump(classifier, f) 

