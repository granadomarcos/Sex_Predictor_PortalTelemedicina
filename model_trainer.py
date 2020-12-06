'''
Imports
'''
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import _pickle as cPickle
import joblib
import sys

'''
Open test data
'''
dataset = open(sys.argv[1], 'rb')

dataset = pd.read_csv(ofile, sep=',')

'''
Data preprocessing
'''

#Cheching missing values
dataset.isnull().sum()

#Delete missing values
dataset.dropna(inplace=True)


def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0

# Remove index feature due the non usability for model development
remove_features(['index'])


# Split features and predictor class
classes = dataset['sex']
dataset.drop('sex', axis=1, inplace=True)


# Training a SVM algorithm.
clf = svm.SVC().fit(dataset,classes)

# Split train and test data in 80/20

X_train, X_test, y_train, y_test = train_test_split(dataset, classes, test_size=0.2, random_state=123)

# Scale train and test data.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler2 = StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

# Train a model
clf.fit(X_train, y_train)

'''

# Resultados de predição.
y_pred  = clf.predict(X_test)

'''

# Persist model in disk
joblib.dump(clf, 'model.pkl')


