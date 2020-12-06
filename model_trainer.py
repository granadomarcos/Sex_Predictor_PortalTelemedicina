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


dataset = open(sys.argv[1], 'rb')

dataset = pd.read_csv(ofile, sep=',')

#Checando valores Missing
dataset.isnull().sum()

#Delete missing values
dataset.dropna(inplace=True)

def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0

# Remove features
remove_features(['index'])


# Separa a classe dos dados
classes = dataset['sex']
dataset.drop('sex', axis=1, inplace=True)

# Treinando o algoritmo de SVM.
clf = svm.SVC().fit(dataset,classes)

# Utiliza a função train_test_split para separar conjunto de treino e teste em 80/20

X_train, X_test, y_train, y_test = train_test_split(dataset, classes, test_size=0.2, random_state=123)

# Scala os dados de treino e teste.
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

scaler2 = StandardScaler().fit(X_test)
X_test = scaler2.transform(X_test)

# Treina o algoritmo
clf.fit(X_train, y_train)

# Resultados de predição.
y_pred  = clf.predict(X_test)

joblib.dump(clf, 'model.pkl')


