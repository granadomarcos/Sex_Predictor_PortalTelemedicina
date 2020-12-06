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
Read new sample data from command line
'''
dataset_pre = open(sys.argv[1], 'rb')

dataset_pre = pd.read_csv(ofile, sep=',')


'''
Data preprocessing
'''
#Checando valores Missing
dataset_pre.isnull().sum()

#Delete missing values
dataset_pre.dropna(inplace=True)

def remove_features(lista_features):
    for i in lista_features:
        dataset_pre.drop(i, axis=1, inplace=True)
    return 0

# Remove index feature
remove_features(['index'])

'''
Loading trained model from disk
'''
model = joblib.load('model.pkl')

'''
Predict new data
'''

df = pd.DataFrame({'sex':model.predict(dataset_pre)})

'''
Save csv file with new prediction
'''
df.to_csv('newsample_PREDICTIONS_MARCOSGRANADO_.csv', sep=',', index=False)