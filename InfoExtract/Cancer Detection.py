import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score

import json
import nltk
import ssl

''' SSl Certificate'''
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

import urllib.request


# Loading Dataset from url to DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
col_names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
df = pd.read_csv(url, names=col_names)
print(df)

# Print the shape of the DataFrame
print(df.shape)

'''Data Preprocessing'''
df.replace('?',-99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)

# Print the shape of the new DataFrame
print(df.shape)

# Assigning X and Y from datasets for training
X = np.array(df.drop(['class'], axis=1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Defining models to train
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC(gamma='auto')))

#Testing for accuracy
seed = 8
scoring = 'accuracy'

# Evaluate each model ie KNN, SVM
results = []
names = []

for name, model in models:
       kfold = model_selection.KFold(n_splits = 10, shuffle=True, random_state = seed)
       # Evaluate score by Cross-Validation
       cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
       results.append(cv_results)
       names.append(name)
       msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
       print(msg)

# Make predictions on validation dataset
for name, model in models:
       model.fit(X_train, y_train)
       predictions = model.predict(X_test)
       print(name)
       print(accuracy_score(y_test, predictions))
       print(classification_report(y_test, predictions))

clf = SVC(gamma='auto') # create support-vector-classifier

# get accuracy score for it
clf.fit(X_train,y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

# Prediction for any example
example = np.array ([[4,2,1,1,1,2,3,2,1]])

example = example.reshape(len(example), -1) #reshape to get a column vector
prediction = clf.predict(example)
#print(prediction)
if prediction==4:
       print('Malignant\n')
elif prediction==2:
       print('Benign\n')

