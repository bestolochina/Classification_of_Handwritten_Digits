# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow.keras.datasets import mnist
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_predict = model.predict(features_test)
    score = accuracy_score(target_test, y_predict)
    print(f'Model: {model}\nAccuracy: {score}\n')


path = r"C:\Users\aaa\datasets\mnist.npz"
with np.load(path) as data:
    X = data['x_train']
    y = data['y_train']

X, y = X[0:6000], y[0:6000]
X = np.reshape(X, (X.shape[0], -1))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

models = [KNeighborsClassifier(),
          DecisionTreeClassifier(random_state=40),
          LogisticRegression(random_state=40),
          RandomForestClassifier(random_state=40)]

for model in models:
    fit_predict_eval(model=model,
                     features_train=x_train,
                     features_test=x_test,
                     target_train=y_train,
                     target_test=y_test)

print(f'The answer to the question: RandomForestClassifier - 0.939')
