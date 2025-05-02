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
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV


def fit_predict_eval(model, features_train, features_test, target_train, target_test):
    model.fit(features_train, target_train)
    y_predict = model.predict(features_test)
    score = accuracy_score(target_test, y_predict)
    return score


def tune_and_evaluate_models(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray
) -> tuple[GridSearchCV, GridSearchCV]:

    # --- KNN GridSearch ---
    knn_params = {
        'n_neighbors': [4],
        'weights': ['distance'],
        'algorithm': ['auto', 'brute']
    }
    # knn_params = {
    #     'n_neighbors': [3, 4],
    #     'weights': ['uniform', 'distance'],
    #     'algorithm': ['auto', 'brute']
    # }

    knn_grid = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=knn_params,
        scoring='accuracy',
        n_jobs=-1
    )
    knn_grid.fit(x_train, y_train)

    # --- Random Forest GridSearch ---
    rf_params = {
        'n_estimators': [800],
        'max_depth': [30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['log2'],
        'class_weight': ['balanced_subsample'],
        'bootstrap': [True]
    }
    # rf_params = {
    #     'n_estimators': [600, 800],
    #     'max_depth': [30, 50, None],
    #     'min_samples_split': [2, 5],
    #     'min_samples_leaf': [1, 2],
    #     'max_features': ['sqrt', 'log2'],
    #     'class_weight': ['balanced_subsample'],
    #     'bootstrap': [True]
    # }
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(random_state=40),
        param_grid=rf_params,
        scoring='accuracy',
        n_jobs=-1
    )
    rf_grid.fit(x_train, y_train)

    return knn_grid, rf_grid


path = r"C:\Users\aaa\datasets\mnist.npz"
with np.load(path) as data:
    X = data['x_train']
    y = data['y_train']

X, y = X[0:6000], y[0:6000]
X = np.reshape(X, (X.shape[0], -1))
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)

normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.transform(x_test)

knn_grid, rf_grid = tune_and_evaluate_models(x_train_norm, x_test_norm, y_train, y_test)

param_knn = knn_grid.best_params_
knn_best_test = knn_grid.best_estimator_
knn_accuracy = fit_predict_eval(knn_best_test, x_train_norm, x_test_norm, y_train, y_test)

param_rf = rf_grid.best_params_
rf_best_test = rf_grid.best_estimator_
rf_accuracy = fit_predict_eval(rf_best_test, x_train_norm, x_test_norm, y_train, y_test)

print(f'''K-nearest neighbours algorithm
best estimator: {knn_best_test}
accuracy: {knn_accuracy}
''')
print(f'''Random forest algorithm
best estimator: {rf_best_test}
accuracy: {rf_accuracy}
''')
