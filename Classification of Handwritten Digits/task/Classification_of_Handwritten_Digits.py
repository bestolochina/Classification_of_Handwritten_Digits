# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow.keras.datasets import mnist

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = r"C:\Users\aaa\datasets\mnist.npz"
with np.load(path) as data:
    X = data['x_train']
    y = data['y_train']

X = X[0:6000]
y = y[0:6000]
X = np.reshape(X, (X.shape[0], -1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)
proportion = round(pd.Series(y_train).value_counts(normalize=True), 2)

print(f'x_train shape: {X_train.shape}')
print(f'x_test shape: {X_test.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'y_test shape: {y_test.shape}')
print(f'Proportion of samples per class in train set:')
print(proportion)
