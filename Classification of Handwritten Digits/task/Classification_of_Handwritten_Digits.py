# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# from tensorflow.keras.datasets import mnist

import numpy as np

path = r"C:\Users\aaa\datasets\mnist.npz"
with np.load(path) as data:
    x_train = data['x_train']
    y_train = data['y_train']
    # x_test = data['x_test']
    # y_test = data['y_test']
x, y, z = x_train.shape
x_train = np.reshape(x_train, (x, y * z))
print(f'''Classes: {np.unique(y_train)}
Features' shape: {x_train.shape}
Target's shape: {y_train.shape}
min: {np.min(x_train)}, max: {np.max(x_train)}''')
