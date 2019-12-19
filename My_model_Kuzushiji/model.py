import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import scikitplot as skplt
from sklearn.metrics import classification_report,confusion_matrix


X_train = np.load('kmnist-train-imgs.npz')['arr_0']
y_train = np.load('kmnist-train-labels.npz')['arr_0']
X_test = np.load('kmnist-test-imgs.npz')['arr_0']
y_test = np.load('kmnist-test-labels.npz')['arr_0']


#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)

plt.figure(figsize=(6,6))
plt.imshow(X_train[1],cmap='inferno')
plt.title(y_train[1])
plt.show()



