"""
Training and using a KNN for classification of 2D data.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

# Get some training data, simple XOR or somethin'
U = np.array([[   1,    0],
              [   0,    1],
              [  -1,    0],
              [   0,   -1],
              [ 0.5,  0.5],
              [-0.5,  0.5],
              [ 0.5, -0.5],
              [-0.5, -0.5]])
Y = np.array([[1],
              [1],
              [1],
              [1],
              [0],
              [0],
              [0],
              [0]])

# Create and train KNN
knn = KNN(nu=2, ny=1, nl=10, neuron='sigmoid')
knn.train(nepochs=100, U=U, Y=Y, method='ekf', P=0.2, Q=0, R=0.2, step=1, pulse_T=0.1)

# Evaluation
F = knn.classify(U, high=1, low=0)
print("Classification Accuracy: {}%\n".format(int(100*len(np.argwhere(F==Y))/len(Y))))
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_title("True Classifications")
ax.scatter(U[:, 0], U[:, 1], c=Y)
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Learned Classifications")
ax.scatter(U[:, 0], U[:, 1], c=F)
plt.show()
