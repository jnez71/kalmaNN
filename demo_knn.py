"""
Training and using a KNN.

"""
# Dependencies
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN, load_knn

# Create KNN
knn = KNN(nu=1, ny=1, nl=12, neuron='sigmoid')

# Get some training data
stdev = 0.05
U = np.arange(-10, 10, 0.1)
Y = np.exp(-U**2) + np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))

# Train
knn.train(nepochs=400, U=U, Y=Y, method='ekf', P=0.2, Q=0, R=stdev**2)

# Evaluation
X = np.arange(-15, 15, 0.01)
F = knn.feedforward(X)
plt.scatter(U, Y, c='r')
plt.plot(X, F, c='b', linewidth=3)
plt.grid(True)
plt.show()
