"""
Training and using a KNN for 1D data interpolation and extrapolation.
Comparison of training methods, EKF vs SGD.

"""
# Dependencies
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

# Get some training data, a fun compact function or somethin'
stdev = 0.05
U = np.arange(-10, 10, 0.2)
Y = np.exp(-U**2) + 0.5*np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))

# Create two identical KNN's that will be trained differently
np.random.seed(1)
knn_ekf = KNN(nu=1, ny=1, nl=10, neuron='sigmoid')
knn_sgd = KNN(nu=1, ny=1, nl=10, neuron='sigmoid')

# Train
knn_ekf.train(nepochs=200, U=U, Y=Y, method='ekf', P=0.1, Q=0, R=stdev**2, pulse_T=-1)
knn_sgd.train(nepochs=200, U=U, Y=Y, method='sgd', step=0.1, pulse_T=-1)

# Evaluation
X = np.arange(-15, 15, 0.01)
plt.scatter(U, Y, c='b', s=5)
plt.plot(X, knn_ekf.feedforward(X), c='g', lw=3, label='EKF')
plt.plot(X, knn_sgd.feedforward(X), c='k', ls=':', lw=2, label='SGD')
plt.grid(True)
plt.legend(fontsize=22)
plt.show()
