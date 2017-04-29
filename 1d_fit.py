"""
Training and using a KNN for 1D data interpolation and extrapolation.
Comparison of training methods, EKF vs SGD.

"""
# Dependencies
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from knn import KNN

# Get some noisy training data, a fun compact function
stdev = 0.05
U = np.arange(-10, 10, 0.2)
Y = np.exp(-U**2) + 0.5*np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))

# Create two identical KNN's that will be trained differently
knn_ekf = KNN(nu=1, ny=1, nl=10, neuron='logistic')
knn_sgd = KNN(nu=1, ny=1, nl=10, neuron='logistic')

# Train
nepochs_ekf = 100
nepochs_sgd = 400
knn_ekf.train(nepochs=nepochs_ekf, U=U, Y=Y, method='ekf', P=0.5, Q=0, R=stdev**2, pulse_T=0.75)
knn_sgd.train(nepochs=nepochs_sgd, U=U, Y=Y, method='sgd', step=0.05, pulse_T=0.5)

# Evaluation
X = np.arange(-15, 15, 0.01)
plt.suptitle("Data Fit", fontsize=22)
plt.scatter(U, Y, c='b', s=5)
plt.plot(X, knn_ekf.feedforward(X), c='g', lw=3, label='EKF: {} epochs'.format(nepochs_ekf))
plt.plot(X, knn_sgd.feedforward(X), c='k', ls=':', lw=2, label='SGD: {} epochs'.format(nepochs_sgd))
plt.grid(True)
plt.legend(fontsize=22)
plt.show()
