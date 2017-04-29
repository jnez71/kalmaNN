#!/usr/bin/env python
"""
Use the 1D data interpolation/extrapolation problem to benchmark convergence
variance. Comparison of training methods, EKF vs SGD.

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import kalmann

# Get some noisy training data, a fun compact function
stdev = 0.05
U = np.arange(-10, 10, 0.2)
Y = np.exp(-U**2) + 0.5*np.exp(-(U-3)**2) + np.random.normal(0, stdev, len(U))

# Repeat fitting experiment many times
nepochs_ekf = 100; nepochs_sgd = 400
ekf_results = []; sgd_results = []
for i in xrange(50):

    # Create two identical KNN's that will be trained differently
    knn_ekf = kalmann.KNN(nu=1, ny=1, nl=10, neuron='logistic')
    knn_sgd = kalmann.KNN(nu=1, ny=1, nl=10, neuron='logistic')

    # Train
    RMS_ekf, trcov = knn_ekf.train(nepochs=nepochs_ekf, U=U, Y=Y, method='ekf', P=0.5, Q=0, R=stdev**2, pulse_T=-1)
    RMS_sgd, _ = knn_sgd.train(nepochs=nepochs_sgd, U=U, Y=Y, method='sgd', step=0.05, pulse_T=-1)

    # Store results
    ekf_results.append(RMS_ekf[-1])
    sgd_results.append(RMS_sgd[-1])

# Evaluation
fig = plt.figure()
xlim = [0.33, 0.36]
fig.suptitle("Histogram of Final RMS Errors", fontsize=22)
ax = fig.add_subplot(2, 1, 1)
ax.hist(ekf_results, 20, normed=1)
ax.set_xlim(xlim)
ax.set_ylabel("Using EKF", fontsize=18)
ax.grid(True)
ax = fig.add_subplot(2, 1, 2)
ax.hist(sgd_results, 20, normed=1)
ax.set_xlim(xlim)
ax.set_ylabel("Using SGD", fontsize=18)
ax.set_xlabel("RMS", fontsize=18)
ax.grid(True)
fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.set_title("Trace of Covariance During Training", fontsize=22)
ax.plot(trcov)
ax.set_xlabel("Iteration", fontsize=16)
ax.grid(True)
plt.show()
