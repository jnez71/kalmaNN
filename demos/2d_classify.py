"""
Training and using a KNN for classification of 2D data.
Comparison of training methods, EKF vs SGD.

"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import kalmann

# Get some noisy training data classifications, spirals!
n = 100
stdev = 0.2
U = np.zeros((n*3, 2))
Y = np.zeros((n*3, 1), dtype='uint8')
for j in xrange(3):
    ix = range(n*j, n*(j+1))
    r = np.linspace(0, 1, n)
    t = np.linspace(j*4, (j+1)*4, n) + np.random.normal(0, stdev, n)
    U[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    Y[ix] = j
Y[-20:-18] = 0

# Create two identical KNN's that will be trained differently
knn_ekf = kalmann.KNN(nu=2, ny=1, nl=10, neuron='logistic')
knn_sgd = kalmann.KNN(nu=2, ny=1, nl=10, neuron='logistic')

# Train
nepochs_ekf = 100
nepochs_sgd = 200
knn_ekf.train(nepochs=nepochs_ekf, U=U, Y=Y, method='ekf', P=0.2, Q=0, R=stdev**2, pulse_T=2)
knn_sgd.train(nepochs=nepochs_sgd, U=U, Y=Y, method='sgd', step=0.1, pulse_T=2)

# Use the KNNs as classifiers
F_ekf = knn_ekf.classify(U, high=2, low=0)
F_sgd = knn_sgd.classify(U, high=2, low=0)
print("EKF Classification Accuracy: {}%".format(int(100*np.sum(F_ekf==Y)/len(Y))))
print("SGD Classification Accuracy: {}%\n".format(int(100*np.sum(F_sgd==Y)/len(Y))))

# Evaluation
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
ax.set_title("True Classifications", fontsize=22)
ax.scatter(U[:, 0], U[:, 1], c=Y)
plt.axis('equal')
ax = fig.add_subplot(1, 3, 2)
ax.set_title("EKF: {} epochs".format(nepochs_ekf), fontsize=22)
ax.scatter(U[:, 0], U[:, 1], c=F_ekf)
plt.axis('equal')
ax = fig.add_subplot(1, 3, 3)
ax.set_title("SGD: {} epochs".format(nepochs_sgd), fontsize=22)
ax.scatter(U[:, 0], U[:, 1], c=F_sgd)
plt.axis('equal')
plt.show()
