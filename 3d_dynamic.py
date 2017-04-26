"""
Training and using a KNN for 3D-state dynamical prediction.

"""
# Dependencies
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from knn import KNN, load_knn

# Get some training data from the simulation of a nonlinear system, the Lorenz Attractor!
dt = 0.01  # physical resolution
tf = 100  # experiment duration
T = np.arange(0, tf, dt, dtype=np.float64)  # time record
X = np.zeros((len(T), 3), dtype=np.float64)  # state record
Xdot = np.zeros_like(X)  # state derivative record
x = np.array([1, 1, 1], dtype=np.float64)  # initial condition
for i, t in enumerate(T):
    X[i] = np.copy(x)  # record
    Xdot[i] = np.array((10*(x[1]-x[0]),
                        x[0]*(28-x[2])-x[1],
                        x[0]*x[1]-2.6*x[2]))  # dynamic
    x = x + Xdot[i]*dt  # step simulation
per = 0.01  # training data sampling period
skip = int(per/dt)

# Create and train KNN
knn = KNN(nu=3, ny=3, nl=30, neuron='tanh')
knn.train(nepochs=1, U=X[::skip], Y=Xdot[::skip], method='ekf', P=0.5, R=0.5, pulse_T=1)
# knn.save("lorenz")
# knn = load_knn('lorenz')

# Use KNN to simulate system from same initial condition
Xh = np.zeros_like(X)
xh = X[0]
for i, t in enumerate(T):
    Xh[i] = np.copy(xh)
    xh = xh + knn.feedforward(xh)*dt

# Evaluation
lim = int(1*len(T))
fig1 = plt.figure()
fig1.suptitle("Evolution", fontsize=22)
ax = fig1.gca(projection='3d')
ax.plot(X[0:lim:skip, 0], X[0:lim:skip, 1], X[0:lim:skip, 2], c='k', lw=1, ls=':', label="True")
ax.plot(Xh[0:lim:skip, 0], Xh[0:lim:skip, 1], Xh[0:lim:skip, 2], c='m', lw=1, label="Predict")
ax.set_xlim([-20, 20])
ax.set_ylim([-20, 30])
ax.set_zlim([0, 50])
ax.set_xlabel("x", fontsize=16)
ax.set_ylabel("y", fontsize=16)
ax.set_zlabel("z", fontsize=16)
plt.legend()
plt.show()
