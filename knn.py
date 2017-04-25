"""
Contains a class for using and EKF-training a basic neural network.
This is primarily to demonstrate the advantages of EKF-training.
See the class docstrings for more details.
This module also includes a function for loading stored KNN objects.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from scipy.linalg import block_diag
from time import time
import pickle

##########

def load_knn(filename):
    """
    Loads a stored KNN object saved with the string filename.
    Returns the loaded object.

    """
    if not isinstance(filename, str):
        raise ValueError("The filename must be a string.")
    if filename[-4:] != '.knn':
        filename = filename + '.knn'
    with open(filename, 'rb') as input:
        return pickle.load(input)

##########

class KNN:
    """
    Class for a feedforward neural network (NN). Currently only handles 1 hidden-layer,
    is always fully-connected, and uses the same activation function type for every neuron.
    There is no output saturation, so classification must be handled as a real function fit.
    The NN can be trained by extended kalman filter (EKF) or stochastic gradient descent (SGD).

    """
    def __init__(self, nu, ny, nl, neuron, sprW=5):
        """
            nu: dimensionality of input; integer
            ny: dimensionality of output; integer
            nl: number of hidden-layer neurons; integer
        neuron: activation function type; 'sigmoid', 'tanh', or 'relu'
          sprW: spread of initial randomly sampled synapse weights; float scalar

        """
        # Function dimensionalities
        self.nu = int(nu)
        self.ny = int(ny)
        self.nl = int(nl)

        # Neuron type
        if neuron == 'sigmoid':
            self.sig = lambda V: (1 + np.exp(-V))**-1
            self.dsig = lambda sigV: sigV * (1 - sigV)
        elif neuron == 'tanh':
            self.sig = lambda V: np.tanh(V)
            self.dsig = lambda sigV: 1 - sigV**2
        elif neuron == 'relu':
            self.sig = lambda V: np.clip(V, 0, np.inf)
            self.dsig = lambda sigV: np.float64(sigV > 0)
        else:
            raise ValueError("The neuron argument must be 'sigmoid', 'tanh', or 'relu'.")
        self.neuron = neuron

        # Initial synapse weight matrices
        sprW = np.float64(sprW)
        self.W = [sprW*(2*np.random.sample((nl, nu+1))-1),
                  sprW*(2*np.random.sample((ny, nl+1))-1)]
        self.nW = sum(map(np.size, self.W))
        self.P = None

        # Function for pushing signals through a synapse with bias
        self._affine_dot = lambda W, V: np.dot(np.atleast_1d(V), W[:, :-1].T) + W[:, -1]

####

    def save(self, filename):
        """
        Saves the current NN to a file with the given string name.

        """
        if not isinstance(filename, str):
            raise ValueError("The filename must be a string.")
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

####

    def feedforward(self, U, get_l=False):
        """
        Feeds forward an (m by nu) array of inputs U through the NN.
        Returns the associated (m by ny) output matrix, and optionally
        the intermediate activations l.

        """
        if U.ndim == 1: U = U[:, np.newaxis]
        l = self.sig(self._affine_dot(self.W[0], U))
        h = self._affine_dot(self.W[1], l)
        if get_l: return h, l
        return h

####

    def train(self, nepochs, U, Y, method, P=None, Q=None, R=None, step=1, pulse=1):
        """
        nepochs: number of epochs (presentations of the training data); integer
              U: input training data; float array m samples by nu inputs
              Y: output training data; float array m samples by ny outputs
         method: extended kalman filter ('ekf') or stochastic gradient descent ('sgd')
              P: initial weight covariance for ekf; float scalar or (nW by nW) array
              Q: process covariance for ekf; float scalar or (nW by nW) array
              R: data covariance for ekf; float scalar or (ny by ny) array
           step: step size scaling; float scalar
          pulse: number of seconds between displaying current training status; float

        If method is 'sgd' then P, Q, and R are unused, so carefully choose step.
        If method is 'ekf' then step=1 is "optimal", R must be specified, and:
            P is None: P = self.P if self.P has been created by previous training
            Q is None: Q = 0
        If P, Q, or R are given as scalars, they will scale an identity matrix.
        Set pulse to -1 to suppress training status display.

        """
        # Verify data
        U = np.array(U, dtype=np.float64)
        Y = np.array(Y, dtype=np.float64)
        if len(U) != len(Y):
            raise ValueError("Number of input data points must match number of output data points.")
        if U.ndim != 1 and U.shape[-1] != self.nu:
            raise ValueError("Shape of U must be (m by nu) or (m,).")
        if Y.ndim != 1 and Y.shape[-1] != self.ny:
            raise ValueError("Shape of Y must be (m by ny) or (m,).")

        # Set-up
        if method == 'ekf':
            self.update = self._ekf
            if P is None:
                if self.P is None:
                    raise ValueError("Initial P not specified.")
            elif np.isscalar(P):
                self.P = P*np.eye(self.nW)
            else:
                if np.shape(P) != (self.nW, self.nW):
                    raise ValueError("P must be a float scalar or (nW by nW) array.")
                self.P = np.array(P, dtype=np.float64)
            self.Q_nonzero = True
            if Q is None:
                self.Q = np.zeros((self.nW, self.nW))
                self.Q_nonzero = False
            elif np.isscalar(Q):
                self.Q = Q*np.eye(self.nW)
            else:
                if np.shape(Q) != (self.nW, self.nW):
                    raise ValueError("Q must be a float scalar or (nW by nW) array.")
                self.Q = np.array(Q, dtype=np.float64)
            if R is None:
                raise ValueError("R must be specified for EKF training.")
            elif np.isscalar(R):
                self.R = R*np.eye(self.ny)
            else:
                if np.shape(R) != (self.ny, self.ny):
                    raise ValueError("R must be a float scalar or (ny by ny) array.")
                self.R = np.array(R, dtype=np.float64)
        elif method == 'sgd':
            self.update = self._sgd
        else:
            raise ValueError("The method argument must be either 'ekf' or 'sgd'.")
        last_pulse = 0

        # Shuffle data between epochs
        for epoch in xrange(nepochs):
            rand_idx = np.random.permutation(len(U))
            U_shuffled = U[rand_idx]
            Y_shuffled = Y[rand_idx]

            # Train
            for i, (u, y) in enumerate(zip(U_shuffled, Y_shuffled)):

                # Forward propagation
                h, l = self.feedforward(u, get_l=True)

                # Do the learning
                self.update(u, y, h, l, step)

                # Heartbeat
                if time() - last_pulse > pulse:
                    print("  Epoch: {}/{}".format(epoch, nepochs))
                    print("------------------")
                    print("    MSE: {}".format(np.round(np.mean(np.square(Y - self.feedforward(U))), 6)))
                    if method == 'ekf': print("tr(Cov): {}".format(np.round(np.trace(self.P), 6)))
                    print("------------------\n\n")
                    last_pulse = time()

####

    def _ekf(self, u, y, h, l, step):

        # Compute NN jacobian
        D = (self.W[1][:, :-1]*self.dsig(l)).flatten()
        H = np.hstack((np.hstack((np.outer(D, u), D[:, np.newaxis])).reshape(self.ny, self.W[0].size),
                       block_diag(*np.tile(np.concatenate((l, [1])), self.ny).reshape(self.ny, self.nl+1))))

        # Kalman gain
        S = H.dot(self.P).dot(H.T) + self.R
        K = self.P.dot(H.T).dot(npl.inv(S))

        # Update weight estimates and covariance
        dW = step*K.dot(y-h)
        self.W[0] = self.W[0] + dW[:self.W[0].size].reshape(self.W[0].shape)
        self.W[1] = self.W[1] + dW[self.W[0].size:].reshape(self.W[1].shape)
        self.P = self.P - K.dot(H).dot(self.P)
        if self.Q_nonzero: self.P = self.P + self.Q

####

    def _sgd(self, u, y, h, l, step):
        print("SGD NOT YET IMPLEMENTED")  ###
        assert False
