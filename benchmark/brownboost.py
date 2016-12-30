import math
import numpy as np
from scipy import special
import random
import copy as cp
from sklearn.tree import DecisionTreeClassifier


class BrownBoost:

    def __init__(self, weakLearner, c=10, nu=0, stop_criterion=0.01):
        """ Initiates BrownBoost classifier

        :param weakLearner: the boosted weak learner
        :param c:
        :param nu:
        :param stop_criterion:
        """
        self.weakLearner = weakLearner
        self.c = c
        self.nu = nu
        self.stop_criterion = stop_criterion
        self.alphas = []
        self.H = []

    def fit(self, X, Y):
        """ Trains the classifier

        Parameters:
            X -- array-like
                The training input samples.
            Y -- array-like
                The target values
        Returns:
            self
        """

        # Initiate parameters
        self.alphas = []
        self.Hs = []
        s = self.c
        R = np.zeros(X.shape[0])
        k=0
        maxK = 10000
        while s > 0 and k < maxK :
            k += 1

            # Weights update
            W = np.exp(-(R + s) ** 2 / self.c)

            # Weak learner training
            h = cp.deepcopy(self.weakLearner)
            h.fit(X, Y, sample_weight=W)

            # Prediction
            H = h.predict(X)

            # Correlation, used in initializing alpha
            gamma = (W * H * Y).sum()

            # Solve equation => extract t and alpha
            alpha, t = self.solveEq(R, H, s, gamma, Y)

            # Update margins
            R = R + alpha * H * Y
            # Update remaining time
            s = s - t

            # For final prediction
            self.alphas.append(alpha)
            self.Hs.append(h)

    def predict(self, X):
        """ Classify the samples

        Parameters:
            X -- array-like
                The training input samples.
        Returns:
            Y -- array of shape = [n_samples]
                The predicted values
        """

        Y = np.zeros(X.shape[0])
        for i in range(0, len(self.Hs)):
            Y += self.alphas[i] * self.Hs[i].predict(X)
        return np.sign(Y)

    def solveEq(self, R, H, s, gamma, Y):
        """ Computes alpha and t

        Parameters :
            R -- array
                margins for the samples
            H -- array
                predicted values
            s -- float
                'time remaining'
            gamma -- float
                correlation
            Y -- array-like
                the target values
        Returns :
            alpha -- float
            t -- float
        """

        # Starting point
        alpha = min([0.25, gamma])
        t = (alpha ** 2) / 3

        A = R + s
        B = H * Y

        k = 0
        maxK = 1000
        variation = self.stop_criterion + 1

        while k < maxK and variation > self.stop_criterion:
            D = A + (alpha * B - t)
            W = np.exp(-(D ** 2) / self.c)

            # Coefficients for computing alpha and t (jacobian)
            w = W.sum()
            u = (W * D * B).sum()
            b = (W * B).sum()
            v = (W * D * (B ** 2)).sum()
            e = (special.erf(D / math.sqrt(self.c)) - special.erf(A / math.sqrt(self.c))).sum()

            # alpha and t update
            sqrtPiC = math.sqrt(math.pi * self.c)
            alpha_1 = alpha + (self.c * w * b + sqrtPiC * u * e) / 2 * (u * w - u * b)
            t_1 = t + (self.c * (b ** 2) + sqrtPiC * v * e) / 2 * (u * w - u * b)
            # Variation with previous iteration
            variation = math.sqrt((alpha - alpha_1) ** 2 + (t - t_1) ** 2)

            alpha = alpha_1
            t = t_1

            k += 1

        return alpha, t


# if __name__ == '__main__':
#     print('Start')
#     X = np.zeros((100, 10))
#     Y = np.ones(100)
#
#     dt = DecisionTreeClassifier(max_depth=1)
#     BB = BrownBoost(dt)
#     BB.fit(X, Y)
#     BB.predict(X)
