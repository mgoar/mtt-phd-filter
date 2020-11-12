import numpy as np
from dataclasses import dataclass


@dataclass
class motionmodel:
    """Motion model class"""

    def CVmodel(self, T: float, sigma: float):

        self.d = 4
        self.F = np.array([[1, 0, T, 0], [0, 1, 0, T],
                           [0, 0, 1, 0], [0, 0, 0, 1]])
        self.Q = sigma**2*np.array([[T**4/4, 0, T**3/2, 0], [0, T**4/4, 0, T**3/2], [
                                   T**3/2, 0, T**2, 0], [0, T**3/2, 0, T**2]])
        self.f = lambda x: self.F*x

    def CTmodel(self, T: float, sigmaV: float, sigmaOmega: float):

        self.d = 5
        self.f = lambda x: x + \
            np.array([T*x[2]*np.cos(x[3]), T*x[2]*np.sin(x[3]), 0, T*x[4], 0])
        self.F = lambda x: np.array([[1, 0, T*np.cos(x[3]), -T*x[2]*np.sin(x[3]), 0],
                                     [0, 1, T*np.sin(x[3]), +
                                      T*x[2]*np.cos(x[3]), 0],
                                     [0, 0, 1, 0, 0],
                                     [0, 0, 0, 1, T],
                                     [0, 0, 0, 0, 1]])
        self.G = np.append(np.zeros([2, 2]), [
                           [1, 0], [0, 0], [0, 1]]).reshape([self.d, 2])
        self.Q = np.matmul(self.G, np.diag(
            [sigmaV**2, sigmaOmega**2])).dot(self.G.T)
