import numpy as np
from dataclasses import dataclass


@dataclass
class measmodel:
    """Measurement model class"""

    def CVmeasmodel(self, sigma: float):

        self.d = 2
        self.H = lambda x: np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R = sigma**2 * np.eye(self.d)
        self.h = lambda x: np.matmul(self.H(x), x)

    def CTmeasmodel(self, sigma: float):

        self.d = 2
        self.H = lambda x: np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        self.R = sigma**2 * np.eye(self.d)
        self.h = lambda x: np.matmul(self.H(x), x)

    def bearingmeasmodel(self, sigma: float, s: np.ndarray):

        self.d = 1
        def rng(x): return np.linalg.norm(x[0:1] - s)
        self.h = lambda x: np.arctan2(x[1] - s[1], x[0] - s[0])

        # Measurement model Jacobian
        self.H = lambda x: np.array(
            [[-(x[1] - s[1]) / (rng(x)**2), (x[0] - s[0]) / (rng(x)**2), np.zeros([np.shape(x)[1] - 2])]])

        # Measurement noise covariance
        self.R = sigma**2

    def rangebearingmeasmodel(
            self,
            sigma_r: float,
            sigma_b: float,
            s: np.ndarray):

        self.d = 2
        def rng(x): return np.linalg.norm(x[0:2] - s)
        def ber(x): return np.arctan2(x[1] - s[1], x[0] - s[0])

        self.h = lambda x: np.array([rng(x), ber(x)])

        # Measurement model Jacobian
        self.H = lambda x: np.array([np.pad([(x[0] - s[0]) / rng(x),
                                             (x[1] - s[1]) / rng(x)],
                                            (0,
                                             x.shape[0] - 2)),
                                     np.pad([-(x[1] - s[1]) / (rng(x)**2),
                                             (x[0] - s[0]) / (rng(x)**2)],
                                            (0,
                                             x.shape[0] - 2))])

        # Measurement noise covariance
        self.R = np.array([[sigma_r**2, 0], [0, sigma_b**2]])
