import numpy as np
from dataclasses import dataclass


@dataclass
class modelgen:
    """Tracking model class"""

    def sensormodel(self, P_D: float, lambda_c: float, range_c: np.ndarray):

        self.P_D = P_D
        self.lambda_c = lambda_c
        self.range_c = range_c
        if range_c.shape[0] > 1:
            V = (range_c[0][1]-range_c[0][0])*(range_c[1][1]-range_c[1][0])
        else:
            V = range_c[1]-range_c[0]

        self.pdf_c = 1/V
        self.intensity_c = lambda_c * self.pdf_c

    def groundtruth(self, n_births: int, x_0: np.ndarray, t_birth: np.ndarray, t_death: np.ndarray, K: int):

        self.n_births = n_births
        self.x_0 = x_0
        self.t_birth = t_birth
        self.t_death = t_death
        self.K = K
