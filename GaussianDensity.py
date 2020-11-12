import numpy as np
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from dataclasses import dataclass
from collections import namedtuple
import motionmodel
import measmodel

GaussState = namedtuple('GaussState', ['x', 'P'])


@dataclass
class GaussianDensity:
    """Gaussian density class"""

    def predict(state: namedtuple, motion_model: motionmodel):

        state_ = GaussState(motion_model.f(state.x), np.matmul(motion_model.F(
            state.x), state.P).dot(motion_model.F(state.x).T)+motion_model.Q)

        return state_

    def update(self, state, z, meas_model):

        # Measurement model Jacobian
        H_ = meas_model.H(state.x)

        # Innovation covariance
        S_ = np.matmul(H_, state.P).dot(H_.T) + meas_model.R

        # Make sure matrix S is positive definite
        S_ = (S_+S_.T)/2

        # S^(-1)
        Sinv = np.linalg.inv(S_)

        # Kalman gain
        K = state.P*H_.T*Sinv

        state_ = GaussState(state.x + K*(z-measmodel.h(state.x)),
                            (np.eye(np.size(state.x)[0]-K*H_))*state.P)

        return state_

    def predictedLikelihood(self, state, z, meas_model):

        # Measurement model Jacobian
        H_ = meas_model.H(state.x)

        # Innovation covariance
        S_ = np.matmul(H_, state.P).dot(H_.T) + meas_model.R

        # Make sure matrix S is positive definite
        S_ = (S_+S_.T)/2

        z_hat = meas_model.h(state.x)

        log_lklhd = []
        for m in range(z.ndim):
            thisZ = z[m:]
            log_lklhd.append(multivariate_normal.logpdf(
                thisZ, z_hat, S_, allow_singular=True))

        return log_lklhd

    def ellipsoidalGating(state, z, meas_model, gating_size):

        # Measurement model Jacobian
        H_ = meas_model.H(state.x)

        # Innovation covariance
        S_ = np.matmul(H_, state.P).dot(H_.T) + meas_model.R

        # Make sure matrix S is positive definite
        S_ = (S_+S_.T)/2

        # S^(-1)
        Sinv = np.linalg.inv(S_)

        # Mahalanobis distance
        isZInGate = [True if
                     (distance.mahalanobis(z_, meas_model.h(
                         state.x), Sinv)**2 < gating_size)
                     else False
                     for z_ in z]

        return isZInGate, z[isZInGate]
