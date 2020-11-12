import numpy as np
import GaussianDensity
import motionmodel
import measmodel
import DensityUtils
from dataclasses import dataclass
from collections import namedtuple

PHDComp = namedtuple('PHDComp', ['x', 'P'])


@dataclass
class PHDfilter:
    """Probability Hypothesis Density (PHD) class"""

    density: GaussianDensity
    w: np.ndarray
    obj_state: np.ndarray

    def __init__(self, density_handle: GaussianDensity, birth_model: namedtuple):

        self.density = density_handle
        self.w = birth_model.w
        self.obj_state = [PHDComp(xx, PP)
                          for xx, PP in zip(birth_model.x,
                                            birth_model.P)
                          ]

    def predict(self, motion_model: motionmodel, P_S: float):

        # Predict each Gaussian component in the Poisson intensity for pre-existing objects
        for l in range(len(self.obj_state)):
            thisState = PHDComp(self.obj_state[l].x, self.obj_state[l].P)
            pred_ = GaussianDensity.GaussianDensity.predict(
                thisState, motion_model)

            # Re-definition namedtuple
            pred_state = PHDComp(pred_.x, pred_.P)
            self.obj_state.append(pred_state)

        # Add Poisson birth intensity to the Poisson intensity for pre-existing objects
        w_ = self.w + np.log(P_S)
        allW = np.concatenate((self.w, np.squeeze(w_)), axis=0)
        self.w = np.squeeze(allW)

    def update(self, z_: np.ndarray, meas_model: measmodel, P_D: float, intensity: float, gating_size: float):

        # Construct update components resulted from missed detections
        misseddetect = self.obj_state
        w_k = self.w + np.log(1-P_D)

        # Perform ellipsoidal gating for each Gaussian component in the Poisson intensity
        isZinGate = []

        for l in range(len(self.obj_state)):
            thisState = PHDComp(self.obj_state[l].x, self.obj_state[l].P)
            [isZ, whichZinGate] = GaussianDensity.GaussianDensity.ellipsoidalGating(
                thisState, z_, meas_model, gating_size)
            isZinGate.append(isZ)

        # Construct Kalman update components with measurements inside the gates
        KalmanK = []
        KalmanP = []
        KalmanZhat = []

        for l in range(len(self.obj_state)):
            thisState = PHDComp(self.obj_state[l].x, self.obj_state[l].P)

            H_ = meas_model.H(thisState.x)
            zhat = meas_model.h(thisState.x)
            S_ = np.matmul(H_, thisState.P).dot(H_.T) + meas_model.R
            S_ = (S_+S_.T)/2

            # S^(-1)
            Sinv = np.linalg.inv(S_)

            K_ = np.matmul(thisState.P, H_.T).dot(Sinv)

            P_ = np.matmul(
                (np.eye(self.obj_state[l].x.shape[0]) - np.matmul(K_, H_)), thisState.P)

            KalmanK.append(K_)
            KalmanP.append(P_)
            KalmanZhat.append(zhat)

        detected = []
        wtilde_ = []
        for idx, isZ_ in enumerate(isZinGate):

            if(len(z_[isZ_]) != 0):

                thisState = PHDComp(
                    self.obj_state[idx].x, self.obj_state[idx].P)
                measIdx = np.nonzero(isZ_)
                for l in range(np.count_nonzero(isZ_)):
                    thisDetected = PHDComp(np.squeeze(
                        thisState.x+np.matmul(KalmanK[idx], z_[measIdx[0][l]] - KalmanZhat[idx])), KalmanP[idx])

                    log_like = self.density.predictedLikelihood(
                        thisState, z_[measIdx[0][l]], meas_model)
                    wtilde = np.log(P_D) + self.w[idx] + log_like

                    wtilde_.append(wtilde)
                    detected.append(thisDetected)

        w_ = np.asarray(wtilde_) - np.log(intensity +
                                          np.sum(np.exp(np.asarray(wtilde_))))

        if(np.ndim(np.squeeze(w_)) != 0):
            self.w = np.concatenate((w_k, np.squeeze(w_)), axis=0)
        else:
            self.w = np.concatenate((w_k, np.squeeze(w_).shape), axis=0)

        if(len(detected) != 0):
            self.obj_state = misseddetect + detected

    def component_reduction(self, reduction: namedtuple):

        # Prune
        idx = DensityUtils.prune(self.w, reduction.w_min)

        # Merge
        temp = []
        for val in idx[0]:
            s_ = PHDComp(np.squeeze(
                self.obj_state[val].x), self.obj_state[val].P)
            temp.append(s_)

        ww, merged_comps = DensityUtils.merge(
            self.w[idx], temp, reduction.merg_th)

        # Cap
        idx_cap = DensityUtils.cap(ww, reduction.M)

        if(idx_cap.shape[0] != 1):
            self.w = np.array([ww[i] for i in idx_cap])
            self.obj_state = [
                PHDComp(merged_comps[i].x, merged_comps[i].P) for i in idx_cap]
        else:
            self.w = ww
            self.obj_state = merged_comps

    def phd_state(self):

        # Get a mean estimate of the cardinality of objects by taking the summation of the weights of the Gaussian components rounded to the nearest integer
        n = np.min([np.rint(np.sum(np.exp(self.w))), self.w.size]).astype(int)

        # Extract n object states form the means of the n Gaussiaon cmponents with the highest weights
        idx_w_sorted = np.argsort(self.w)[::-1]

        temp = [self.obj_state[ii] for ii in idx_w_sorted]

        state_estimate = []
        for k in range(n):
            state_estimate.append(temp[k].x)

        return state_estimate
