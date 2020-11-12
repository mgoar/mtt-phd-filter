import numpy as np
from collections import namedtuple

GaussComp = namedtuple('GaussComp', ['x', 'P'])


def moment_matching(w, gauss):

    if(w.shape[0] == 1):
        return gauss[0].x, gauss[0].P
    else:
        w_ = np.exp(w)

        x_ = np.zeros(gauss[0].x.shape)
        P_ = np.zeros(gauss[0].P.shape)

        for idx, g in enumerate(gauss):
            x_ = x_ + w_[idx]*gauss[idx].x

        for idx, g in enumerate(gauss):
            P_ = P_ + w_[idx]*gauss[idx].P + w_[idx] * \
                np.matmul((gauss[idx].x-x_), (gauss[idx].x-x_).T)

        return x_, P_


def normalize_log_w(wghts):

    if(wghts.shape[0] <= 1):
        return np.array([0.]), wghts
    else:
        w_temp = wghts[np.argsort(-wghts)]
        log_sum = np.max(
            w_temp)+np.log(1+np.sum(np.exp(wghts[np.argsort(-wghts)[1:]]-np.max(w_temp))))
        return wghts-log_sum, log_sum


def prune(w, threshold: float):

    return np.where(np.exp(w) >= threshold)


def merge(w, hypotheses, M):

    w_merged = []
    if(w.shape[0] == 1):
        return w, hypotheses
    else:
        I = np.arange(len(hypotheses))
        el = 0

        merged = []

        while(I.shape[0] != 0):
            Ij = []

            # Find component with highest weight
            I_w_max = np.argmax(w)

            for i in I:
                temp = hypotheses[i].x - hypotheses[I_w_max].x
                val = temp.T.dot(np.linalg.inv(
                    hypotheses[I_w_max].P).dot(temp))

                if(val < M):
                    Ij.append(i)

            toReduce = []
            for i in Ij:
                toReduce.append(hypotheses[i])

            ww, w_sum = normalize_log_w(w[Ij])

            # Moment matching
            xx, PP = moment_matching(ww, toReduce)

            thisGauss = GaussComp(xx, PP)
            w_merged.append(np.squeeze(w_sum))
            merged.append(thisGauss)

            I_temp = np.setdiff1d(np.asarray(I), np.asarray(Ij))

            if(I_temp.shape[0] != 0):
                I = np.asarray(I_temp)
            else:
                I = np.array([])

            w[Ij] = np.log(1e-32)

            el = el+1

        return np.squeeze(w_merged), merged


def cap(w, threshold: int):

    idx_w = np.argsort(w)

    return np.array(idx_w[0:np.min([threshold, idx_w.shape[0]])])
