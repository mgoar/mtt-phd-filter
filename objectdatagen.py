import numpy as np
import modelgen
import motionmodel
import copy
from collections import namedtuple


def objectdatagen(
        ground_truth: modelgen.modelgen,
        motion_model: motionmodel.motionmodel):
    """Object data generation function"""

    K = ground_truth.K

    Data = namedtuple('Data', ['X', 'N'])

    ObjectData = Data([[]] * K, np.zeros([K]))

    for i in np.arange(0, ground_truth.n_births):
        state = ground_truth.x_0[i]
        for k in np.arange(ground_truth.t_birth[i], np.min(
                [ground_truth.t_death[i], K])).astype(int):
            state_ = np.random.multivariate_normal(
                motion_model.f(state), motion_model.Q)

            obj_k = ObjectData.X[k]
            obj_k.append(state_)
            ObjectData.X[k] = copy.deepcopy(obj_k)
            obj_k.clear()

            ObjectData.N[k] = ObjectData.N[k] + 1

            state = state_

    return ObjectData
