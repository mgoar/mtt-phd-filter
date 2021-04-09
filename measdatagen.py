import numpy as np
import copy
import measmodel
import modelgen


def measdatagen(data, meas_model: measmodel.measmodel, sensor_model: modelgen):
    """Measurement data generation function"""

    MeasData = [[]] * sensor_model.K

    for k in range(len(data.N)):

        if (data.N[k] > 0):
            idx = np.where(np.random.rand(
                data.N[k].astype(int)) <= sensor_model.P_D)
            if (len(idx) != 0):
                meas_k = MeasData[k]
                for i in np.arange(0, len(idx)):
                    state = data.X[k][i]
                    z = np.random.multivariate_normal(
                        meas_model.h(state), meas_model.R).T

                    meas_k.append(z)

        N_c = np.random.poisson(sensor_model.lambda_c)

        clutter = np.reshape(np.repeat(sensor_model.range_c[:, 0], N_c), [2, N_c]) + np.matmul(
            np.diag(np.matmul(sensor_model.range_c, np.array([-1, 1]))), np.random.rand(meas_model.d, N_c))

        all_meas_k = [[row[i] for row in clutter] for i in range(N_c)]

        all_meas_k.append(z)

        MeasData[k] = copy.deepcopy(np.asarray(all_meas_k))
        meas_k.clear()

    return MeasData
