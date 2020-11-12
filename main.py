import GaussianDensity
import PHDfilter
import modelgen
import motionmodel
import measmodel
import objectdatagen
import measdatagen
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats.distributions import chi2
from collections import namedtuple


# Object detection probability
P_D = 0.98

# Object survival probability
P_S = 0.99

# Clutter rate
lambda_c = 5

gaussianDensity = GaussianDensity.GaussianDensity()

# Sensor model - range/bearing model
range_c = np.array([[-1000, 1000], [-np.pi, np.pi]])

tracking_model = modelgen.modelgen()
tracking_model.sensormodel(P_D, lambda_c, range_c)

# Ground truth
n_b = 4
K = 100
t_b = np.zeros([n_b])
t_d = np.zeros([n_b])

PHDComp = namedtuple('PHDComp', ['x', 'P'])

# Track 0
x_0 = PHDComp(np.array([0, 0, 5, 0, np.pi/180]), [])
t_b[0] = 0
t_d[0] = 49

# Track 1
x_1 = PHDComp(np.array([20, 20, -20, 0, np.pi/90]), [])
t_b[1] = 19
t_d[1] = 69

# Track 2
x_2 = PHDComp(np.array([-20, 10, -10, 0, np.pi/360]), [])
t_b[2] = 39
t_d[2] = 89

# Track 3
x_3 = PHDComp(np.array([-10, -10, 8, 0, np.pi/270]), [])
t_b[3] = 59
t_d[3] = K-1

x_ = []

x_.append(x_0.x)
x_.append(x_1.x)
x_.append(x_2.x)
x_.append(x_3.x)

Birth = namedtuple('Birth', ['w', 'x', 'P'])
birth_model = Birth(np.repeat([np.log(.03)], 4), x_, [
                    np.diag(np.square([1, 1, 1, np.pi/90, np.pi/90]))]*4)

# Non-linear CT motion model
T = 1
sigmaV = 1
sigmaOmega = np.pi/180
motion_model = motionmodel.motionmodel()
motion_model.CTmodel(T, sigmaV, sigmaOmega)

# Non-linear range/bearing measurement model
sigma_r = 5
sigma_b = np.pi/180
s = np.array([300, 400])
meas_model = measmodel.measmodel()
meas_model.rangebearingmeasmodel(sigma_r, sigma_b, s)

P_G = 0.999
w_min = 1e-3
merg_th = 4
M = 100

ReductionParams = namedtuple("ReductionParams", ['w_min', 'M', 'merg_th'])
reduction_parameters = ReductionParams(w_min, M, merg_th)

gating_size = chi2.ppf(P_G, meas_model.d)

tracking_model.groundtruth(n_b, x_, t_b, t_d, K)

Obj = objectdatagen.objectdatagen(tracking_model, motion_model)
Z = measdatagen.measdatagen(Obj, meas_model, tracking_model)


def _plot(X_hat):

    fig, ax = plt.subplots()

    # Plot estimates
    epoch = 45
    x_hat = np.asarray([t[0][0] for ii, t in enumerate(X_hat) if ii < epoch])
    y_hat = np.asarray([t[0][1] for ii, t in enumerate(X_hat) if ii < epoch])

    ax.plot(x_hat, y_hat, '.')

    # Plot ground truth
    x_true = np.array([x_0.x[0]])
    y_true = np.array([x_0.x[1]])
    for idx, obj in enumerate(Obj.X):
        # Cardinality
        N = len(obj)

        x_true_temp = np.array([obj[n][0] for n in range(N)])
        y_true_temp = np.array([obj[n][1] for n in range(N)])

        x_true = np.concatenate((x_true, x_true_temp))
        y_true = np.concatenate((y_true, y_true_temp))

    ax.plot(x_true, y_true, '.')

    ax.set(xlabel='x', ylabel='y',
           title='PHD filter recursion')

    ax.grid()
    ax.set_xlim([-300, 300])
    ax.set_ylim([-300, 300])

    plt.legend(('PHD filter', 'Ground truth'),
               loc='upper right')
    
    ax.set_aspect('equal', 'box')
    plt.show()


def GMPHDfilter():

    K = len(Z)

    estimates = []

    phdFilter = PHDfilter.PHDfilter(gaussianDensity, birth_model)

    for k in range(K):

        # Update
        phdFilter.update(Z[k], meas_model, tracking_model.P_D,
                         tracking_model.intensity_c, gating_size)

        # Hypotheses reduction
        phdFilter.component_reduction(reduction_parameters)

        # Extract state estimate
        estimates.append(phdFilter.phd_state())

        # Predict
        phdFilter.predict(motion_model, P_S)

    return estimates


def main():

    phdEstimates = GMPHDfilter()

    _plot(phdEstimates)


if __name__ == "__main__":

    main()
