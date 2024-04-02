
# Probability Hypothesis Density (PHD) filter

Python code for the Multi-Target Tracking (MTT) Probability Hypothesis Density (PHD) filtering recursion. A non-linear Gaussian motion model (coordinated turn) with constant probability of survival and a range/bearing measurement model are implemented. Part of the code has been adapted from the edX course "Multi-Object Tracking for Automotive Systems".

![Example of filter recursion.](output.png)

### Usage
Run ```main.py``` to perform the filtering recursion.

### Dependencies
* NumPy
* SciPy
* Matplotlib
* collections

### PHD filter
A PHD filter is a random finite set (RFS) approach to multi-target tracking where both states and measurements are immersed into sets of random length and values. Thus a Bayesian filtering recursion is performed on these sets in order to dynamically estimate the quantities of interest. A closed form of the filter recursion is obtained under the following assumptions:

* Both target motion model and measurement model are linear Gaussian, i.e.:

$$
\begin{align*}
f_{k \vert k-1} (x \vert x_{k-1}) & = \mathcal{N} (x; F_{k-1} x_{k-1}, Q_{k-1}) \\
g_{k} (z \vert x) & = \mathcal{N} (z; H_{k} x, R_{k}) 
\end{align*}
$$

* The probabilities of survival and detection are constant.

* The birth intensity is a Gaussian mixture (GM) of the form:


$$
\begin{align*}
\gamma_{k} (x) & = \sum_{i=1}^{J_{\gamma}, k} w_{\gamma, k}^{(i)} \mathcal{N} (x; m_{\gamma, k}^{(i)}, P_{\gamma, k}^{(i)})
\end{align*}
$$


#### Prediction
First, a prediction is done for the $J_{\gamma, k}$ birth targets. Existing targets generate $J_{k-1}$ components in the GM with parameters:

$$
\begin{align*}
& i = 0 \\
& \textrm{for }  j = 1, \dots, J_{\gamma, k} \\
& \quad i := i +1 \\
& \quad w_{k\vert k-1}^{(i)} = w_{\gamma,k}^{j}, m_{k\vert k-1}^{(i)} = m_{\gamma\vert k}^{(j)}, P_{k\vert k-1}^{(i)}=P_{\gamma\vert k}^{(j)} \\
& \textrm{end} \\
& \textrm{for }  j = 1, \dots, J_{\beta, k} \\
& \quad \textrm{for }  j = 1, \dots, J_{k-1} \\
& \quad \quad i := i +1 \\
& \quad \quad w_{k\vert k-1}^{(i)} = w_{k-1}^{l} w_{\beta, k}^{j}, \\
& \quad \quad m_{k\vert k-1}^{(i)} = d_{\beta, k-1}^{(j)}+F_{\beta, k-1}^{(j)}m_{k-1}^{(l)}, \\
& \quad \quad P_{k\vert k-1}^{(i)} = Q_{\beta, k-1}^{(j)}+F_{\beta, k-1}^{(j)}P_{k-1}^{(l)}(F_{\beta, k-1}^{(j)})^{T}, \\
& \quad \textrm{end} \\
& \textrm{end} \\
& \textrm{for }  j = 1, \dots, J_{k-1} \\
& \quad i := i +1 \\
& \quad w_{k\vert k-1}^{(i)} = p_{S,k}w_{k-1}^{(j)}, m_{k\vert k-1}^{(i)} = F_{k-1} m_{k-1}^{(j)}, P_{k\vert k-1}^{(i)}=Q_{k-1}+F_{k-1}P_{k-1}^{(j)}F_{k-1}^{T} \\
& \textrm{end} \\
& J_{k\vert k-1} = i
\end{align*}
$$

#### Update
Similarly, a Kalman filter update is performed on the detected objects (i.e., for measurements associated to an existing target) with posterior parameters:

$$
\begin{align*}
& \textrm{for }  j = 1, \dots, J_{k \vert k-1} \\
& \quad w_{k}^{(j)} = (1-p_{D,k}) w_{k \vert k-1}^{(j)} \\
& \quad m_{k}^{(j)} = m_{k \vert k-1}^{(j)}, P_{k}^{(j)}=P_{k \vert k-1}^{(j)} \\
& \textrm{end} \\
& l := 0 \\
& \textrm{for each }  z \in Z_k \\
& l := l+1 \\
& \quad \textrm{for } j=1, \dots, J_{k \vert k-1} \\
& \quad \quad w_{k}^{(lJ_{k \vert k-1}+j)} = p_{D,k} w_{k \vert k-1}^{(j)} \mathcal{N} \left(z; z_{k \vert k-1}^{(j)}, S_k \right), \\
& \quad \quad m_{k}^{(lJ_{k \vert k-1}+j)} = m_{k \vert k-1}^{(j)}+K_{k}^{(j)}(z-z_{k \vert k-1}^{(j)}), \\
& \quad \quad P_{k}^{(lJ_{k \vert k-1}+j)} = P_{k \vert k}^{(j)}, \\
& \quad \textrm{end} \\
& \textrm{end} \\
& J_{k} = lJ_{k\vert k-1}+J_{k\vert k-1}
\end{align*}
$$

The weights must take into account clutter intensity by dividing them by the term 

$$
\kappa_k(z)+\sum_{i=1}^{J_{k \vert k-1}} w_k^{(lJ_{k \vert k-1}+i)}
$$

for $j=1, \dots, J_{k\vert k-1}$.

## References
<a id="1">[1]</a>
Vo, B. N., & Ma, W. K. (2006). The Gaussian mixture probability hypothesis density filter. IEEE Transactions on signal processing, 54(11), 4091-4104.
