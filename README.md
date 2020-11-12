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

![\begin{align*}
f_{k \vert k-1} (x \vert x_{k-1}) & = \mathcal{N} (x; F_{k-1} x_{k-1}, Q_{k-1}) \\
g_{k} (z \vert x) & = \mathcal{N} (z; H_{k} x, R_{k}) 
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Af_%7Bk+%5Cvert+k-1%7D+%28x+%5Cvert+x_%7Bk-1%7D%29+%26+%3D+%5Cmathcal%7BN%7D+%28x%3B+F_%7Bk-1%7D+x_%7Bk-1%7D%2C+Q_%7Bk-1%7D%29+%5C%5C%0Ag_%7Bk%7D+%28z+%5Cvert+x%29+%26+%3D+%5Cmathcal%7BN%7D+%28z%3B+H_%7Bk%7D+x%2C+R_%7Bk%7D%29+%0A%5Cend%7Balign%2A%7D%0A)

* The probabilities of survival and detection are constant.

* The birth intensity is a Gaussian mixture (GM) of the form:

![\begin{align*}
\gamma_{k} (x) & = \sum_{i=1}^{J_{\gamma}, k} w_{\gamma, k}^{(i)} \mathcal{N} (x; m^{(i)}_{\gamma, k}, P_{\gamma, k}^{(i)})
\end{align*}
](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0A%5Cgamma_%7Bk%7D+%28x%29+%26+%3D+%5Csum_%7Bi%3D1%7D%5E%7BJ_%7B%5Cgamma%7D%2C+k%7D+w_%7B%5Cgamma%2C+k%7D%5E%7B%28i%29%7D+%5Cmathcal%7BN%7D+%28x%3B+m%5E%7B%28i%29%7D_%7B%5Cgamma%2C+k%7D%2C+P_%7B%5Cgamma%2C+k%7D%5E%7B%28i%29%7D%29%0A%5Cend%7Balign%2A%7D%0A)


#### Prediction

First, a prediction is done for the ![J_{\gamma, k}
](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+J_%7B%5Cgamma%2C+k%7D%0A) birth targets. Existing targets generate ![J_{k-1}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+J_%7Bk-1%7D) components in the GM with parameters:

![\begin{align*}
w_{k \vert k-1}^{J_{k-1}+(i)} & = p_{S, k} w_{k-1}^{(j)} \\
m_{k \vert k-1}^{J_{k-1}+(i)} & = F_{k-1} m_{k-1}^{(j)}, P_{k \vert k-1}^{J_{k-1}+(i)} = F_{k-1} P_{k-1}^{(j)} F_{k-1}^{T} + Q_{k-1}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Aw_%7Bk+%5Cvert+k-1%7D%5E%7BJ_%7Bk-1%7D%2B%28i%29%7D+%26+%3D+p_%7BS%2C+k%7D+w_%7Bk-1%7D%5E%7Bj%7D+%5C%5C%0Am_%7Bk+%5Cvert+k-1%7D%5E%7BJ_%7Bk-1%7D%2B%28i%29%7D+%26+%3D+F_%7Bk-1%7D+m_%7Bk-1%7D%5E%7B%28j%29%7D%2C+P_%7Bk+%5Cvert+k-1%7D%5E%7BJ_%7Bk-1%7D%2B%28i%29%7D+%3D+F_%7Bk-1%7D+P_%7Bk-1%7D%5E%7B%28j%29%7D+F_%7Bk-1%7D%5E%7BT%7D+%2B+Q_%7Bk-1%7D%0A%5Cend%7Balign%2A%7D)


#### Update

Similarly, a Kalman filter update is performed on the detected objects (i.e., for measurements associated to an existing target) with posterior parameters:

![\begin{align*}
w_k & = p_{D,k} w_{k \vert k-1}^{(j)} \mathcal{N} (z; \hat{z}_{k \vert k-1}^{(j)}, S_{k}^{(j)}) \\
m_k & = m_{k \vert k-1}^{(j)} + K_{k}^{(j)} (z-\hat{z}_{k \vert k-1}^{(j)}) \\
P_k & = P_{k \vert k}^{(j)}
\end{align*}](https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cbegin%7Balign%2A%7D%0Aw_k+%26+%3D+p_%7BD%2Ck%7D+w_%7Bk+%5Cvert+k-1%7D%5E%7B%28j%29%7D+%5Cmathcal%7BN%7D+%28z%3B+%5Chat%7Bz%7D_%7Bk+%5Cvert+k-1%7D%5E%7Bh%7D%2C+S_%7Bk%7D%5E%7Bh%7D%29+%5C%5C%0Am_k+%26+%3D+m_%7Bk+%5Cvert+k-1%7D%5E%7B%28j%29%7D+%2B+K_%7Bk%7D%5E%7B%28j%29%7D+%28z-%5Chat%7Bz%7D_%7Bk+%5Cvert+k-1%7D%5E%7B%28j%29%7D%29+%5C%5C%0AP_k+%26+%3D+P_%7Bk+%5Cvert+k%7D%5E%7B%28j%29%7D%0A%5Cend%7Balign%2A%7D)

The weights must take into account clutter intensity by dividing them by the term ![\kappa_{k} (z) + \sum_{i=1}^{J_{k \vert k-1}} w_k^{(i)}](https://render.githubusercontent.com/render/math?math=%5Ctextstyle+%5Ckappa_%7Bk%7D+%28z%29+%2B+%5Csum_%7Bi%3D1%7D%5E%7BJ_%7Bk+%5Cvert+k-1%7D%7D+w_k%5E%7B%28i%29%7D).


---
## References
<a id="1">[1]</a>
Vo, B. N., & Ma, W. K. (2006). The Gaussian mixture probability hypothesis density filter. IEEE Transactions on signal processing, 54(11), 4091-4104.

Kudos to [https://tex-image-link-generator.herokuapp.com](https://tex-image-link-generator.herokuapp.com) for generating the links to the tex formulas.
