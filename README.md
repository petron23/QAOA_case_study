# QAOA_case_study
Reimplementing and trying out the case study for electric vehicle recharging optimization problem based on "Theory and Implementation of the
Quantum Approximate Optimization" arXiv:2301.09535.

The general form of QUBO has the form:

$\min\limits_{\mathbf{b} \in \{0, 1\}^n} f_3(\mathbf{b})$

$f_3( \mathbf{b}) = \mathbf{b}^T A \mathbf{b}+ L \mathbf{b} + c$

$A \in \mathbb{R}^{n \times n}; \quad L \in \mathbb{R}^{n \times 1}; \quad c \in \mathbb{R}$

(The precise form of $f_3$ is derived in nb1 where it is shown how to obtain the it from a QCIO.)
Let us transform it into an Ising Hamiltonian. Writing the equation in coordinate representation

\sum\limits_{i=0}^{n-1} \sum\limits_{j>i}^{n-1} a_{ij} b_i b_j + \sum\limits_{i=0}^{n-1} l_i b_i + c

The following operation is applied:

$$b_i \leftrightarrow \frac{1}{2} \left( I^{\bigotimes n} - \sigma_Z^{(i)} \right)$$

$I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$,

$\sigma_Z = I\otimes ... \otimes I \otimes Z_{[\text{at ith pos}]} \otimes I \otimes...\otimes I$,

$Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$.

This substitution can be understood by using an Ising lattice that has instead of $\{-1, 1\}$ the values $\{0, 1\}$ on its sites. This is demonstrated below. 
