import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

from lab11.task1 import get_fraction, get_t

A = np.array([
    [0, 1],
    [0, 0]
])

B_1 = np.array([
    [1, 1, 0],
    [0, 1, 0]
])

B_2 = np.array([[0],
                [1]])

C_1 = np.array([[1, 0]])
D_1 = np.array([[0, 0, 1]])

task1_C_2s = np.array([
    [[1, 0],
     [0, 1],
     [0, 0]],
    [[1, 1],
     [0, 2],
     [0, 0]],
])
task1_D_2s = np.array([[[0], [0], [1]], [[0], [0], [2]]])

ts = get_t(15)
w = np.vstack([0.05 * np.sin(ts), 0.01 * np.sin(10 * ts), 0.01 * np.sin(10 * ts)])

omega_i = sympy.Symbol("omega",real=True) * sympy.I

C_2 = task1_C_2s[0]
D_2 = task1_D_2s[0]

gammas = [1.4, 2, 10]

def generate_Hinf_obs(a, b_1, b_2, c_1, c_2, d_1, d_2, gamma):
    R_1 = c_1.T @ np.linalg.inv(d_1 @ d_1.T) @c_1 - (gamma**-2) * c_2.T @ c_2
    R_2 = b_2 @ np.linalg.inv(d_2.T @ d_2) @ b_2.T - (gamma**-2) * b_1 @ b_1.T
    p = scipy.linalg.solve_continuous_are(a.T, np.identity(R_1.shape[0]), b_1@b_1.T, np.linalg.inv(R_1))
    q = scipy.linalg.solve_continuous_are(a, np.identity(R_2.shape[0]), c_2.T@c_2, np.linalg.inv(R_2))
    if np.max(np.linalg.eig(p@q)[0]) < gamma ** 2:
        l = -p@np.linalg.inv(np.identity(q.shape[0])-(gamma**-2)*q@p)@(c_1+(gamma**-2)*d_1@b_1.T@q).T@np.linalg.inv(d_1@d_1.T)
        k = -np.linalg.inv(d_2.T@d_2)@b_2.T@q
        return k, l, q
    return None

for gamma in gammas:
    K_4_1, L_4_1, Q_1 = generate_Hinf_obs(A, B_1, B_2, C_1, C_2, D_1, D_2, gamma)
    print(K_4_1)