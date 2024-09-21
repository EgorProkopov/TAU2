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

def generate_H2_obs(a, b_1, c_1, d_1):
    p = scipy.linalg.solve_continuous_are(a.T, c_1.T, b_1@b_1.T, d_1@d_1.T)
    return -p @ c_1.T @np.linalg.inv(d_1 @ d_1.T)

def generate_H2_obs(a, b_1, c_1, d_1):
    p = scipy.linalg.solve_continuous_are(a.T, c_1.T, b_1 @ b_1.T, d_1 @ d_1.T)
    return -p @ c_1.T @ np.linalg.inv(d_1 @ d_1.T)


for i in range(2):
    print('\n______________________________')
    task1_C_2 = task1_C_2s[i]
    task1_D_2 = task1_D_2s[i]
    # check_controllability_eigens(A, B_2)
    # check_observability_eigens(task1_C_2, A)
    Q = task1_C_2.T @ task1_C_2
    R = task1_D_2.T @ task1_D_2
    K, S, E = control.lqr(A, B_2, Q, R)
    K = -K
    print(f'\[C_2 = {a2l.to_ltx(task1_C_2, print_out=False)}; D_2 = {a2l.to_ltx(task1_D_2, print_out=False)};\]')
    # print(f'\[C_2^T D_2 = 0: {np.all(task1_C_2.T @ task1_D_2 == 0)}\]')
    # print(f'\[D_2^T D_2 \\text{"{ обратима}"}: {np.linalg.det(task1_D_2.T @ task1_D_2) != 0}\]')
    # print(f'\[spec(A-B_2 K) = {a2l.to_ltx(E, print_out=False)}\]')
    # print(f'\[Q = {a2l.to_ltx(S, print_out=False)}\]')
    print(f'\[K = {a2l.to_ltx(K, print_out=False)}\]')

    L = generate_H2_obs(A, B_1, C_1, D_1)
    print(f'\[L = {a2l.to_ltx(L, print_out=False)}\]')

    new_A = np.block([[A, B_2 @ K], [-L @ C_1, A + B_2 @ K + L @ C_1]])
    new_B = np.block([[B_1], [-L @ D_1]])
    new_C = np.block([task1_C_2, -task1_D_2 @ K])
    ss = control.ss(new_A, new_B, new_C, 0)
    tf = control.ss2tf(ss)

    smatrix = []
    for row in range(tf.noutputs):
        srow = []
        for col in range(tf.ninputs):
            srow.append(get_fraction(tf[row, col]))
        smatrix.append(srow)
    smatrix = sympy.Matrix(smatrix)
    sympy.print_latex(smatrix)

    gram_obs = control.gram(ss, "o")
    print(f'\[||W||_{"{H_2}"} = {np.sqrt(np.trace(new_B.T @ gram_obs @ new_B))}\]')

    # Simulation
    resp = control.forced_response(ss, X0=[1, 2, 3, 4], T=ts, U=w)
    for indx, z in enumerate(resp.outputs):
        plt.plot(ts, z, label=f'$z_{indx}$')
    plt.xlabel('t, c')
    plt.ylabel('z')
    plt.legend()
    plt.close()

    # Frequency response
    for ni in range(task1_C_2.shape[0]):
        for nj in range(B_1.shape[1]):
            mag, phase, omega = control.bode(tf[ni, nj], omega=np.arange(10 ** -3, 10 ** 3, 10 ** -2), plot=False)
            plt.plot(omega, mag)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('w, rad/s')
    plt.ylabel('Amp')
    plt.close()

    # Singular values plot
    sigma, omega = control.singular_values_plot(ss, plot=False)
    for s in sigma:
        plt.plot(omega, s)
    plt.grid()
    plt.xlabel('$\omega, рад/с$')
    plt.ylabel('$\sigma$')
    plt.close()

    print(f'\[||W||_H_\\{"infty"} = {sigma.max()} \]')

