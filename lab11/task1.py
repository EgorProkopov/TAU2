import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

np.set_printoptions(precision=2)
sympy.init_printing()
p = sympy.Symbol("p")
s = sympy.Symbol("s")
t = sympy.Symbol("t")
w = sympy.Symbol("w")
I = sympy.I

def get_t(end_t = 10, dt=0.001, start_t = 0):
    return np.linspace(start_t, end_t, int(end_t / dt))


def get_controllability_matrix(A, B):
    ctrb_m = np.hstack((B, *[(np.linalg.matrix_power(A, i)) @ B for i in range(1, A.shape[0])]))
    assert np.allclose(control.ctrb(A, B), ctrb_m), 'Smth wrong'
    return ctrb_m


def get_observability_matrix(A, C):
    obsv_m = np.vstack((C, *[C @ np.linalg.matrix_power(A, i) for i in range(1, A.shape[0])]))
    assert np.allclose(control.obsv(A, C), obsv_m), 'Smth wrong'
    return obsv_m


def check_controllability_eigens(A, B):
    eig_vals = np.linalg.eigvals(A)
    print(f'Eigen values of A:')
    for val in eig_vals:
        print(
            f"   {np.array([val])}: {'controllable' if np.linalg.matrix_rank(np.hstack(((A - val * np.eye(A.shape[0])), B))) == A.shape[0] else 'not controllable'}")


def check_observability_eigens(C, A):
    eig_vals = np.linalg.eigvals(A)
    print(f'Eigen values of A:')
    for val in eig_vals:
        print(
            f"   {np.array([val])}: {'observable' if np.linalg.matrix_rank(np.vstack(((A - val * np.eye(A.shape[0])), C))) == A.shape[0] else 'not observable'}")

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

def get_fraction(tf):
    num, den = tf.num[0][0], tf.den[0][0]
    den_ = sum((0 if abs(co) < 1e-3 else co) * omega_i**i for i, co in enumerate(reversed(den)))
    num_ = sum((0 if abs(co) < 1e-3 else co) * omega_i**i for i, co in enumerate(reversed(num)))
    return num_ / den_


for i in range(2):
    print('\n______________________________')
    task1_C_2 = task1_C_2s[i]
    task1_D_2 = task1_D_2s[i]
    check_controllability_eigens(A, B_2)
    check_observability_eigens(task1_C_2, A)
    Q = task1_C_2.T @ task1_C_2
    R = task1_D_2.T @ task1_D_2
    K, S, E = control.lqr(A, B_2, Q, R)
    print(f'\[C_2 = {a2l.to_ltx(task1_C_2, print_out=False)}; D_2 = {a2l.to_ltx(task1_D_2, print_out=False)};\]')
    print(f'\[C_2^T D_2 = 0: {np.all(task1_C_2.T @ task1_D_2 == 0)}\]')
    print(f'\[D_2^T D_2 \\text{"{ обратима}"}: {np.linalg.det(task1_D_2.T @ task1_D_2) != 0}\]')
    print(f'\[spec(A-B_2 K) = {a2l.to_ltx(E, print_out=False)}\]')
    print(f'\[Q = {a2l.to_ltx(S, print_out=False)}\]')
    print(f'\[K = {a2l.to_ltx(K, print_out=False)}\]')

    ss = control.ss(A - B_2 @ K, B_1, task1_C_2 - task1_D_2 @ K, np.zeros((task1_C_2.shape[0], B_1.shape[1])))
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
    print(f'\[||W||_{"{H_2}"} = {np.sqrt(np.trace(B_1.T @ gram_obs @ B_1))}\]')

    # Simulation
    resp = control.forced_response(ss, X0=np.ones((2, 1)), T=ts, U=w)
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
