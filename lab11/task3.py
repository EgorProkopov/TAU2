import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

from lab11.utils import get_fraction, get_t

SAVE_PATH = r"/home/egr/TAU/TAU2/lab11/images/task3"

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
    [[1, 1],
     [0, 1],
     [0, 0]],
    [[0, 0],
     [1, 0],
     [0, 0]],
])
task1_D_2s = np.array([[[0], [0], [1]], [[1], [0], [1]]])

ts = get_t(25)
w = np.vstack([0.05 * np.sin(ts), 0.01 * np.sin(10 * ts), 0.01 * np.sin(10 * ts)])

omega_i = sympy.Symbol("omega",real=True) * sympy.I

C_2 = task1_C_2s[0]
D_2 = task1_D_2s[0]

def generate_Hinf(a, b_2, c_2, d_2, b_1, gamma):
    R = b_2@np.linalg.inv(d_2.T@d_2)@b_2.T-(gamma**-2)*b_1@b_1.T
    q = scipy.linalg.solve_continuous_are(a,np.identity(R.shape[0]),c_2.T@c_2,np.linalg.inv(R))
    return -np.linalg.inv(d_2.T@d_2)@b_2.T@q

gammas = [2, 4, 6]

for i in range(3):
    print('\n______________________________')
    print(f'-----------VAR_{i+1}, gamma={gammas[i]}----------\n')
    print('\n\subsubsubsection{gamma = ' + str(gammas[i]) + '}')
    # check_controllability_eigens(A, B_2)
    # check_observability_eigens(C_2, A)
    Q = C_2.T @ C_2
    R = D_2.T @ D_2
    K = -generate_Hinf(A, B_2, C_2, D_2, B_1, gammas[i])
    # print(f'\[C_2 = {a2l.to_ltx(C_2, print_out=False)}; D_2 = {a2l.to_ltx(D_2, print_out=False)};\]')
    # print(f'\[C_2^T D_2 = 0: {np.all(C_2.T @ D_2 == 0)}\]')
    # print(f'\[D_2^T D_2 \\text{"{ обратима}"}: {np.linalg.det(D_2.T @ D_2) != 0}\]')
    print(f'\[spec(A-B_2 K) = {np.linalg.eigvals(A - B_2 @ K)}\]')
    # print(f'\[Q = {a2l.to_ltx(S, print_out=False)}\]')
    print(f'\[K = {a2l.to_ltx(K, print_out=False)}\]')

    ss = control.ss(A - B_2 @ K, B_1, C_2 - D_2 @ K, np.zeros((C_2.shape[0], B_1.shape[1])))
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
    plt.savefig(os.path.join(SAVE_PATH, f"sim_{i + 1}.png"))
    plt.close()

    # Frequency response
    for ni in range(C_2.shape[0]):
        for nj in range(B_1.shape[1]):
            mag, phase, omega = control.bode(tf[ni, nj], omega=np.arange(10 ** -3, 10 ** 3, 10 ** -2), plot=False)
            plt.plot(omega, mag)
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('w, rad/s')
    plt.ylabel('Amp')
    plt.savefig(os.path.join(SAVE_PATH, f"amp_{i + 1}.png"))
    plt.close()

    # Singular values plot
    sigma, omega = control.singular_values_plot(ss, plot=False)
    for s in sigma:
        plt.plot(omega, s)
    plt.grid()
    plt.xlabel('$\omega, рад/с$')
    plt.ylabel('$\sigma$')
    plt.savefig(os.path.join(SAVE_PATH, f"singular_{i + 1}.png"))
    plt.close()

    print(f'\[||W||_H_\\{"infty"} = {sigma.max()} \]')

