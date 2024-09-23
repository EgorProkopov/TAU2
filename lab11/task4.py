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

SAVE_PATH = r"/home/egr/TAU/TAU2/lab11/images/task4"

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

gammas = [4, 8, 12]

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

for i in range(3):
    print('\n______________________________')
    print(f'-----------VAR_{i + 1}, gamma={gammas[i]}----------\n')
    print('\n\subsubsubsection{gamma = ' + str(gammas[i]) + '}')
    K, L, Q = generate_Hinf_obs(A, B_1, B_2, C_1, C_2, D_1, D_2, gammas[i])
    print(f'spec(A-B_2 K) = {np.linalg.eigvals(A - B_2 @ K)}')
    print(f'K = {a2l.to_ltx(K, print_out=False)}')
    print(f'spec(A + B_2K)={np.linalg.eigvals(A + B_2 @ K)}')
    print(f'Q = {a2l.to_ltx(Q, print_out=False)}')
    print(f'L = {a2l.to_ltx(L, print_out=False)}')
    print(f'spec(A + LC_1)={np.linalg.eigvals(A + L @ C_1)}')

    A_new = np.block([
        [A + B_2 @ K, -B_2 @ K],
        [-(L @ D_1 + B_1) * (gammas[i] ** -2) @ B_1.T @ Q, A + L @ C_1 + (L @ D_1 + B_1) * (gammas[i] ** -2) @ B_1.T @ Q]
    ])
    B_new = np.block([
        [B_1],
        [L @ D_1 + B_1]
    ])
    C_new = np.block([C_2 + D_2 @ K, -D_2 @ K])
    D_new = np.zeros((C_2.shape[0], D_1.shape[1]))

    ss = control.ss(A_new, B_new, C_new, D_new)
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
    print(f'||W||_{"{H_2}"} = {np.sqrt(np.trace(B_new.T @ gram_obs @ B_new))}')

    # Simulation
    resp = control.forced_response(ss, X0=[1, 2, 3, 4], T=ts, U=w)
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
    plt.xlabel('$\\omega, рад/с$')
    plt.ylabel('$\\sigma$')
    plt.savefig(os.path.join(SAVE_PATH, f"singular_{i + 1}.png"))
    plt.close()

    print(f'||W||_H_\\{"infty"} = {sigma.max()}')
