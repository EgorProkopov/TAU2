import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

from lab12.utils import get_t


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


def get_control_by_state(A1, A2, B1, B2, C2, D2):
    K, S, E = control.lqr(A1, B1, np.eye(A1.shape[0]) * 1, np.eye(B1.shape[1]) * 5.0)
    K = -K
    P = cvxpy.Variable((A1.shape[0], A2.shape[0]))
    Y = cvxpy.Variable((B1.shape[1], A2.shape[0]))
    prob = cvxpy.Problem(cvxpy.Minimize(0), [C2 @ P + D2 == 0, P @ A2 - A1 @ P == B1 @ Y + B2])
    print('Optimization error: ', prob.solve(solver=cvxpy.ECOS))
    K2 = Y.value - K @ P.value

    print(f'\[K_1 = {a2l.to_ltx(K, print_out=False)}\]')
    print(f'\[spec(A + B_1 K_1) = {a2l.to_ltx(E, print_out=False)}\]')
    print(f'\[K_2 = {a2l.to_ltx(K2, print_out=False)}\]')
    return K, K2, control.ss(A1 + B1 @ K, B2 + B1 @ K2, C2, D2)


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    sympy.init_printing()
    p = sympy.Symbol("p")
    s = sympy.Symbol("s")
    t = sympy.Symbol("t")
    w = sympy.Symbol("w")
    I = sympy.I

    SAVE_PATH = r"/home/egr/TAU/TAU2/lab12/images/task1"

    task1_A1 = np.array([
        [1, 0, 0],
        [0, 3, 1],
        [0, -1, 4]
    ])

    task1_B1 = np.ones((3, 1))

    task1_A2 = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, -2, 0]
    ])

    task1_B2 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    task1_C2 = np.array([[1, 1, 1]])
    task1_D2 = 0

    print(f'\[A_1 = {a2l.to_ltx(task1_A1, print_out=False)}\]')
    print(f'\[A_2 = {a2l.to_ltx(task1_A2, print_out=False)}\]')
    print(f'\[B_1 = {a2l.to_ltx(task1_B1, print_out=False)}\]')
    print(f'\[B_2 = {a2l.to_ltx(task1_B2, print_out=False)}\]')
    print(f'\[C_2 = {a2l.to_ltx(task1_C2, print_out=False)}\]')
    print(f'\[D_2 = {0}\]')

    check_controllability_eigens(task1_A1, task1_B1)
    eig_vals_A1 = np.linalg.eigvals(task1_A1)
    eig_vals_A2 = np.linalg.eigvals(task1_A2)
    print(eig_vals_A1, eig_vals_A2)

    ts = get_t(30)
    w_ss = control.ss(task1_A2, np.zeros_like(task1_A2), np.zeros_like(task1_A2), np.zeros_like(task1_A2))
    ws = control.forced_response(w_ss, X0=np.ones(task1_A2.shape[0]), T=ts).states

    K1, K2, ss = get_control_by_state(task1_A1, task1_A2, task1_B1, task1_B2, task1_C2, task1_D2)
    resp = control.forced_response(ss, T=ts, X0=np.ones(task1_A1.shape[0]), U=ws)

    plt.plot(ts, resp.outputs[0])
    plt.xlabel('t, c')
    plt.ylabel('z')
    plt.title("z")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, f"sim_z.png"))
    plt.close()

    for i in range(task1_A1.shape[0]):
        plt.plot(ts, resp.states[i], label=f'$x_{i}$')
    plt.xlabel('t, c')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, f"sim_x.png"))
    plt.close()

    for i in range(task1_A2.shape[0]):
        plt.plot(ts, ws[i], label=f'$w_{i}$')
    plt.xlabel('t, c')
    plt.ylabel('w')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, f"sim_w.png"))
    plt.close()
