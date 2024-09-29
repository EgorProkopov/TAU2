import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

from lab12.task1 import get_control_by_state, check_controllability_eigens, check_observability_eigens
from lab12.utils import get_t


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    sympy.init_printing()
    p = sympy.Symbol("p")
    s = sympy.Symbol("s")
    t = sympy.Symbol("t")
    w = sympy.Symbol("w")
    I = sympy.I

    SAVE_PATH = r"/home/egr/TAU/TAU2/lab12/images/task2"

    # %%
    task4_A1 = np.array([
        [1, 0, 0, 0],
        [0, 2, 0, 0],
        [0, 0, 3, 0],
        [0, 0, 0, 4]
    ])
    task4_B1 = np.array([
        [1],
        [4],
        [7],
        [10]
    ])
    task4_B2 = np.array([
        [1, 0, 0, 0],
        [0, 4, 0, 0],
        [0, 0, 7, 0],
        [0, 0, 0, 10]
    ])

    task4_A2 = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, -2, 0]
    ])

    # y
    task4_C1 = np.array([[1, 2, 3, 1]])
    task4_D1 = np.array([[3, 2, 1, 4]])

    # z
    task4_C2 = task4_C1
    task4_D2 = task4_C1

    print(f'\[A_1 = {a2l.to_ltx(task4_A1, print_out=False)}\]')
    print(f'\[A_2 = {a2l.to_ltx(task4_A2, print_out=False)}\]')
    print(f'\[B_1 = {a2l.to_ltx(task4_B1, print_out=False)}\]')
    print(f'\[B_2 = {a2l.to_ltx(task4_B2, print_out=False)}\]')
    print(f'\[C_2 = {a2l.to_ltx(task4_C2, print_out=False)}\]')
    print(f'\[D_2 = {a2l.to_ltx(task4_D2, print_out=False)}\]')
    print(f'\[C_1 = {a2l.to_ltx(task4_C1, print_out=False)}\]')
    print(f'\[D_1 = {a2l.to_ltx(task4_D1, print_out=False)}\]')

    A_obs = np.block([
        [task4_A1, task4_B2],
        [np.zeros((4, 4)), task4_A2]
    ])
    C_obs = np.block([
        [task4_C1, task4_D1]
    ])
    L, _, _ = control.lqe(A_obs, np.eye(8), C_obs, np.eye(8), 1)
    L1, L2 = -L[:4], -L[4:]

    print(f'\[L_1 = {a2l.to_ltx(L1, print_out=False)}\]')
    print(f'\[L_2 = {a2l.to_ltx(L2, print_out=False)}\]')

    Ae = np.block([
        [task4_A1 + L1 @ task4_C1, task4_B2 + L1 @ task4_D1],
        [L2 @ task4_C1, task4_A2 + L2 @ task4_D1]
    ])
    print(f'\[\sigma (A_e) = {a2l.to_ltx(np.linalg.eigvals(Ae), print_out=False)}\]')
    K1, K2, ss = get_control_by_state(task4_A1, task4_A2, task4_B1, task4_B2, task4_C2, task4_D2)
    A_new = np.block([
        [task4_A1 + task4_B1 @ K1, np.zeros((4, 4)), np.zeros((4, 4))],
        [np.zeros((4, 4)), task4_A1 + L1 @ task4_C1, task4_B2 + L1 @ task4_D1],
        [np.zeros((4, 4)), L2 @ task4_C1, task4_A2 + L2 @ task4_D1]
    ])

    B_new = np.block([
        [task4_B2 + task4_B1 @ K2],
        [np.zeros((4, 4))],
        [np.zeros((4, 4))]
    ])

    C_new = np.block([
        [task4_C1, np.zeros((1, 4)), np.zeros((1, 4))],
        [task4_C1, -task4_C1, -task4_D1],
        [task4_C2, np.zeros((1, 4)), np.zeros((1, 4))]
    ])
    D_new = np.block([
        [task4_D1],
        [task4_D1],
        [task4_D2]
    ])

    reg_mat = np.block([
        [task4_A1 + task4_B1 @ K1 + L1 @ task4_C1, task4_B2 + task4_B1 @ K2 + L1 @ task4_D1],
        [L2 @ task4_C2, task4_A2 + L2 @ task4_D2],
    ])
    print(f'\[\sigma (R) = {a2l.to_ltx(np.round(np.linalg.eigvals(reg_mat), 2), print_out=False)}\]')
    print(f'\[\sigma (A_2) = {a2l.to_ltx(np.round(np.linalg.eigvals(task4_A2), 1), print_out=False)}\]')

    ts = get_t(30)
    ss = control.ss(A_new, B_new, C_new, D_new)
    wss = control.ss(task4_A2, np.zeros((4, 1)), np.zeros((1, 4)), 0)
    ws = control.forced_response(wss, X0=[1, 1, 1, 1], T=ts).states
    resp = control.forced_response(ss, T=ts, X0=[1] * A_new.shape[0], U=ws)

    for i in range(4):
        plt.plot(ts, resp.states[i], label=f'$x_{i}$')
    plt.xlabel('t, c')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, 'sim_x.png'))
    plt.close()

    for i in range(4, 8):
        plt.plot(ts, resp.states[i], label='$e_{x_' + str(i % 4) + '}$')
    plt.xlabel('t, c')
    plt.ylabel('$e_x$')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, 'sim_ex.png'))
    plt.close()

    for i in range(8, 12):
        plt.plot(ts, resp.states[i], label='$e_{w_' + str(i % 4) + '}$')
    plt.xlabel('t, c')
    plt.ylabel('$e_w$')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, 'sim_ew.png'))
    plt.close()

    plt.plot(ts, resp.outputs[0])
    plt.xlabel('t, c')
    plt.ylabel('z')
    plt.title("z")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, 'sim_z.png'))
    plt.close()

