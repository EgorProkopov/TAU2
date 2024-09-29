import matplotlib.pyplot as plt
import numpy as np
import control
import sympy
import os
import scipy
import cvxpy
import sympy.plotting
import array_to_latex as a2l

from lab12.task1 import get_control_by_state
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

    task2_A1 = np.array([
        [1, 0, 0],
        [0, 3, 1],
        [0, -1, 4]
    ])

    task2_B1 = np.ones((3, 1))

    task2_A2 = np.array([
        [0, 1, 0, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 2],
        [0, 0, -2, 0]
    ])

    task2_B2 = np.zeros((task2_A1.shape[0], task2_A2.shape[0]))

    task2_C2 = np.array([[1, 1, 1]])
    task2_D2 = np.array([[1, 1, 1, 1]])

    print(f'\[A_1 = {a2l.to_ltx(task2_A1, print_out=False)}\]')
    print(f'\[A_2 = {a2l.to_ltx(task2_A2, print_out=False)}\]')
    print(f'\[B_1 = {a2l.to_ltx(task2_B1, print_out=False)}\]')
    print(f'\[B_1 = {a2l.to_ltx(task2_B2, print_out=False)}\]')
    print(f'\[C_2 = {a2l.to_ltx(task2_C2, print_out=False)}\]')
    print(f'\[D_2 = {a2l.to_ltx(task2_D2, print_out=False)}\]')

    ts = get_t(30)
    w_ss = control.ss(task2_A2, np.zeros_like(task2_A2), np.zeros_like(task2_A2), np.zeros_like(task2_A2))
    ws = control.forced_response(w_ss, X0=np.ones(task2_A2.shape[0]), T=ts).states

    K1, K2, ss = get_control_by_state(task2_A1, task2_A2, task2_B1, task2_B2, task2_C2, task2_D2)

    resp = control.forced_response(ss, T=ts, X0=np.ones(task2_A1.shape[0]), U=ws)

    plt.plot(ts, resp.outputs[0])
    plt.xlabel('t, c')
    plt.ylabel('z')
    plt.title("z")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, "sim_z.png"))
    plt.close()

    for i in range(task2_A1.shape[0]):
        plt.plot(ts, resp.states[i], label=f'$x_{i}$')
    plt.xlabel('t, c')
    plt.ylabel('x')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, "sim_x.png"))
    plt.close()

    for i in range(task2_A2.shape[0]):
        plt.plot(ts, ws[i], label=f'$w_{i}$')
    plt.xlabel('t, c')
    plt.ylabel('w')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, "sim_w.png"))
    plt.close()

    plt.plot(ts, resp.states[0] + resp.states[2], label='$x_0 + x_2$')
    plt.plot(ts, ws[0] + ws[2], label='$w_0 + w_2$')
    plt.xlabel('t, c')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(SAVE_PATH, "sim_w2.png"))
    plt.close()

