import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt

import control
import cvxpy

from kurs.chapter1 import *
from kurs.chapter2 import *
from kurs.chapter3 import *
from kurs.chapter4 import *
from kurs.utils import *


# task 1
def get_k_lqr(A, B, Q, R):
    k, _, _ = control.lqr(A, B, Q, R)
    new_spec = np.linalg.eigvals(A + B @ k)
    return k, new_spec


def draw_nonlinear_lqr(k, x0, time, q, r, save_path):
    ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
    ss_nonlin.set_inputs(2)

    response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0)

    xs = response_nonlin.states
    us = (- k @ xs).reshape(-1)

    fig, ax = plt.subplots(4, figsize=(8, 12))
    for i, state in enumerate(response_nonlin.states):
        ax[i].plot(time, state, label=f'Q = ${q}I_Q$; R = ${r}I_R$')
        ax[i].legend()
        ax[i].set_xlabel('t')
        ax[i].set_ylabel('$x_i$')
        ax[i].grid()

    fig_u, ax_u = plt.subplots(1, figsize=(8, 12))
    ax_u.plot(time, us, label=f'Q = ${q}I_Q$; R = ${r}I_R$')
    ax_u.legend()
    ax_u.set_xlabel('t')
    ax_u.set_ylabel('$x_i$')
    ax_u.grid()

    fig.savefig(f'{save_path}/task{q}_{r}_states.png')
    fig_u.savefig(f'{save_path}/task5_{q}_{r}_us.png')


def task1(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(20)

    q = 1.0
    r = 1.0
    Q = np.diag(np.ones((A.shape[0]))) * q
    R = np.diag(np.ones((B.shape[1]))) * r

    k, new_spec = get_k_lqr(A, B, Q, R)
    k = -k
    # контрол выдает отрицательное значение матрицы k

    print(f"K: \n{k}")
    print(f"new_spec: \n{new_spec}")

    save_path = r"chapter5_reports/task1"
    draw_nonlinear_lqr(k, x0, time, q, r, save_path)


# ------------------------------------------
# task 2
def task2(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(10)
    qs = [1.0, 1.0, 10.0, 10.0]
    rs = [1.0, 10.0, 0.1, 10.0]

    for q, r in zip(qs, rs):
        Q = np.diag(np.ones((A.shape[0]))) * q
        R = np.diag(np.ones((B.shape[1]))) * r

        k, new_spec = get_k_lqr(A, B, Q, R)
        k = -k

        print(f"K: \n{k}")
        print(f"new_spec: \n{new_spec}")

        save_path = r"chapter5_reports/task2"
        draw_nonlinear_lqr(k, x0, time, q, r, save_path)


# ------------------------------------------
# task 3
def task3(A, B, C, D):
    pass


# ------------------------------------------
# task 4
def task4(A, B, C, D):
    pass


# ------------------------------------------
# task 5
def task5(A, B, C, D):
    pass


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    A = get_A()
    B = get_B()
    C = get_C()
    D = get_D()

    print_taks_1 = True
    print_taks_2 = False
    print_taks_3 = True
    print_taks_4 = True
    print_taks_5 = True

    if print_taks_1:
        task1(A, B, C, D)

    if print_taks_2:
        task2(A, B, C, D)

    if print_taks_3:
        task3(A, B, C, D)

    if print_taks_4:
        task4(A, B, C, D)

    if print_taks_5:
        task5(A, B, C, D)
