import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt

import control
import cvxpy

from kurs.chapter1 import *
from kurs.chapter2 import *
from kurs.utils import *


# task 1
def set_gamma(eigvals: list):
    return np.diag(eigvals)


def set_y(A, B):
    return np.ones((B.shape[1], A.shape[0]))


def get_k_modal(A, B, G, Y):
    P = cvxpy.Variable(A.shape)
    objective = cvxpy.Minimize(cvxpy.sum_squares(A @ P - P @ G - B @ Y))
    problem = cvxpy.Problem(objective)
    err = problem.solve()
    k = - Y @ np.linalg.pinv(P.value)
    new_spec = np.linalg.eigvals(A + B @ k)
    return k, new_spec


def updfcn_modal(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    k = params.get('K', np.zeros((1, 4)))

    u[0] = (k @ x).reshape(-1)[0]

    return np.array(
    [
        x[1],
        1 / (M + m*np.sin(x[2])**2) * (-m*l*np.sin(x[2])*x[3]**2 + m*g*np.cos(x[2])*np.sin(x[2]) + u[0] + u[1]*np.cos(x[2])/l),
        x[3],
        1 / (M + m*np.sin(x[2])**2) * (-m*np.cos(x[2])*np.sin(x[2])*x[3]**2 + (M+m)*g*np.sin(x[2])/l + (M+m)*g*u[1]/(m*l**2) + u[0]*np.cos(x[2])/l)
    ])


def draw_and_compare_nonlinear_response_modal(ss_lin, ss_nonlin, x0, time):
    resp_lin = control.initial_response(ss_lin, T=time, X0=x0)
    resp_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))

    fig, ax = plt.subplots(4,  figsize=(16, 24))
    fig.suptitle(f"$x_0$: {x0}", fontsize=18)

    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, resp_lin.states[i], label='linear', linewidth=8)
        ax[i].plot(time, resp_nonlin.states[i], '---', label='nonlinear', linewidth=8)

        ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend(fontsize=12)

        plt.savefig(f'chapter3_reports/task3_1_{"_".join([str(x) for x in x0])}.jpg')


def task1(A, B, C, D):
    time = set_time(5)

    gamma = set_gamma([-1.0, -2.0, -3.0, -4.0])
    y = set_y(A, B)
    k, new_spec = get_k_modal(A, B, gamma, y)

    ss_mod = control.ss(A + B @ k, np.zeros_like(A), np.zeros_like(A), np.zeros_like(A))

    ss_nonlin = control.NonlinearIOSystem(updfcn_modal, params={"K": k})
    ss_nonlin.set_inputs(2)

    x0_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    for x0 in x0_list:
        print(*x0)
        print("\n")
        draw_and_compare_nonlinear_response(ss_mod, ss_nonlin, x0, time)
        print("\n-------------------------------------------------")

# ------------------------------------------
# task 2
def task2(A, B, C, D):
    pass


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
    print_taks_2 = True
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
