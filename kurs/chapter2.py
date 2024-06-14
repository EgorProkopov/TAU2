import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt

import control

from kurs.chapter1 import get_system_params, get_A, get_B, get_C, get_D
from kurs.utils import tf_to_symbolic_fraction, set_time


# Task 1
def get_eigvals(matrix):
    return np.linalg.eigvals(matrix)


def get_eigvectors(matrix):
    return np.linalg.eig(matrix)[1]


def get_controllability_matrix(A, B):
    ctrb_m = np.hstack((B, *[(np.linalg.matrix_power(A, i)) @ B for i in range(1, A.shape[0])]))
    return ctrb_m


def check_controllability(A, B):
    U = get_controllability_matrix(A, B)
    print(f'U: \n{U}')
    print(f'Rank U = {np.linalg.matrix_rank(U)}')
    eig_vals = np.linalg.eigvals(A)
    print(f'Controllability of eigvalues:')
    for val in eig_vals:
        print(f"{val}: {'controllable' if np.linalg.matrix_rank(np.hstack(((A - val * np.eye(A.shape[0])), B))) == A.shape[0] else 'not controllable'}")


def task1(A, B, C, D):
    print("Eigvals:")
    print(get_eigvals(A))

    print("Eigvectors:")
    print(get_eigvectors(A))

    print("Controllability:")
    print(check_controllability(A, B))


# -----------------------
# Task 2
def get_wuy(A, B, C):
    ss_u = control.ss(A, B, C, np.zeros((2, 1)))
    tf_u = control.tfdata(control.ss2tf(ss_u))
    return tf_u


def get_wfy(A, D, C):
    ss_f = control.ss(A, D, C, np.zeros((2, 1)))
    tf_f = control.tfdata(control.ss2tf(ss_f))
    return tf_f


def task2(A, B, C, D):
    print("W_uy:")
    print(tf_to_symbolic_fraction(
        get_wuy(A, B, C)[0][0][0],
        get_wuy(A, B, C)[1][0][0])
    )
    print(tf_to_symbolic_fraction(
        get_wuy(A, B, C)[0][1][0],
        get_wuy(A, B, C)[1][1][0])
    )

    print("W_fy:")
    print(tf_to_symbolic_fraction(
        get_wfy(A, D, C)[0][0][0],
        get_wfy(A, D, C)[1][0][0])
    )
    print(tf_to_symbolic_fraction(
        get_wfy(A, D, C)[0][1][0],
        get_wfy(A, D, C)[1][1][0])
    )


# -----------------------
# Task3
def draw_linear_response(ss, x0, time):
    response = control.initial_response(ss, T=time, X0=x0)
    fig, ax = plt.subplots(4, figsize=(16, 32))
    for i in range(4):
        ax[i].set_title(f"$x_{i+1}$")
        ax[i].plot(time, response.states[i])
        ax[i].set_xlabel('t')
        ax[i].grid()


def task3(A, B, C, D):
    ss_u = control.ss(A, B, C, np.zeros((2, 1)))
    time = set_time(end_t=3)
    x0_list = [
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ]
    for x0 in x0_list:
        print(*x0)
        print("\n")
        draw_linear_response(ss_u, x0, time)
        print("\n-------------------------------------------------")


# -----------------------
# Task4
def updfcn(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    return np.array([
        x[1],
        1 / (M + m*np.sin(x[2])**2) * (-m*l*np.sin(x[2])*x[3]**2 + m*g*np.cos(x[2])*np.sin(x[2]) + u[0] + u[1]*np.cos(x[2])/l),
        x[3],
        1 / (M + m*np.sin(x[2])**2) * (-m*np.cos(x[2])*np.sin(x[2])*x[3]**2 + (M+m)*g*np.sin(x[2])/l + (M+m)*g*u[1]/(m*l**2) + u[0]*np.cos(x[2])/l)
    ])


def draw_and_compare_nonlinear_response(ss_lin, ss_nonlin, x0, time):
    resp = control.initial_response(ss_lin, T=time, X0=x0)
    fig, ax = plt.subplots(4, figsize=(16, 32))

    resp_non_lin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, resp.states[i], label='lin')
        ax[i].plot(time, resp_non_lin.states[i], label='nonlin')

        ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend()


def task4(A, B, C, D):
    ss_lin = control.ss(A, B, C, np.zeros((2, 1)))
    ss_nonlin = control.NonlinearIOSystem(updfcn)
    ss_nonlin.set_inputs(2)

    time = set_time(end_t=3)
    x0_list = [
        [0.1, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.1]
    ]
    for x0 in x0_list:
        print(*x0)
        print("\n")
        draw_and_compare_nonlinear_response(ss_lin, ss_nonlin, x0, time)
        print("\n-------------------------------------------------")


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

    if print_taks_1:
        task1(A, B, C, D)

    if print_taks_2:
        task2(A, B, C, D)

    if print_taks_3:
        task3(A, B, C, D)

    if print_taks_4:
        task4(A, B, C, D)
