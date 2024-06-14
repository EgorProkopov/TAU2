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
    response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))

    fig, ax = plt.subplots(4,  figsize=(16, 24))
    fig.suptitle(f"$x_0$: {x0}", fontsize=18)

    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, resp_lin.states[i], label='linear', linewidth=8)
        ax[i].plot(time, response_nonlin.states[i], '--', label='nonlinear', linewidth=8)

        ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend(fontsize=12)

        plt.savefig(f'chapter3_reports/task1/task1_{"_".join([str(x) for x in x0])}.jpg')


def task1(A, B, C, D):
    time = set_time(5)

    gamma = set_gamma([-1.0, -2.0, -3.0, -4.0])
    y = set_y(A, B)
    k, new_spec = get_k_modal(A, B, gamma, y)
    print(f"spec(A + BK): {new_spec}")

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
        draw_and_compare_nonlinear_response_modal(ss_mod, ss_nonlin, x0, time)
        print("\n-------------------------------------------------")


# ------------------------------------------
# task 2
def task2(A, B, C, D):
    time = set_time(5)
    specs = [
        [-1.0, -2.0, -3.0, -4.0],
        [-0.1, -0.2, -0.3, -0.4]
    ]
    gammas = [
        set_gamma(specs[0]),
        set_gamma(specs[1]),
        np.array([
            [-1, -1, 0, 0],
            [1, -1, 0, 0],
            [0, 0, -2, -2],
            [0, 0, 2, -2]
        ])
    ]

    y = set_y(A, B)

    for gamma in gammas:
        k, new_spec = get_k_modal(A, B, gamma, y)
        print(f"new spec:\n {new_spec}")
        x0 = np.array([0.0, 0.0, 1.0, 0.0])
        ss_nonlin = control.NonlinearIOSystem(updfcn_modal, params={"K": k})
        ss_nonlin.set_inputs(2)
        response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
        print(f'gamma:\n {np.linalg.eigvals(gamma)} \n\n'
            f'max a:\n {round(np.abs(response_nonlin.states[0]).max(), 1)}\n'
            f'max phi:\n {round(np.abs(response_nonlin.states[3]).max(), 1)}\n'
            f'max u:\n {round(np.abs(k @ response_nonlin.states).max(), 1)}')
        print("\n")


# ------------------------------------------
# task 3
def get_l_modal(A, C, gamma, y):
    q = cvxpy.Variable(A.shape)
    objective = cvxpy.Minimize(cvxpy.sum_squares(gamma @ q - q @ A - y @ C))
    problem = cvxpy.Problem(objective)
    err = problem.solve()
    l = np.linalg.pinv(q.value) @ y
    new_spec = np.linalg.eigvals(A + l @ C)
    return l, new_spec


def updfcn_modal_observer(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    L = params.get('L', np.zeros((1, 4)))
    C = params.get('C', np.zeros((1, 4)))

    return np.array(
    [
        x[1],
        1 / (M + m*np.sin(x[2])**2) * (-m*l*np.sin(x[2])*x[3]**2 + m*g*np.cos(x[2])*np.sin(x[2]) + u[0] + u[1]*np.cos(x[2])/l),
        x[3],
        1 / (M + m*np.sin(x[2])**2) * (-m*np.cos(x[2])*np.sin(x[2])*x[3]**2 + (M+m)*g*np.sin(x[2])/l + (M+m)*g*u[1]/(m*l**2) + u[0]*np.cos(x[2])/l)
    ]) + L @ (C@x - u)


def draw_and_nonlinear_response_modal_observer(ss_nonlin, ss_nonlin_observer, x0, time, save_path=r"chapter3_reports/task3"):
    response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
    response_nonlin_observer = control.input_output_response(ss_nonlin_observer, T=time, X0=x0, U=np.zeros((2, len(time))))
    error = response_nonlin_observer.states - response_nonlin.states

    fig, ax = plt.subplots(4,  figsize=(16, 24))
    fig.suptitle(f"$x_0$: {x0}", fontsize=18)

    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, response_nonlin.states[i], label='nonlinear', linewidth=8)
        ax[i].plot(time, response_nonlin_observer.states[i], '--', label='observer', linewidth=8)
        ax[i].plot(time, error[i], color='r', label='error', linewidth=8)

        ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend(fontsize=12)

        plt.savefig(f'{save_path}/task1_{"_".join([str(x) for x in x0])}.jpg')


def task3(A, B, C, D):
    time = set_time(5)

    gamma = set_gamma([-1.0, -2.0, -3.0, -4.0])
    y = np.ones((A.shape[0], C.shape[0]))
    l, new_spec = get_l_modal(A, C, gamma, y)

    k, new_spec = get_k_modal(A, B, gamma, set_y(A, B))
    ss_nonlin = control.NonlinearIOSystem(updfcn_modal, params={"K": k})
    ss_nonlin.set_inputs(2)

    ss_nonlin_observer = control.NonlinearIOSystem(updfcn_modal_observer, params={"L": l, 'C': C})
    ss_nonlin_observer.set_inputs(2)

    x0_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    for x0 in x0_list:
        print(*x0)
        print("\n")
        draw_and_nonlinear_response_modal_observer(ss_nonlin, ss_nonlin_observer, x0, time)
        print("\n-------------------------------------------------")


# ------------------------------------------
# task 4
def task4(A, B, C, D):
    time = set_time(5)
    specs = [
        [-1.0, -2.0, -3.0, -4.0],
        [-0.1, -0.2, -0.3, -0.4]
    ]
    gammas = [
        set_gamma(specs[0]),
        set_gamma(specs[1]),
        np.array([
            [-1, -1, 0, 0],
            [1, -1, 0, 0],
            [0, 0, -2, -2],
            [0, 0, 2, -2]
        ])
    ]

    y = np.ones((A.shape[0], C.shape[0]))

    for i, gamma in enumerate(gammas):
        l, new_spec = get_l_modal(A, C, gamma, y)
        print(f"new spec:\n {new_spec}")
        x0 = np.array([0.0, 0.0, 1.0, 0.0])
        k, new_spec = get_k_modal(A, B, gamma, set_y(A, B))
        ss_nonlin = control.NonlinearIOSystem(updfcn_modal, params={"K": k})
        ss_nonlin.set_inputs(2)

        ss_nonlin_observer = control.NonlinearIOSystem(updfcn_modal_observer, params={"L": l, 'C': C})
        ss_nonlin_observer.set_inputs(2)

        print(*x0)
        print("\n")
        draw_and_nonlinear_response_modal_observer(
            ss_nonlin, ss_nonlin_observer, x0, time,
            save_path=f"chapter3_reports/task4/{i+1}"
        )
        print("\n-------------------------------------------------")


# ------------------------------------------
# task 5
def updfcn_modal_full(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    L = params.get('L', np.zeros((1, 4)))
    K = params.get('K', np.zeros((1, 4)))
    C = params.get('C', np.zeros((1, 4)))

    u[0] = (K @ x[4:]).reshape(-1)[0]

    dxh = np.array([
        x[4 + 1],
        1 / (M + m * np.sin(x[4 + 2]) ** 2) * (
                    -m * l * np.sin(x[4 + 2]) * x[4 + 3] ** 2 + m * g * np.cos(x[4 + 2]) * np.sin(x[4 + 2]) + u[0] + u[
                1] * np.cos(x[4 + 2]) / l),
        x[4 + 3],
        1 / (M + m * np.sin(x[4 + 2]) ** 2) * (
                    -m * np.cos(x[4 + 2]) * np.sin(x[4 + 2]) * x[4 + 3] ** 2 + (M + m) * g * np.sin(x[4 + 2]) / l + (
                        M + m) * g * u[1] / (m * l ** 2) + u[0] * np.cos(x[4 + 2]) / l)
    ]) + L @ (C @ x[4:] - C @ x[:4])

    dx = np.array([
        x[1],
        1 / (M + m * np.sin(x[2]) ** 2) * (
                    -m * l * np.sin(x[2]) * x[3] ** 2 + m * g * np.cos(x[2]) * np.sin(x[2]) + u[0] + u[1] * np.cos(
                x[2]) / l),
        x[3],
        1 / (M + m * np.sin(x[2]) ** 2) * (
                    -m * np.cos(x[2]) * np.sin(x[2]) * x[3] ** 2 + (M + m) * g * np.sin(x[2]) / l + (M + m) * g * u[
                1] / (m * l ** 2) + u[0] * np.cos(x[2]) / l)
    ])

    return np.hstack((dx, dxh))


def task5(A, B, C, D):
    y = np.ones((A.shape[0], C.shape[0]))
    gamma = set_gamma([-1.0, -2.0, -3.0, -4.0])
    l, new_spec = get_l_modal(A, C, gamma, y)
    k, new_spec = get_k_modal(A, B, gamma, set_y(A, B))

    time = set_time(5)
    x0 = np.array([0.0, 0.0, 1.0, 0.0])

    ss_nonlin_observer = control.NonlinearIOSystem(updfcn_modal_observer, params={"L": l, 'C': C})
    ss_nonlin_observer.set_inputs(2)
    ss_nonlin_out = control.NonlinearIOSystem(updfcn_modal_full, params={"K": k, "L": l, 'C': C})
    ss_nonlin_out.set_inputs(2)

    response_nonlin_observer = control.input_output_response(ss_nonlin_observer, T=time, X0=x0, U=np.zeros((2, len(time))))
    response_nonlin_out = control.input_output_response(ss_nonlin_out, T=time, X0=np.hstack((x0, x0 + 0.1)), U=C @ response_nonlin_observer.states)

    fig, ax = plt.subplots(4, figsize=(16, 24))
    fig.suptitle(f"$x_0$: {x0}", fontsize=18)

    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, response_nonlin_out.states[i], label='out', linewidth=8)

        ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend(fontsize=12)

        plt.savefig(f'chapter3_reports/task5/task1_{"_".join([str(x) for x in x0])}.jpg')


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    A = get_A()
    B = get_B()
    C = get_C()
    D = get_D()

    print_taks_1 = False
    print_taks_2 = False
    print_taks_3 = False
    print_taks_4 = False
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
