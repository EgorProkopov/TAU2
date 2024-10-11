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
    new_spec = np.linalg.eigvals(A + B @ (-k))
    return k, new_spec


def draw_nonlinear_lqr(k, x0, time, q, r, save_path):
    ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
    ss_nonlin.set_inputs(2)

    response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0)

    xs = response_nonlin.states
    us = (- k @ xs).reshape(-1)

    fig, ax = plt.subplots(4, figsize=(16, 24))
    for i, state in enumerate(response_nonlin.states):
        ax[i].plot(time, state, label=f'Q = ${q}I_Q$; R = ${r}I_R$', linewidth=8)
        ax[i].legend(fontsize=24)
        ax[i].set_xlabel('t', fontsize=24)
        ax[i].set_ylabel(f"$x_{i+1}$", fontsize=24)
        ax[i].grid()

    fig_u, ax_u = plt.subplots(1, figsize=(16, 24))
    ax_u.plot(time, us, label=f'Q = ${q}I_Q$; R = ${r}I_R$', linewidth=8)
    ax_u.legend(fontsize=24)
    ax_u.set_xlabel('t', fontsize=24)
    ax_u.grid()

    fig.savefig(f'{save_path}/task{x0}{q}_{r}_states.png')
    fig_u.savefig(f'{save_path}/task{x0}_{q}_{r}_us.png')


def task1(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(20)

    # q = 1.0
    # r = 1.0

    q = 1.0
    r = 1.0

    Q = np.diag(np.ones((A.shape[0]))) * q
    R = np.diag(np.ones((B.shape[1]))) * r

    k, new_spec = get_k_lqr(A, B, Q, R)
    k = -k
    # контрол выдает отрицательное значение матрицы k

    print(f"K: \n{k}")
    print(f"new_spec: \n{new_spec}")

    save_path = f"chapter5_reports//task1"
    draw_nonlinear_lqr(k, x0, time, q, r, save_path)


# ------------------------------------------
# task 2
def task2(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(75)
    # qs = [10.0, 1.0, 10.0, 0.1]
    # rs = [1.0, 10.0, 0.1, 10.0]

    qs = [0.000001]
    rs = [100000]

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
def updfcn_lqr(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]


    K = params.get('K', np.zeros((1, 4)))
    u[0] = (K @ x).reshape(-1)[0]
    return np.array([
        x[1],
        1 / (M + m*np.sin(x[2])**2) * (-m*l*np.sin(x[2])*x[3]**2 + m*g*np.cos(x[2])*np.sin(x[2]) + u[0] + u[1]*np.cos(x[2])/l),
        x[3],
        1 / (M + m*np.sin(x[2])**2) * (-m*np.cos(x[2])*np.sin(x[2])*x[3]**2 + (M+m)*g*np.sin(x[2])/l + (M+m)*g*u[1]/(m*l**2) + u[0]*np.cos(x[2])/l)
    ])


def updfcn_k_f(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]


    K = params.get('K', np.zeros((1, 4)))
    u[0] = (K @ x).reshape(-1)[0]

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

    D = params.get('D', np.zeros((1, 4)))
    std_f = params.get('std_f', 1)
    noise = np.random.normal(0, std_f, (1, 1))
    f = (D @ noise).reshape(-1)
    dx += f

    return dx


def updfcn_lqe(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    L = params.get('L', np.zeros((1, 4)))
    C = params.get('C', np.zeros((1, 4)))

    return np.array([
        x[1],
        1 / (M + m * np.sin(x[2]) ** 2) * (
                    -m * l * np.sin(x[2]) * x[3] ** 2 + m * g * np.cos(x[2]) * np.sin(x[2]) + u[0] + u[1] * np.cos(
                x[2]) / l),
        x[3],
        1 / (M + m * np.sin(x[2]) ** 2) * (
                    -m * np.cos(x[2]) * np.sin(x[2]) * x[3] ** 2 + (M + m) * g * np.sin(x[2]) / l + (M + m) * g * u[
                1] / (m * l ** 2) + u[0] * np.cos(x[2]) / l)
    ]) + L @ (C @ x - u)


def get_kalman(A, C, q, r):
    Q, R = np.diag(q)**2, np.diag(r)**2
    l, p, _ = control.lqe(A, np.eye(4), C, Q, R)
    l = -l
    new_spec = np.linalg.eigvals(A + l @ C)
    return l, new_spec


def draw_nonlinear_kalman(C, D, k, l, x0, time, q, r, save_path):
    ss_nonlin_f = control.NonlinearIOSystem(updfcn_modal, params={"D": D, "std_f": q, 'K': k})
    ss_nonlin_f.set_inputs(2)
    resp_nonlin_f = control.input_output_response(ss_nonlin_f, T=time, X0=x0, U=np.zeros((2, len(time))))

    ss_nonlin_lqe = control.NonlinearIOSystem(updfcn_lqe, params={"L": l, "C": C})
    ss_nonlin_lqe.set_inputs(2)
    resp_nonlin_lqe = control.input_output_response(ss_nonlin_lqe, T=time, X0=[1.1, 0.1, 0.1, 0.1], U=C @ resp_nonlin_f.states)
    err = abs(resp_nonlin_lqe.states - resp_nonlin_f.states)

    fig, ax = plt.subplots(4, figsize=(16, 24))
    for i in range(4):
        ax[i].plot(time, err[i], label=f'$e_{i} = |x_i - \\hat x_i|$', linewidth=8)
        ax[i].set_xlabel('t', fontsize=24)
        ax[i].set_ylabel(f"$x_{i + 1}$", fontsize=24)
        ax[i].grid()
        ax[i].legend(fontsize=24)
    fig.savefig(f'{save_path}/task_34_{q}_{r}.png')


def task3(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(20)

    q = [1.0] * 4
    r = [1.0] * 2

    l, new_spec = get_kalman(A, C, q, r)
    print(f"L: \n{l}")
    print(f"new_spec: \n{new_spec}")

    q_k = 1.0
    r_k = 1.0
    Q_k = np.diag(np.ones((A.shape[0]))) * q_k
    R_k = np.diag(np.ones((B.shape[1]))) * r_k

    k, new_spec = get_k_lqr(A, B, Q_k, R_k)
    k = -k

    save_path = r"chapter5_reports/task3"
    draw_nonlinear_kalman(C, D, k, l, x0, time, q_k, r_k, save_path)


# ------------------------------------------
# task 4
def task4(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(20)

    q = [1.0] * 4
    r = [0.1] * 2

    l, new_spec = get_kalman(A, C, q, r)
    print(f"L: \n{l}")
    print(f"new_spec: \n{new_spec}")

    q_k = 1.0
    r_k = 1.0
    Q_k = np.diag(np.ones((A.shape[0]))) * q_k
    R_k = np.diag(np.ones((B.shape[1]))) * r_k

    k, new_spec_k = get_k_lqr(A, B, Q_k, R_k)
    k = -k

    print(f"k: \n{k}")
    print(f"new_spec_k: \n{new_spec_k}")

    save_path = r"chapter5_reports/task4"

    f, xi = np.random.normal(0, q, (len(time), 4)), np.random.normal(0, r, (len(time), 2))
    u = np.hstack([f, xi])

    new_A = np.block([[A + B @ k, -B @ k], [np.zeros((A.shape[0], k.shape[1] * 2 - C.shape[1])), A + l @ C]])
    new_B = np.block([[np.diag([1] * 4), np.zeros((4, 2))], [np.diag([1] * 4), l]])

    ss_lin = control.ss(new_A, new_B, 0 * new_A, 0 * new_B)
    response = control.forced_response(ss_lin, T=time, U=u.T, X0=np.array([1.0, 0, 0.0, 0.0, 1.1, 0.1, 0.1, 0.1]))

    fig, axs = plt.subplots(4, figsize=(16, 24))
    fig.suptitle("Состояние", fontsize=48)
    fig_e, axs_e = plt.subplots(4, figsize=(16, 24))
    fig_e.suptitle("Ошибка", fontsize=48)
    for i, state in enumerate(response.states[:4]):
        axs[i].plot(time, state, label=f'$ x_{i+1} $', linewidth=8)
        axs_e[i].plot(time, response.states[4 + i], label=f'$ e_{i+1} $', linewidth=8)

    for i in range(4):
        axs[i].set_xlabel(f"t", fontsize=30)
        axs[i].set_ylabel(f"e_{i+1}", fontsize=30)
        axs[i].grid(True)
        axs[i].legend(fontsize=30)

    for i in range(4):
        axs_e[i].set_xlabel(f"t", fontsize=30)
        axs[i].set_ylabel(f"e_{i+1}", fontsize=30)
        axs_e[i].grid(True)
        axs_e[i].legend(fontsize=30)

    fig.savefig(f'{save_path}/task5_lqg_lin_x.jpg')
    fig_e.savefig(f'{save_path}/task5_lqg_lin_err.jpg')


# ------------------------------------------
# task 5
def updfcn_lqg(t, x, u, params):
    system_params = get_system_params()

    m = system_params["m"]
    M = system_params["M"]
    l = system_params["l"]
    g = system_params["g"]

    L = params.get('L', np.zeros((1, 4)))
    K = params.get('K', np.zeros((1, 4)))
    C = params.get('C', np.zeros((1, 4)))
    D = params.get('D', np.zeros((1, 4)))
    std_f = params.get('std_f', 1)
    noise = np.random.normal(0, std_f, (1, 1))
    f = (D @ noise).reshape(-1)

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
    ]) + f

    return np.hstack((dx, dxh))


def task5(A, B, C, D):
    x0 = [1.0, 0, 0.0, 0.0]
    time = set_time(30)

    q_l = [10.0] * 4
    r_l = [1.0] * 2

    l, new_spec = get_kalman(A, C, q_l, r_l)
    print(f"L: \n{l}")
    print(f"new_spec: \n{new_spec}")

    q_k = 10.0
    r_k = 0.1
    Q_k = np.diag(np.ones((A.shape[0]))) * q_k
    R_k = np.diag(np.ones((B.shape[1]))) * r_k

    k, new_spec_k = get_k_lqr(A, B, Q_k, R_k)
    k = -k

    print(f"k: \n{k}")
    print(f"new_spec_k: \n{new_spec_k}")

    save_path = r"chapter5_reports/task5"

    ss_nonlin_lqg = control.NonlinearIOSystem(updfcn_lqg, params={"L": l, "C": C, "D": D, "std_f": q_l[0], 'K': k})
    ss_nonlin_lqg.set_inputs(2)

    resp_nonlin_LQG = control.input_output_response(
        ss_nonlin_lqg, T=time, X0=np.hstack((x0, [1.1, 0.1, 0.1, 0.1])),
        U=np.zeros((2, len(time)))
    )

    fig, axs = plt.subplots(4, figsize=(16, 24))
    fig.suptitle("Состояние", fontsize=48)
    fig_e, axs_e = plt.subplots(4, figsize=(16, 24))
    fig_e.suptitle("Ошибка", fontsize=48)

    for i, state in enumerate(resp_nonlin_LQG.states[:4]):
        axs[i].plot(time, resp_nonlin_LQG.states[i], label=f'$ x_{i+1} $')
        axs_e[i].plot(time, abs(resp_nonlin_LQG.states[i]-resp_nonlin_LQG.states[i + 4]), label=f'$ e_{i+1} $')

    for i in range(4):
        axs[i].set_xlabel(f"t", fontsize=30)
        axs[i].set_ylabel(f"x_{i + 1}", fontsize=30)
        axs[i].grid(True)
        axs[i].legend(fontsize=30)

    for i in range(4):
        axs_e[i].set_xlabel(f"t, [c]", fontsize=30)
        axs[i].set_ylabel(f"e_{i + 1}", fontsize=30)
        axs_e[i].grid(True)
        axs_e[i].legend(fontsize=30)

    fig.savefig(f'{save_path}/task5_LQG_nonlin_x{q_k}_{q_l}.jpg')
    fig_e.savefig(f'{save_path}/task5_LQG_nonlin_err{q_k}_{q_l}.jpg')


if __name__ == "__main__":
    font = {
        'weight': 'bold',
        'size': 24
    }
    matplotlib.rc('font', **font)
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
