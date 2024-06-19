import numpy as np
import scipy
import sympy
import matplotlib.pyplot as plt

import control
import cvxpy

from kurs.chapter1 import *
from kurs.chapter2 import *
from kurs.chapter3 import *
from kurs.utils import *


# task 1

def get_k_lmi(A, B, alpha):
    P = cvxpy.Variable(A.shape, PSD=True)
    Y = cvxpy.Variable((B.shape[1], B.shape[0]))
    prob = cvxpy.Problem(cvxpy.Maximize(0), [P >> np.eye(4), P @ A.T + A @ P + 2 * alpha * P + Y.T @ B.T + B @ Y << 0])
    prob.solve()
    k = Y.value @ np.linalg.inv(P.value)
    new_spec = np.linalg.eigvals(A + B @ k)
    return k, new_spec


def updfcn_lmi(t, x, u, params):
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


def draw_nonlinear_response_lmi(ss_lin, ss_nonlin, x0, time):
    resp_lin = control.initial_response(ss_lin, T=time, X0=x0)
    response_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))

    fig, ax = plt.subplots(4, figsize=(8, 12))
    fig.suptitle(f"$x_0$: {x0}", fontsize=12)

    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, resp_lin.states[i], label='linear', linewidth=6)
        ax[i].plot(time, response_nonlin.states[i], '--', label='nonlinear', linewidth=6)

        # ax[i].set_xlabel('t')
        ax[i].grid()
        ax[i].legend(fontsize=8)

        plt.savefig(f'chapter4_reports/task1/task1_{"_".join([str(x) for x in x0])}.jpg')


def task1(A, B, C, D):
    alpha = 1
    k, new_spec = get_k_lmi(A, B, alpha)

    x0_list = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]
    time = set_time(5)

    ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
    ss_nonlin.set_inputs(2)

    ss_lin = control.ss(A + B @ k, np.zeros_like(A), np.zeros_like(A), np.zeros_like(A))

    for x0 in x0_list:
        print(*x0)
        print("\n")
        draw_nonlinear_response_lmi(ss_lin, ss_nonlin, x0, time)
        print("\n-------------------------------------------------")


# ------------------------------------------
# task 2
def draw_compare_nonlinear_alphas(A, B, x0, time, alphas):
    save_path = r"chapter4_reports/task2"
    fig, ax = plt.subplots(4, figsize=(8, 12))
    us = []
    for alpha in alphas:
        k, new_spec = get_k_lmi(A, B, alpha)
        ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
        ss_nonlin.set_inputs(2)
        resp_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
        us.append((k @ resp_nonlin.states).reshape(-1))
        for i in range(4):
            ax[i].set_title(f"$x_{i + 1}$")
            ax[i].plot(time, resp_nonlin.states[i], label=f"$\\alpha={alpha}$")
            ax[i].set_xlabel('t')
            ax[i].grid(True)
            ax[i].legend()

        print(
            f'alpha: ${alpha}$  max x: {round(np.abs(resp_nonlin.states[0]).max(), 2)} max phi:  {round(np.abs(resp_nonlin.states[2]).max(), 2)}  max u: {round(np.abs(k @ resp_nonlin.states).max(), 1)} \\\\')
    plt.savefig(f'{save_path}/task4_2_{"_".join([str(x) for x in x0])}.png')
    plt.close()

    plt.title(f"$u(t)$, $x_0=${x0}")
    for i in range(len(alphas)):
        plt.plot(time, us[i], label=f"$\\alpha={alphas[i]}$")
    plt.legend()
    plt.savefig(f'{save_path}/task4_2_u_{"_".join([str(x) for x in x0])}.png')

    plt.close()


def task2(A, B, C, D):
    x0 = [1.0, 1.0, 0.0, 0]
    time = set_time(5)
    alphas = [0.1, 0.5, 1, 2]
    draw_compare_nonlinear_alphas(A, B, x0, time, alphas=alphas)


# ------------------------------------------
# task 3
def get_k_lmi_mu(A, B, alpha, x0, mu=None):
    P = cvxpy.Variable(A.shape, PSD=True)
    Y = cvxpy.Variable((B.shape[1], B.shape[0]))

    if mu is None:
        mu_ = cvxpy.Variable((1, 1))
    else:
        mu_ = mu
    sub1 = cvxpy.bmat([
        [P, x0],
        [x0.T, [[1]]]
    ])

    if mu is None:
        sub2 = cvxpy.bmat([
            [P, Y.T],
            [Y, mu_]
        ])
    else:
        sub2 = cvxpy.bmat([
            [P, Y.T],
            [Y, [[mu_ * mu_]]]
        ])

    prob = cvxpy.Problem(cvxpy.Maximize(0) if mu is not None else cvxpy.Minimize(mu_),
                         [P >> np.eye(4),
                          P @ A.T + A @ P + 2 * alpha * P + Y.T @ B.T + B @ Y << 0,
                          sub1 >> 0, sub2 >> 0])
    res = prob.solve(solver="CLARABEL")

    k = Y.value @ np.linalg.inv(P.value)

    return k, np.sqrt(res)


def draw_compare_nonlinear_alpha_mu(A, B, x0, alpha, time):
    save_path = r"chapter4_reports/task3"

    k, mu = get_k_lmi_mu(A, B, alpha, x0.reshape((4, 1)))

    fig, ax = plt.subplots(4, figsize=(8, 12))
    print(f'K = {k}')
    print(f'spec(A + BK) = {np.linalg.eigvals(A + B @ k)}')
    ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
    ss_nonlin.set_inputs(2)

    ss_lin = control.ss(A + B @ k, np.zeros_like(A), np.zeros_like(A), np.zeros_like(A))

    resp = control.initial_response(ss_lin, T=time, X0=x0)
    resp_nonlin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
    fig.suptitle(f"$\\alpha={alpha}$")
    for i in range(4):
        ax[i].set_title(f"$x_{i + 1}$")
        ax[i].plot(time, resp.states[i], label="linear")
        ax[i].plot(time, resp_nonlin.states[i], '--', label="nonlinear")

        ax[i].set_xlabel('t')
        ax[i].grid(True)
        ax[i].legend()

    plt.savefig(f'{save_path}/task4_3_{alpha}.png')
    plt.show()

    plt.clf()
    plt.title(f"$u(t)$, $\\alpha={alpha}$")
    plt.plot(time, (k @ resp.states).reshape(-1), label="linear")
    plt.plot(time, (k @ resp_nonlin.states).reshape(-1), '--', label="nonlinear")
    plt.legend()
    plt.savefig(f'{save_path}/task4_3_u_{alpha}.png')


def task3(A, B, C, D):
    x0 = np.array([1.0, 0.0, 0.0, 0.0])
    time = set_time(30)
    alphas = [0.1, 0.5, 1.0]

    for alpha in alphas:
        draw_compare_nonlinear_alpha_mu(A, B, x0, alpha, time)


# ------------------------------------------
# task 4
def get_l_lmi(a, c, alpha):
    Q = cvxpy.Variable(a.shape, PSD=True)
    Y = cvxpy.Variable((c.shape[1], c.shape[0]))
    prob = cvxpy.Problem(
        cvxpy.Maximize(0),
        [Q >> np.eye(4), a.T@Q + Q@A + 2*alpha*Q + c.T@Y.T + Y@c << 0]
    )
    prob.solve()
    l = np.linalg.inv(Q.value) @ Y.value
    new_spec = np.linalg.eigvals(a + l @ c)
    return l, new_spec


def draw_nonlinear_lmi_observer(ss_nonlin, x0s, time):
    save_path = r"chapter4_reports/task4"
    fig, ax = plt.subplots(4, figsize=(8, 12))
    for x0 in x0s:
        x0 = np.array(x0)
        resp_non_lin = control.input_output_response(ss_nonlin, T=time, X0=x0, U=np.zeros((2, len(time))))
        resp_non_lin_obs = control.input_output_response(ss_nonlin, T=time, X0=x0 + 0.1, U=C @ resp_non_lin.states)
        err = abs(resp_non_lin_obs.states - resp_non_lin.states)
        for i in range(4):
            ax[i].plot(time, err[i], label=f'{x0}')
            ax[i].set_xlabel('t')
            ax[i].grid()
            ax[i].legend()
    plt.savefig(f'{save_path}/task4_4.png')


def task4(A, B, C, D):
    alpha = 1
    l, new_spec = get_l_lmi(A, C, alpha)
    print(f"l:\n {l}")
    print(f"new_spec: \n{new_spec}")

    k, new_spec = get_k_lmi(A, B, alpha=1)
    time = set_time(10)

    ss_nonlin = control.NonlinearIOSystem(updfcn_lmi, params={"K": k})
    ss_nonlin.set_inputs(2)

    ss_nonlin_obs = control.NonlinearIOSystem(updfcn_modal_observer, params={"L": l, 'C': C})
    ss_nonlin_obs.set_inputs(2)

    x0s = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ]

    draw_nonlinear_lmi_observer(ss_nonlin, x0s, time)
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

    print_taks_1 = False
    print_taks_2 = False
    print_taks_3 = False
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
