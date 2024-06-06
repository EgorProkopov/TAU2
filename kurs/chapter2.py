import numpy as np
import scipy
import sympy

import control

from kurs.chapter1 import get_A, get_B, get_C, get_D
from kurs.utils import tf_to_symbolic_fraction


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


def get_Wuy(A, B, C):
    ss_u = control.ss(A, B, C, np.zeros((1, 1)))
    tf_u = control.tfdata(control.ss2tf(ss_u))
    return tf_u


def get_Wfy(A, D, C):
    ss_f = control.ss(A, D, C, np.zeros((2, 1)))
    tf_f = control.tfdata(control.ss2tf(ss_f))
    return tf_f


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    A = get_A()
    B = get_B()
    C = get_C()
    D = get_D()

    print_taks_1 = False
    print_taks_2 = True
    print_taks_3 = True
    print_taks_4 = True

    if print_taks_1:
        print("Eigvals:")
        print(get_eigvals(A))

        print("Eigvectors:")
        print(get_eigvectors(A))

        print("Controllability:")
        print(check_controllability(A, B))

    if print_taks_2:
        print("W_uy:")
        print(tf_to_symbolic_fraction(get_Wuy(A, B, C)[0][0][0], get_Wuy(A, B, C)[1][0][0]))
        print("W_fy:")
        print(tf_to_symbolic_fraction(get_Wfy(A, D, C)[0][0][0], get_Wfy(A, D, C)[1][0][0]))

    if print_taks_3:
        pass

    if print_taks_4:
        pass
