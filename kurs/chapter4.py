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
def task1(A, B, C, D):
    pass


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
