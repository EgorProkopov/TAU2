import sympy
import numpy as np


def tf_to_symbolic_fraction(num, den):
    x = sympy.symbols('s')

    length_num = len(num)
    length_den = len(den)

    sym_num, sym_den = 0, 0

    counter = 1
    for i in range(length_num):
        sym_num += round(num[i], 2) * (x ** (length_num - counter))
        counter += 1

    counter = 1
    for i in range(length_den):
        sym_den += round(den[i], 2) * (x ** (length_den - counter))
        counter += 1
    return sym_num / sym_den


def set_time(end_t=10, dt=0.001, start_t=0):
    return np.linspace(start_t, end_t, int(end_t / dt))
