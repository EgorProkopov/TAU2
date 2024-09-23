import sympy
import numpy as np

def get_fraction(tf):
    omega_i = sympy.Symbol("omega", real=True) * sympy.I
    num, den = tf.num[0][0], tf.den[0][0]
    den_ = sum((0 if abs(co) < 1e-3 else co) * omega_i**i for i, co in enumerate(reversed(den)))
    num_ = sum((0 if abs(co) < 1e-3 else co) * omega_i**i for i, co in enumerate(reversed(num)))
    return num_ / den_

def get_t(end_t = 10, dt=0.001, start_t = 0):
    return np.linspace(start_t, end_t, int(end_t / dt))