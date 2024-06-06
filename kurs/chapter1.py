import numpy as np


def get_A(m=1, M=10, l=1, g=9.8):
    A = np.array([
        [0, 1, 0, 0],
        [0, 0, m*g / M, 0],
        [0, 0, 0, 1],
        [0, 0, (M+m)*g / (M*l), 0]
    ])
    return A


def get_B(m=1, M=10, l=1, g=9.8):
    B = np.array([
        [0],
        [1 / M],
        [0],
        [1 / (M * l)]
    ])
    return B


def get_C(m=1, M=10, l=1, g=9.8):
    C = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0]
    ])
    return C


def get_D(m=1, M=10, l=1, g=9.8):
    D = np.array([
        [0],
        [1 / (M * l)],
        [0],
        [(M + m) / (M * m * l ** 2)]
    ])
    return D


if __name__ == "__main__":
    print("A:")
    print(get_A())
    print("\n")

    print("B:")
    print(get_B())
    print("\n")

    print("C:")
    print(get_C())
    print("\n")

    print("D:")
    print(get_D())
    print("\n")
