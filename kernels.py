import numpy as np
from scipy import linalg


def polynomial_kernel(x, y, p=5):
    return (1 + np.inner(x, y)) ** p


def gaussian_kernel(x, y, sigma=4.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


def linear_kernel(x1, x2):

    return np.inner(x1, x2)




