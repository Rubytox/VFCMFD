#!/usr/bin/env python

import sys
import math as m

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def hough(theta, x, y):
    return x * m.cos(theta) + y * m.sin(theta)


def draw():
    abscisses = np.deg2rad(np.linspace(-90, 90, 100))

    image = Image.open(sys.argv[1])
    data = np.asarray(image)

    R = np.ceil((data.shape[0]**2 + data.shape[1]**2)**0.5)
    # Dimensions of matrix : [-R, R] x [0, 180]

    cos_theta = np.cos(abscisses)
    sin_theta = np.sin(abscisses)
    nb_abscisses = len(abscisses)
    accumulator = np.zeros((int(2 * R), nb_abscisses), dtype=np.uint64)

    ylist, xlist = np.nonzero(data)

    for x, y in zip(xlist, ylist):
        for index in range(nb_abscisses):
            rho = round(x * cos_theta[index] + y * sin_theta[index]) + R
            accumulator[int(rho), index] += 1

    plt.imshow(accumulator)
    plt.show()


if __name__ == "__main__":
    draw()
