#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    angles = []
    norms = []
    with open(sys.argv[1], "r") as data:
        line = data.readline()
        while line:
            angle, norm = line.split(",")
            norm = abs(float(norm))
            angle = float(angle)

            if angle >= 180:
                angle -= 360
            elif angle <= -180:
                angle += 360

            angle = abs(angle)

            angles.append(angle)
            norms.append(norm)

            line = data.readline()

    plt.scatter(angles, norms)

    plt.title("Répartition des différences des normes en fonction des différences des angles")
    ax = plt.gca()
    ax.set_xlabel("Différences des angles")
    ax.set_ylabel("Différences des normes")
    plt.show()


if __name__ == "__main__":
    main()
