#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    angles = []
    with open(sys.argv[1], "r") as data:
        line = data.readline()
        while line:
            angle = float(line)
            angles.append(angle)

            line = data.readline()

    hist, bin_edges = np.histogram(angles, bins=50, normed=False)

    width = bin_edges[1] - bin_edges[0]
    plt.bar(bin_edges[:-1], hist, width=width, color='red', alpha=0.5)

    plt.title("Répartition des différences des normes")
    plt.show()


if __name__ == "__main__":
    main()
