#!/usr/bin/env python

import matplotlib.pyplot as plt
import json


def main():
    filename = "3D.txt"
    
    values = []
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            rho, theta, length = line.split(',')
            values.append((float(rho), float(theta), float(length)))
            line = f.readline()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for rho, theta, length in values:
        ax.scatter(rho, theta, length)

    ax.set_xlabel("rho")
    ax.set_ylabel("theta")
    ax.set_zlabel("length")

    plt.show()


if __name__ == "__main__":
    main()
