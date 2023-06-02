#!/usr/bin/env python

import matplotlib.pyplot as plt
import json


def main():
    clusters = []

    filename = "clusters.txt"
    with open(filename, "r") as f:
        nbClusters = int(f.readline())
        for i in range(nbClusters):
            clusters.append([])
        line = f.readline()
        while line:
            nb, rho, theta, _ = line.split(',')
            clusters[int(nb)].append((float(rho), float(theta)))
            line = f.readline()

    lines = []
    with open("lines.txt", "r") as f:
        line = f.readline()
        while line:
            rho, theta, _ = line.split(',')
            if 'nan' not in rho and 'nan' not in theta:
                lines.append((float(rho), float(theta)))
            line = f.readline()
    X = []
    Y = []
    for line in lines:
        rho = line[0]
        theta = line[1]
        X.append(rho)
        Y.append(theta)
    plt.scatter(X, Y, c="#000000")

    for cluster in clusters:
        X = [t[0] for t in cluster]
        Y = [t[1] for t in cluster]
        plt.scatter(X, Y)


    plt.show()


if __name__ == "__main__":
    main()
