#!/usr/bin/env python

import matplotlib.pyplot as plt


def proches(p1, p2):
    rho1, theta1 = p1
    rho2, theta2 = p2

    rho_threshold = 8000
    theta_thresold = 0.2
    return abs(rho1 - rho2) <= rho_threshold and abs(theta1 - theta2) <= theta_thresold


def clusterGen(first, tab):
    cluster = [first]
    for point in tab:
        if proches(first, point):
            cluster.append(point)
    return cluster


def clustering(rho_theta):
    clusters = []

    while len(rho_theta) > 0:
        first = rho_theta[0]
        cluster = clusterGen(first, rho_theta[1:])
        clusters.append(cluster)
        for point in cluster:
            rho_theta.remove(point)

    return clusters


def filterClusters(clusters):
    threshold = 10
    result = []
    for cluster in clusters:
        if len(cluster) >= threshold:
            result.append(cluster)
    return result


def main():
    rho_theta = []

    filename = "houghCoord.txt"
    with open(filename, "r") as f:
        line = f.readline()
        while line:
            rho, theta = line.split(",")
            if 'nan' in rho or 'nan' in theta:
                line = f.readline()
                continue
            rho = float(rho)
            theta = float(theta)
            rho_theta.append((rho, theta))

            line = f.readline()
    rho_theta = list(set(rho_theta))

    clusters = filterClusters(clustering(rho_theta))
    for cluster in clusters:
        X = [pt[0] for pt in cluster]
        Y = [pt[1] for pt in cluster]
        plt.scatter(X, Y)

    plt.show()


if __name__ == "__main__":
    main()
