#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = dict()
    with open(sys.argv[1], "r") as datafile:
        line = datafile.readline()
        while line:
            size, sensitivity, specificity = line.split(',')
            size = int(size)
            sensitivity = float(sensitivity)
            specificity = 1 - float(specificity)

            if size not in data:
                data[size] = []
            data[size].append((specificity, sensitivity))

            line = datafile.readline()

    for size in data:
        spec = []
        sens = []
        for sp, se in data[size]:
            spec.append(sp)
            sens.append(se)
        data[size] = (sum(spec) / len(spec), sum(sens) / len(sens))

    Sens = [0]
    Spec = [0]
    for size in data:
        Sens.append(data[size][1])
        Spec.append(data[size][0])
    Sens += [1]
    Spec += [1]


    X = np.linspace(0, 1, 10)
    InvX = [1 - x for x in X]

    print(len(data))

    ax = plt.gca()
    ax.plot(Spec, Sens, marker='o')
    ax.plot(X, X)
    ax.plot(X, InvX)

    k = 1
    for i, j in zip(Spec, Sens):
        if k != 1 and k != 15:
            ax.annotate(str(k), xy=(i, j), color='black', horizontalalignment='right', verticalalignment='bottom')
        k += 2
    ax.set_xlabel("Sensitivité")
    ax.set_ylabel("Spécificité")
    # plt.plot(Spec, Spec)

    plt.show()


if __name__ == "__main__":
    main()
