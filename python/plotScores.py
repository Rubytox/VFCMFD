#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt


def main():
    data = [(0, 0)]
    with open(sys.argv[1], "r") as datafile:
        line = datafile.readline()
        while line:
            size, sensitivity, specificity = line.split(',')
            sensitivity = float(sensitivity)
            specificity = 1 - float(specificity)

            data.append((specificity, sensitivity))

            line = datafile.readline()
    data += [(1, 1)]

    Sens = [el[1] for el in data]
    Spec = [el[0] for el in data]

    InvSpec = [1 - el for el in Spec]

    ax = plt.gca()
    plt.title(sys.argv[1])
    ax.plot(Spec, Sens)
    ax.plot(Spec, Spec)
    ax.plot(Spec, InvSpec)
    plt.show()


if __name__ == "__main__":
    main()
